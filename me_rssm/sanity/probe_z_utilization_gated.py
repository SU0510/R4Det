"""Counterfactual z_t utilization probe for GATED RSSM variants (MCI).

Read-only diagnostic. Same scoring, sample selection and output schema as
``tools/diagnose_z_utilization.py`` (imported wholesale), with ONE
difference: the intervention is applied to the FINAL latent ``z_t`` — after
the correct gate, before decoder/readout/state write — via the class's
``_z_intervention`` hook point instead of rewriting ``prior_mu`` /
``posterior_mu`` outputs.

Why the original tool cannot probe MCI: it makes ``mu_q`` exactly the
desired value by rewriting the prior/posterior heads, which is bitwise for
MAI (``z_t == mu_q``) but for MCI the correct gate re-mixes the prior back
in (``z_t = z_sel + dev*(z_sel - mu_p)``), so the tool's exact-propagation
assertion fires even though the intervention is ~93% effective. The gated
probe asks the scientifically right question directly: what does the head
see if the latent it consumes carries nothing (zero_z) or another sample's
latent (shuffle_z)?

Usage (mirrors the original tool):
  python me_rssm/sanity/probe_z_utilization_gated.py \
    --config me_rssm/configs/..._mci.py \
    --checkpoint <work_dir>/iter_450.pth \
    --start-index 3 --limit 4 --batch-size 4 --shuffle-offset 2 \
    --output <work_dir>/z_util_gated_w3.json
"""

import argparse
import importlib.util
import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'dz', os.path.join(_REPO, 'tools', 'diagnose_z_utilization.py'))
dz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dz)


class ZInterventionGated:
    """Zero/replace/prior-only the FINAL z_t on the current frame only."""

    def __init__(self, fusion, mode, n_frames, replacement_z=None):
        assert mode in ('zero_z', 'replace_z', 'prior_only'), mode
        self.fusion = fusion
        self.mode = mode
        self.seq_len = n_frames
        self.replacement_z = replacement_z
        self.call_idx = 0
        self.hooks = []
        self.last_z = None
        self.last_z_norm = None
        self.original_forward = fusion.forward

    def _apply(self, z):
        if self.mode == 'zero_z':
            return torch.zeros_like(z)
        if self.replacement_z is None:
            raise ValueError('replace_z requires a replacement tensor')
        if self.replacement_z.shape != z.shape:
            raise ValueError(
                'replacement z shape mismatch: '
                f'{tuple(self.replacement_z.shape)} vs {tuple(z.shape)}')
        return self.replacement_z.to(device=z.device, dtype=z.dtype)

    def _wrapped_forward(self, feat, velocity=None, use_posterior=True,
                         deterministic=True, detach_state=True):
        self.call_idx += 1
        is_current = self.call_idx == self.seq_len - 1
        active = is_current and self.mode in ('zero_z', 'replace_z')
        if is_current and self.mode == 'prior_only':
            use_posterior = False
        prev = self.fusion._z_intervention
        self.fusion._z_intervention = self._apply if active else None
        try:
            out = self.original_forward(
                feat, velocity=velocity, use_posterior=use_posterior,
                deterministic=deterministic, detach_state=detach_state)
        finally:
            self.fusion._z_intervention = prev
        if active:
            # The hook point replaces z_t inside the forward, so out[4] IS
            # the intervened tensor by construction; assert exactness anyway.
            z_t = out[4]
            if self.mode == 'zero_z':
                assert z_t.abs().max().item() == 0.0, \
                    'zero_z did not propagate to out[4] (z_t)'
            else:
                expected = self.replacement_z.to(
                    device=z_t.device, dtype=z_t.dtype)
                assert torch.equal(z_t.detach(), expected.detach()), \
                    'shuffled z did not propagate to out[4] (z_t)'
            self.last_z = z_t
            self.last_z_norm = float(
                z_t.detach().float().pow(2).sum().sqrt().item())
        return out

    def __enter__(self):
        import types
        intervention = self

        def forward_override(module_self, feat, velocity=None,
                             use_posterior=True, deterministic=True,
                             detach_state=True):
            return intervention._wrapped_forward(
                feat, velocity=velocity, use_posterior=use_posterior,
                deterministic=deterministic, detach_state=detach_state)

        self._installed_forward = types.MethodType(
            forward_override, self.fusion)
        self.fusion.forward = self._installed_forward
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fusion.__dict__.get('forward') is self._installed_forward:
            self.fusion.__dict__.pop('forward', None)
        self.fusion._z_intervention = None
        return False


def run_sequence_gated(model, batch, z_mode='normal', replacement_z=None):
    """Same frame loop as dz.run_sequence, with the gated intervention."""
    n_frames = model.seq_len
    frame_points, frame_img, frame_img_metas, frame_valid = \
        dz.build_sequence_frames(batch)
    if len(frame_img) < n_frames:
        raise ValueError(
            f'dataset returned {len(frame_img)} frames, model needs {n_frames}')

    fusion = model.temporal_fusion
    if fusion is None:
        raise ValueError('model has no temporal_fusion module')
    if not hasattr(fusion, '_z_intervention'):
        raise TypeError(
            f'{type(fusion).__name__} has no _z_intervention hook point; '
            'this probe only applies to gated variants (MCI)')

    z_diag = None
    z_value = None
    context = None
    if z_mode == 'normal':
        context = None
    else:
        context = ZInterventionGated(
            fusion, z_mode, n_frames, replacement_z=replacement_z)

    def _run():
        dz.reset_sequence_counter(fusion)
        for t in range(n_frames - 1):
            valid_t = frame_valid[t]
            if valid_t.any():
                with torch.no_grad():
                    model.extract_feat(
                        frame_points[t], frame_img[t], frame_img_metas[t],
                        is_valid_mask=valid_t, feat_or_dict=0,
                        curr_frame_supervision=False)
            if (~valid_t).any():
                fusion.reset_for_samples(~valid_t)

        current_t = n_frames - 1
        last_hist_valid = frame_valid[current_t - 1] if current_t >= 1 else \
            torch.ones(1, dtype=torch.bool,
                       device=frame_img[current_t].device)
        with torch.no_grad():
            feature_dict = model.extract_feat(
                frame_points[current_t], frame_img[current_t],
                frame_img_metas[current_t], is_valid_mask=last_hist_valid,
                feat_or_dict=1)
            features = feature_dict['pts_feats'][0]
            outs = model.pts_bbox_head(features)
        return feature_dict, features, outs

    if context is None:
        feature_dict, features, outs = _run()
    else:
        with context:
            feature_dict, features, outs = _run()
        z_diag = context.last_z_norm

    if context is not None and context.last_z is not None:
        z_value = context.last_z.detach().float().cpu()
    elif fusion.z_state is not None:
        z_value = fusion.z_state.detach().float().cpu()

    return {
        'features': features.detach().float().cpu(),
        'head_vector': dz.flatten_head_outputs(outs).cpu(),
        'head_scores': dz.head_score_summary(outs),
        'z_t': z_value,
        'z_t_norm': z_diag,
        'rssm_stats': {
            key: float(value.detach().float().cpu().item())
            for key, value in (feature_dict.get('rssm_stats') or {}).items()
            if torch.is_tensor(value) and value.numel() == 1
        },
    }


def diagnose_samples_gated(model, dataset, indices, device, shuffle_offset):
    """Same pairing/scoring as dz.diagnose_samples, gated runner."""
    if len(indices) < 2:
        raise ValueError('batch-shuffle requires at least two samples')

    batches = {
        index: dz.build_batch(dataset, [index], device)
        for index in indices
    }
    normal = {
        index: run_sequence_gated(model, batches[index], z_mode='normal')
        for index in indices
    }
    shuffle_sources = dz.resolve_shuffle_sources(indices, shuffle_offset)

    per_sample = {}
    for pos, index in enumerate(indices):
        base = normal[index]
        ref_index = shuffle_sources[index]
        ref_z = normal[ref_index]['z_t']
        if ref_z is None:
            raise ValueError('normal run did not return z_t')
        variants = {
            'normal': base,
            'zero_z': run_sequence_gated(
                model, batches[index], z_mode='zero_z'),
            'shuffle_z': run_sequence_gated(
                model, batches[index], z_mode='replace_z',
                replacement_z=ref_z),
            'prior_only': run_sequence_gated(
                model, batches[index], z_mode='prior_only'),
        }
        per_sample[index] = {
            'feature_norm': dz.vector_l2(base['features']),
            'head_norm': dz.vector_l2(base['head_vector']),
            'shuffle_source_index': ref_index,
            'variants': {
                name: dz.summarize_variant(variant, base)
                for name, variant in variants.items()
            },
            'normal_rssm_stats': base.get('rssm_stats', {}),
        }
    return per_sample


def parse_args():
    parser = argparse.ArgumentParser(
        description='Gated-variant counterfactual z_t utilization probe.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--limit', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--min-valid-history', type=int, default=2)
    parser.add_argument(
        '--require-valid-history', action='store_true', default=True,
        help='pick samples whose history frames are continuous in image_idx')
    parser.add_argument(
        '--allow-empty-history', action='store_false',
        dest='require_valid_history',
        help='keep the literal start-index even when its history is invalid')
    parser.add_argument('--shuffle-offset', type=int, default=0)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    from mmcv import Config
    from mmcv.runner import load_checkpoint

    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_detector

    import numpy as np

    args = parse_args()
    if args.batch_size >= 2 and args.limit < 2:
        raise ValueError('--limit must be >=2 when --batch-size >=2')
    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model['meta_info'] = {
        'figures_path': '/tmp/r4det_z_util_gated_figures',
        'project_name': 'TJ4D-z-utilization-gated',
    }
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    cfg.data.val.pop('samples_per_gpu', None)
    cfg.data.val.pop('workers_per_gpu', None)
    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().eval()
    assert hasattr(model.temporal_fusion, '_z_intervention'), \
        'target model is not a gated variant'

    indices = dz.select_indices(dataset, args)
    per_sample = diagnose_samples_gated(
        model, dataset, indices, args.gpu_id, args.shuffle_offset)
    result = {
        'indices': indices,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'probe': 'z_utilization_gated (intervention on final z_t)',
        'per_sample': per_sample,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    dz.print_summary(per_sample, indices)
    print('[json]', output)


if __name__ == '__main__':
    main()
