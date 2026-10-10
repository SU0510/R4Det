"""Counterfactual z_t utilization probe for STANDARD (non-gated) RSSM arms.

Read-only diagnostic. Same sample selection, pairing, scoring and JSON schema
as ``tools/diagnose_z_utilization.py`` (imported wholesale), with TWO
deliberate differences, both required for candidate A:

1. The intervention is ``dz.ZIntervention`` — the posterior-head rewrite the
   standard family already uses. ``probe_z_utilization_gated.py`` asserts a
   ``_z_intervention`` class hook point that only the MCI/DC family defines
   (``MotionAlignedInnovationRSSMFusion._z_intervention``); on
   ``SemanticReadoutRSSMFusion`` (parent ``MotionAlignedRSSMFusion``) that probe
   raises ``AssertionError: target model is not a gated variant`` before it
   reads anything. Verified 2026-10-10 against the 450-iter smoke checkpoint.

2. The head is called the way the DETECTOR calls it in eval, i.e. with
   ``cls_logit_residual`` when ``extract_feat`` produced one. dz's runner calls
   ``pts_bbox_head(features)`` with no residual, which is correct for every
   pre-candidate-A generation (no such input existed) but describes a detector
   that is never run here: R4Det.extract_feat -> feature_dict ->
   ``simple_test_pts`` forwards ``rssm_cls_logit_residual`` to the head. Scoring
   the residual-free head would understate how much z moves the class
   decisions, i.e. it would bias the very quantity this probe exists to
   measure.

Everything else (frame loop, valid-history masking, shuffle-source resolution,
feature/head L2 ratios, head-score summary) is dz's own code, so the
shuffle_z / zero_z numbers stay comparable with §49 (baseline 4.67%) and §69.

Usage (mirrors the gated probe):
  CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_z_utilization_standard.py \
    --config me_rssm/configs/..._semreadout.py \
    --checkpoint <work_dir>/epoch_6.pth \
    --start-index 3 --limit 4 --batch-size 4 --shuffle-offset 2 \
    --output <work_dir>/z_util_std_ep6_w3.json
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


def run_sequence_standard(model, batch, z_mode='normal', replacement_z=None):
    """dz.run_sequence, with the deployed class residual handed to the head."""
    n_frames = model.seq_len
    frame_points, frame_img, frame_img_metas, frame_valid = \
        dz.build_sequence_frames(batch)
    if len(frame_img) < n_frames:
        raise ValueError(
            f'dataset returned {len(frame_img)} frames, model needs {n_frames}')

    fusion = model.temporal_fusion
    if fusion is None:
        raise ValueError('model has no temporal_fusion module')
    if not hasattr(fusion, 'posterior_mu'):
        raise TypeError(
            f'{type(fusion).__name__} has no posterior_mu head; this probe '
            'only applies to the standard family (use the gated probe for '
            'MCI/DC variants)')

    z_diag = None
    z_value = None
    context = None
    if z_mode == 'normal':
        context = None
    else:
        context = dz.ZIntervention(
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
            residual = feature_dict.get('rssm_cls_logit_residual')
            # Candidate A: the deployed path passes this; dz's runner does not.
            outs = model.pts_bbox_head(
                feature_dict['pts_feats'], cls_logit_residual=residual) \
                if residual is not None else model.pts_bbox_head(
                    feature_dict['pts_feats'])
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


def diagnose_samples_standard(model, dataset, indices, device, shuffle_offset):
    """Same pairing/scoring as dz.diagnose_samples, standard-family runner."""
    if len(indices) < 2:
        raise ValueError('batch-shuffle requires at least two samples')

    batches = {
        index: dz.build_batch(dataset, [index], device)
        for index in indices
    }
    normal = {
        index: run_sequence_standard(model, batches[index], z_mode='normal')
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
            'zero_z': run_sequence_standard(
                model, batches[index], z_mode='zero_z'),
            'shuffle_z': run_sequence_standard(
                model, batches[index], z_mode='replace_z',
                replacement_z=ref_z),
            'prior_only': run_sequence_standard(
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
        description='Standard-family counterfactual z_t utilization probe.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--limit', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--min-valid-history', type=int, default=2)
    parser.add_argument(
        '--require-valid-history', action='store_true', default=True)
    parser.add_argument(
        '--allow-empty-history', action='store_false',
        dest='require_valid_history')
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
        'figures_path': '/tmp/r4det_z_util_standard_figures',
        'project_name': 'TJ4D-z-utilization-standard',
    }
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    cfg.data.val.pop('samples_per_gpu', None)
    cfg.data.val.pop('workers_per_gpu', None)
    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().eval()

    indices = dz.select_indices(dataset, args)
    per_sample = diagnose_samples_standard(
        model, dataset, indices, args.gpu_id, args.shuffle_offset)
    result = {
        'indices': indices,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'probe': ('z_utilization_standard (posterior-head intervention, '
                  'deployed class residual passed to the head)'),
        'per_sample': per_sample,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    dz.print_summary(per_sample, indices)
    print('[json]', output)


if __name__ == '__main__':
    main()
