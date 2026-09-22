"""Counterfactual diagnostic for stochastic-state (z_t) utilization.

This is a read-only diagnostic. It loads one detector checkpoint, replays a
small validation batch with the current-frame RSSM latent z_t left unchanged
or replaced by zero / a batch-permuted tensor / the prior mean, and reports
how much the current-frame BEV feature and 3D-head outputs change.

The point is to test causality, not just inspect KL statistics: if all four
variants produce nearly identical head outputs, z_t has no causal task in the
current architecture.

Typical run on physical GPU 4:

    CUDA_VISIBLE_DEVICES=4 python tools/diagnose_z_utilization.py \
        --config me_rssm/configs/TJ4D-R4Det_clean_N4_2x4_24e_pretrained_v2_head_snapshot.py \
        --checkpoint /data/lurui/work_dirs/run10_headv2_multiseed/seed_0/epoch_16.pth \
        --limit 2 --batch-size 2
"""

import argparse
import json
import logging
import os
import sys
import types
import warnings

warnings.filterwarnings('ignore')
logging.getLogger('mmcv').setLevel(logging.ERROR)

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import DataContainer
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector


DEFAULT_CONFIG = (
    'me_rssm/configs/'
    'TJ4D-R4Det_clean_N4_2x4_24e_pretrained_v2_head_snapshot.py'
)
DEFAULT_CHECKPOINT = (
    '/data/lurui/work_dirs/run10_headv2_multiseed/seed_0/epoch_16.pth'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Counterfactual z_t utilization diagnostic for R4Det.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--limit', type=int, default=2,
                        help='number of validation sequences in the batch')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='must be >=2 for batch-shuffle to be meaningful')
    parser.add_argument(
        '--require-valid-history', action='store_true', default=True,
        help='pick samples whose history frames are continuous in image_idx')
    parser.add_argument(
        '--allow-empty-history', action='store_false',
        dest='require_valid_history',
        help='keep the literal start-index even when its history is invalid')
    parser.add_argument('--min-valid-history', type=int, default=2)
    parser.add_argument(
        '--shuffle-offset', type=int, default=0,
        help='index into the selected list used as the z_t shuffle source; '
             '0 uses the next sample and -1 uses the last selected sample')
    parser.add_argument(
        '--output', default='/tmp/r4det_z_utilization.json',
        help='JSON output path; use /tmp to avoid repository pollution')
    return parser.parse_args()


def unwrap(value):
    """Recursively unwrap mmcv DataContainer objects without collating."""
    if isinstance(value, DataContainer):
        return unwrap(value.data)
    if isinstance(value, list):
        return [unwrap(item) for item in value]
    if isinstance(value, tuple):
        return tuple(unwrap(item) for item in value)
    return value


def build_batch(dataset, indices, device):
    """Build the sequence-shaped batch expected by R4Det.extract_feat.

    TJ4DDataset returns one sequence per item:
        img:       Tensor [T, C, H, W] inside a DataContainer
        points:    list[T] of variable-length tensors
        img_metas: list[T] of per-frame metadata dicts

    The detector is called frame-by-frame, so we keep frame_points and
    frame_img_metas as Python lists while stacking images into [B, T, C, H, W].
    """
    cuda_device = torch.device('cuda', device)
    samples = []
    for index in indices:
        sample = dataset[index]
        samples.append(dict(
            img=unwrap(sample['img']),
            points=unwrap(sample['points']),
            img_metas=unwrap(sample['img_metas']),
        ))

    if not samples:
        raise ValueError('cannot build an empty diagnostic batch')

    img = torch.stack([sample['img'] for sample in samples], dim=0)
    img = img.to(cuda_device)
    points = []
    img_metas = []
    frame_valid = []
    for t in range(img.size(1)):
        points.append([
            sample['points'][t].to(cuda_device) for sample in samples
        ])
        img_metas.append([sample['img_metas'][t] for sample in samples])
        frame_valid.append(torch.tensor(
            [bool(sample['img_metas'][t]['is_prev_frame_valid'])
             for sample in samples],
            dtype=torch.bool,
            device=cuda_device))

    return dict(points=points, img=img, img_metas=img_metas,
                frame_valid=frame_valid)


def history_validity(dataset, index):
    seq_len = getattr(dataset, 'seq_len', None)
    if seq_len is None or index < seq_len - 1:
        return None
    image_indices = [
        dataset.data_infos[j]['image']['image_idx']
        for j in range(index - seq_len + 1, index + 1)
    ]
    mask = [False] * seq_len
    mask[-1] = True
    contiguous = True
    for t in range(seq_len - 2, -1, -1):
        contiguous = contiguous and (
            image_indices[t + 1] - image_indices[t] == 1)
        mask[t] = contiguous
    return mask


def select_indices(dataset, args):
    if not args.require_valid_history:
        start = args.start_index
        return list(range(start, min(start + args.limit, len(dataset))))

    selected = []
    index = args.start_index
    while index < len(dataset) and len(selected) < args.limit:
        mask = history_validity(dataset, index)
        if mask is not None and sum(mask[:-1]) >= args.min_valid_history:
            selected.append(index)
        index += 1
    if not selected:
        raise ValueError(
            'no validation sample with at least '
            f'{args.min_valid_history} valid history frames from '
            f'start-index={args.start_index}')
    return selected


def resolve_shuffle_sources(indices, offset):
    sources = {}
    for pos, index in enumerate(indices):
        if offset == 0:
            source_pos = (pos + 1) % len(indices)
        elif offset < 0:
            source_pos = (pos + offset) % len(indices)
        else:
            source_pos = (pos + offset) % len(indices)
        sources[index] = indices[source_pos]
    return sources


def build_sequence_frames(batch):
    img = batch['img']
    points = batch['points']
    img_metas = batch['img_metas']
    frame_valid = batch['frame_valid']

    n_frames = img.size(1)
    frame_img = [img[:, t, ...] for t in range(n_frames)]
    return points, frame_img, img_metas, frame_valid


def flatten_head_outputs(outs):
    flat = []

    def visit(value):
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return
        if torch.is_tensor(value):
            flat.append(value.detach().float().reshape(-1))

    visit(outs)
    return torch.cat(flat) if flat else torch.zeros(0)


def head_score_summary(outs):
    if not isinstance(outs, (list, tuple)) or not outs:
        return {}
    cls_scores = outs[0]
    if not isinstance(cls_scores, (list, tuple)) or not cls_scores:
        return {}
    with torch.no_grad():
        logits = torch.cat([x.detach().float().reshape(-1) for x in cls_scores])
        scores = logits.sigmoid()
    return {
        'score_mean': float(scores.mean().item()),
        'score_max': float(scores.max().item()),
        'score_gt_0p3': int((scores > 0.3).sum().item()),
    }


class ZIntervention:
    """Apply one z_t intervention to the current frame only."""

    def __init__(self, fusion, mode, seq_len, replacement_z=None):
        self.fusion = fusion
        self.mode = mode
        self.seq_len = seq_len
        self.replacement_z = replacement_z
        self.call_idx = -1
        self.active = False
        self.original_forward = fusion.forward
        self.hooks = []
        self.last_z = None
        self.last_z_norm = None

    def _make_hook(self, kind):
        def hook(module, inputs):
            if not self.active or not inputs:
                return None
            tensor = inputs[0]
            if kind == 'output_proj':
                z = tensor
                return (self._replace_z(z),)
            if kind == 'decoder':
                combined = tensor
                latent_dim = self.fusion.latent_dim
                h = combined[:, :-latent_dim]
                z = combined[:, -latent_dim:]
                return (torch.cat([h, self._replace_z(z)], dim=1),)
            return None
        return hook

    def _replace_z(self, z):
        if self.mode == 'zero_z':
            return torch.zeros_like(z)
        if self.mode == 'replace_z':
            if self.replacement_z is None:
                raise ValueError('replace_z requires a replacement tensor')
            if self.replacement_z.shape != z.shape:
                raise ValueError(
                    'replacement z shape mismatch: '
                    f'{tuple(self.replacement_z.shape)} vs '
                    f'{tuple(z.shape)}')
            return self.replacement_z.to(device=z.device, dtype=z.dtype)
        return z

    def _wrapped_forward(self, feat, velocity=None, use_posterior=True,
                         deterministic=True, detach_state=True):
        self.call_idx += 1
        is_current = self.call_idx == self.seq_len - 1
        self.active = is_current and self.mode in ('zero_z', 'replace_z')
        if is_current and self.mode == 'prior_only':
            use_posterior = False
        out = self.original_forward(
            feat, velocity=velocity, use_posterior=use_posterior,
            deterministic=deterministic, detach_state=detach_state)
        if is_current:
            z_t = out[4]
            if self.mode == 'zero_z':
                z_t = torch.zeros_like(z_t)
            elif self.mode == 'replace_z':
                z_t = self._replace_z(z_t)
            self.last_z = z_t
            self.last_z_norm = float(
                z_t.detach().float().pow(2).sum().sqrt().item())
        self.active = False
        return out

    def __enter__(self):
        intervention = self

        def forward_override(module_self, feat, velocity=None,
                             use_posterior=True, deterministic=True,
                             detach_state=True):
            return intervention._wrapped_forward(
                feat, velocity=velocity, use_posterior=use_posterior,
                deterministic=deterministic, detach_state=detach_state)

        self.fusion.forward = types.MethodType(forward_override, self.fusion)
        if self.mode in ('zero_z', 'replace_z'):
            self.hooks.append(self.fusion.output_proj.register_forward_pre_hook(
                self._make_hook('output_proj')))
            self.hooks.append(self.fusion.decoder.register_forward_pre_hook(
                self._make_hook('decoder')))
        return self

    def __exit__(self, exc_type, exc, tb):
        for hook in self.hooks:
            hook.remove()
        self.fusion.forward = self.original_forward
        return False


def reset_sequence_counter(fusion):
    fusion.reset_state()


def run_sequence(model, batch, z_mode='normal', replacement_z=None):
    n_frames = model.seq_len
    frame_points, frame_img, frame_img_metas, frame_valid = \
        build_sequence_frames(batch)
    if len(frame_img) < n_frames:
        raise ValueError(
            f'dataset returned {len(frame_img)} frames, model needs {n_frames}')

    fusion = model.temporal_fusion
    if fusion is None:
        raise ValueError('model has no temporal_fusion module')

    z_diag = None
    z_value = None
    context = None
    if z_mode == 'normal':
        context = None
    else:
        context = ZIntervention(
            fusion, z_mode, n_frames, replacement_z=replacement_z)

    def _run():
        reset_sequence_counter(fusion)
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
        last_hist_valid = frame_valid[current_t - 1] if current_t >= 1 else torch.ones(
            1, dtype=torch.bool, device=frame_img[current_t].device)
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
        'head_vector': flatten_head_outputs(outs).cpu(),
        'head_scores': head_score_summary(outs),
        'z_t': z_value,
        'z_t_norm': z_diag,
        'rssm_stats': {
            key: float(value.detach().float().cpu().item())
            for key, value in (feature_dict.get('rssm_stats') or {}).items()
            if torch.is_tensor(value) and value.numel() == 1
        },
    }


def vector_l2(value):
    return float(value.norm().item()) if value.numel() else float('nan')


def vector_l2_ratio(value, reference):
    if reference.numel() == 0:
        return float('nan')
    denom = reference.norm().item()
    return float((value - reference).norm().item() / denom) if denom else float('nan')


def summarize_variant(variant, base):
    base_feat = base['features']
    base_head = base['head_vector']
    return {
        'feature_l2_ratio_vs_normal': vector_l2_ratio(
            variant['features'], base_feat),
        'head_l2_ratio_vs_normal': vector_l2_ratio(
            variant['head_vector'], base_head),
        'feature_l2_vs_normal': vector_l2(
            variant['features'] - base_feat),
        'head_l2_vs_normal': vector_l2(
            variant['head_vector'] - base_head),
        'head_scores': variant['head_scores'],
        'z_t_norm': variant.get('z_t_norm'),
    }


def diagnose_samples(model, dataset, indices, device, shuffle_offset):
    """Run each sequence alone; eval preprocessing accepts batch size one."""
    if len(indices) < 2:
        raise ValueError('batch-shuffle requires at least two samples')

    batches = {
        index: build_batch(dataset, [index], device)
        for index in indices
    }
    normal = {
        index: run_sequence(model, batches[index], z_mode='normal')
        for index in indices
    }
    shuffle_sources = resolve_shuffle_sources(indices, shuffle_offset)

    per_sample = {}
    for pos, index in enumerate(indices):
        base = normal[index]
        ref_index = shuffle_sources[index]
        ref_z = normal[ref_index]['z_t']
        if ref_z is None:
            raise ValueError('normal run did not return z_t')
        variants = {
            'normal': base,
            'zero_z': run_sequence(model, batches[index], z_mode='zero_z'),
            'shuffle_z': run_sequence(
                model, batches[index], z_mode='replace_z',
                replacement_z=ref_z),
            'prior_only': run_sequence(
                model, batches[index], z_mode='prior_only'),
        }
        per_sample[index] = {
            'feature_norm': vector_l2(base['features']),
            'head_norm': vector_l2(base['head_vector']),
            'shuffle_source_index': ref_index,
            'variants': {
                name: summarize_variant(variant, base)
                for name, variant in variants.items()
            },
            'normal_rssm_stats': base.get('rssm_stats', {}),
        }
    return per_sample


def mean_metric(per_sample, indices, variant, metric):
    values = []
    for index in indices:
        node = per_sample[index]
        if variant is not None:
            node = node['variants'][variant]
        value = node[metric]
        if np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float('nan')


def print_summary(per_sample, indices):
    print('\n[z-utilization diagnostic]')
    print(f'indices: {indices}')
    print('normal feature norm: '
          f'{mean_metric(per_sample, indices, None, "feature_norm"):.4f}')
    print('normal head norm   : '
          f'{mean_metric(per_sample, indices, None, "head_norm"):.4f}')
    print('\nCounterfactual deltas vs normal z_t=mu_q:')
    for name in ['zero_z', 'shuffle_z', 'prior_only']:
        feat = mean_metric(
            per_sample, indices, name, 'feature_l2_ratio_vs_normal')
        head = mean_metric(
            per_sample, indices, name, 'head_l2_ratio_vs_normal')
        print(f'  {name:12s} feature_ratio={feat:.6f} '
              f'head_ratio={head:.6f}')
    print('\nIf zero/shuffle/prior ratios are all near zero, z_t has little '
          'causal effect on the detector output.')


def main():
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
        'figures_path': '/tmp/r4det_z_utilization_figures',
        'project_name': 'TJ4D-z-utilization-diagnostic',
    }
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)

    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().eval()

    indices = select_indices(dataset, args)
    per_sample = diagnose_samples(
        model, dataset, indices, args.gpu_id, args.shuffle_offset)
    result = {
        'indices': indices,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'per_sample': per_sample,
    }

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print_summary(per_sample, indices)
    print(f'\n[json] {output}')


if __name__ == '__main__':
    main()
