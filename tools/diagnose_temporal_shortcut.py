"""Counterfactual diagnostic for temporal-fusion shortcut behaviour.

This is a read-only diagnostic. It loads one detector checkpoint, replays a
small validation subset with different history interventions, and reports how
much the current-frame BEV feature and 3D-head outputs change when history is
removed, reversed, or when the RSSM residual path is disabled in memory.

The script never saves checkpoints, never edits configs, and writes only to the
path passed via ``--output`` (default: /tmp).

Typical run on physical GPU 4:

    CUDA_VISIBLE_DEVICES=4 python tools/diagnose_temporal_shortcut.py \
        --config configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py \
        --checkpoint /data/lurui/work_dirs/fgfull_N4_2x4_24e_seed0/epoch_16.pth \
        --limit 2
"""

import argparse
import json
import os
import types
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector


DEFAULT_CONFIG = (
    'configs/r4det/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py'
)
DEFAULT_CHECKPOINT = (
    '/data/lurui/work_dirs/fgfull_N4_2x4_24e_seed0/epoch_16.pth'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Counterfactual history-shortcut diagnostic for R4Det.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--limit', type=int, default=2,
                        help='number of validation sequences to diagnose')
    parser.add_argument(
        '--require-valid-history', action='store_true', default=True,
        help='pick samples whose history frames are continuous in image_idx')
    parser.add_argument(
        '--allow-empty-history', action='store_false',
        dest='require_valid_history',
        help='keep the literal start-index even when its history is invalid')
    parser.add_argument(
        '--min-valid-history', type=int, default=2,
        help='minimum number of valid history frames with --require-valid-history')
    parser.add_argument(
        '--output', default='/tmp/r4det_temporal_shortcut.json',
        help='JSON output path; use /tmp to avoid repository pollution')
    return parser.parse_args()


def to_cuda(value, device):
    if hasattr(value, 'data'):
        value = value.data
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [to_cuda(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cuda(item, device) for item in value)
    if hasattr(value, 'to'):
        return value.to(device)
    return value


def collate_one(dataset, index, device):
    item = dataset[index]
    cuda_device = torch.device('cuda', device)
    return dict(
        points=to_cuda(item['points'], cuda_device),
        img=to_cuda(item['img'], cuda_device),
        img_metas=item['img_metas'],
    )


def history_validity(dataset, index):
    """Return the old->new image-index validity mask for one sequence."""
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
    if not flat:
        return torch.zeros(0)
    return torch.cat(flat)


def head_score_summary(outs):
    """Summarise sigmoid class scores when the Anchor3DHead output is available."""
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
        'score_sum': float(scores.sum().item()),
        'score_gt_0p3': int((scores > 0.3).sum().item()),
    }


def build_sequence_frames(batch):
    img = getattr(batch['img'], 'data', batch['img'])
    points = getattr(batch['points'], 'data', batch['points'])
    img_metas = getattr(batch['img_metas'], 'data', batch['img_metas'])

    if img.dim() == 4:
        img = img.unsqueeze(0)
    if not isinstance(points, list):
        points = [points]
    if img_metas and not isinstance(img_metas[0], list):
        img_metas = [img_metas]

    n_frames = img.size(1)
    if points and len(points) == n_frames and torch.is_tensor(points[0]):
        # Single-sample dataset output: one points tensor per frame.
        frame_points = [[points[t]] for t in range(n_frames)]
    else:
        frame_points = [[p[t] for p in points] for t in range(n_frames)]
    frame_img = [img[:, t, ...] for t in range(n_frames)]
    frame_img_metas = [[meta[t] for meta in img_metas] for t in range(n_frames)]
    frame_valid = [
        torch.tensor([meta[t]['is_prev_frame_valid'] for meta in img_metas],
                     device=img.device)
        for t in range(n_frames)
    ]
    return frame_points, frame_img, frame_img_metas, frame_valid


def run_sequence(model, batch, history_order='normal', force_raw_current=False):
    """Run one sequence and return current BEV features and head outputs."""
    n_frames = model.seq_len
    frame_points, frame_img, frame_img_metas, frame_valid = \
        build_sequence_frames(batch)
    if len(frame_img) < n_frames:
        raise ValueError(f'dataset returned {len(frame_img)} frames, model needs {n_frames}')

    if model.temporal_fusion is not None:
        model.temporal_fusion.reset_state()

        if history_order == 'normal':
            history_indices = list(range(n_frames - 1))
        elif history_order == 'reverse':
            history_indices = list(range(n_frames - 2, -1, -1))
        elif history_order == 'none':
            history_indices = []
        else:
            raise ValueError(f'unknown history_order: {history_order}')

        for t in history_indices:
            valid_t = frame_valid[t]
            if valid_t.any():
                with torch.no_grad():
                    model.extract_feat(
                        frame_points[t], frame_img[t], frame_img_metas[t],
                        is_valid_mask=valid_t, feat_or_dict=0,
                        curr_frame_supervision=False)
            if (~valid_t).any():
                model.temporal_fusion.reset_for_samples(~valid_t)

    current_t = n_frames - 1
    last_hist_valid = frame_valid[current_t - 1] if current_t >= 1 else torch.ones(
        1, dtype=torch.bool, device=frame_img[current_t].device)
    if force_raw_current:
        last_hist_valid = torch.zeros_like(last_hist_valid)

    with torch.no_grad():
        feature_dict = model.extract_feat(
            frame_points[current_t], frame_img[current_t],
            frame_img_metas[current_t], is_valid_mask=last_hist_valid,
            feat_or_dict=1)
        features = feature_dict['pts_feats'][0]
        outs = model.pts_bbox_head(features)

    return {
        'features': features.detach().float().cpu(),
        'head_vector': flatten_head_outputs(outs).cpu(),
        'head_scores': head_score_summary(outs),
        'rssm_stats': {
            key: float(value.detach().float().cpu().item())
            for key, value in (feature_dict.get('rssm_stats') or {}).items()
            if torch.is_tensor(value) and value.numel() == 1
        },
    }


def output_proj_zero_context(model):
    fusion = model.temporal_fusion
    if fusion is None or not hasattr(fusion, 'output_proj'):
        return None

    def zero_output(module, inputs, output):
        return torch.zeros_like(output)

    return fusion.output_proj.register_forward_hook(zero_output)


def prior_only_context(model):
    fusion = model.temporal_fusion
    if fusion is None:
        return None

    original_forward = fusion.forward

    def prior_forward(self, feat, velocity=None, use_posterior=True,
                      deterministic=True, detach_state=True):
        return original_forward(
            feat, velocity=velocity, use_posterior=False,
            deterministic=deterministic, detach_state=detach_state)

    fusion.forward = types.MethodType(prior_forward, fusion)
    return original_forward


def restore_prior_only_context(model, original_forward):
    if original_forward is not None:
        model.temporal_fusion.forward = original_forward


def vector_l2_ratio(value, reference):
    if reference.numel() == 0:
        return float('nan')
    denom = reference.norm().item()
    if denom == 0:
        return float('nan')
    return float((value - reference).norm().item() / denom)


def vector_l2(value):
    return float(value.norm().item()) if value.numel() else float('nan')


def diagnose_one(model, batch):
    variants = {}
    variants['true'] = run_sequence(model, batch, history_order='normal')
    variants['no_history'] = run_sequence(model, batch, history_order='none')
    variants['reverse_history'] = run_sequence(model, batch, history_order='reverse')
    variants['raw_current_fallback'] = run_sequence(
        model, batch, history_order='normal', force_raw_current=True)

    hook = output_proj_zero_context(model)
    try:
        variants['zero_temporal_residual'] = run_sequence(
            model, batch, history_order='normal')
    finally:
        if hook is not None:
            hook.remove()

    original_forward = prior_only_context(model)
    try:
        variants['prior_only'] = run_sequence(model, batch, history_order='normal')
    finally:
        restore_prior_only_context(model, original_forward)

    base_feat = variants['true']['features']
    base_head = variants['true']['head_vector']
    raw_feat = variants['raw_current_fallback']['features']
    raw_head = variants['raw_current_fallback']['head_vector']
    zero_residual_sidecar_diff = vector_l2_ratio(
        variants['zero_temporal_residual']['head_vector'], raw_head)

    result = {
        'feature_norm': vector_l2(base_feat),
        'raw_feature_norm': vector_l2(raw_feat),
        'temporal_residual_ratio': vector_l2_ratio(base_feat, raw_feat),
        'temporal_residual_head_ratio': vector_l2_ratio(base_head, raw_head),
        'zero_residual_matches_raw_fallback_head_ratio': zero_residual_sidecar_diff,
        'variants': {},
        'rssm_stats': variants['true'].get('rssm_stats', {}),
    }
    for name, variant in variants.items():
        result['variants'][name] = {
            'feature_l2_ratio_vs_true': vector_l2_ratio(
                variant['features'], base_feat),
            'head_l2_ratio_vs_true': vector_l2_ratio(
                variant['head_vector'], base_head),
            'feature_l2_vs_true': vector_l2(variant['features'] - base_feat),
            'head_l2_vs_true': vector_l2(variant['head_vector'] - base_head),
            'head_scores': variant['head_scores'],
        }
    return result


def aggregate_results(per_sample):
    keys = [
        'temporal_residual_ratio',
        'temporal_residual_head_ratio',
        'zero_residual_matches_raw_fallback_head_ratio',
    ]
    summary = {}
    for key in keys:
        values = [sample[key] for sample in per_sample if np.isfinite(sample[key])]
        summary[key] = float(np.mean(values)) if values else float('nan')
    for variant in ['no_history', 'reverse_history', 'zero_temporal_residual', 'prior_only']:
        for key in ['feature_l2_ratio_vs_true', 'head_l2_ratio_vs_true']:
            values = [
                sample['variants'][variant][key] for sample in per_sample
                if np.isfinite(sample['variants'][variant][key])
            ]
            summary[f'{variant}_{key}'] = (
                float(np.mean(values)) if values else float('nan'))
    return summary


def print_summary(summary):
    print('\n[temporal-shortcut diagnostic]')
    print(f"temporal residual / raw feature ratio : {summary['temporal_residual_ratio']:.4f}")
    print(f"temporal residual / head-output ratio  : {summary['temporal_residual_head_ratio']:.4f}")
    print(f"zero-residual vs raw-fallback head diff: "
          f"{summary['zero_residual_matches_raw_fallback_head_ratio']:.4e}")
    print('\nHistory interventions (mean L2 ratio vs true history):')
    for variant in ['no_history', 'reverse_history', 'prior_only']:
        print(f"  {variant:24s} feature="
              f"{summary[f'{variant}_feature_l2_ratio_vs_true']:.4f}  "
              f"head={summary[f'{variant}_head_l2_ratio_vs_true']:.4f}")
    print(f"  {'zero_temporal_residual':24s} feature="
          f"{summary['zero_temporal_residual_feature_l2_ratio_vs_true']:.4f}  "
          f"head={summary['zero_temporal_residual_head_l2_ratio_vs_true']:.4f}")
    print('\nInterpretation: small no_history/reverse_history deltas indicate '
          'the detector is insensitive to history; small zero_temporal_residual '
          'deltas indicate the 3D head mostly ignores the RSSM correction.')


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model['meta_info'] = {
        'figures_path': '/tmp/r4det_temporal_shortcut_figures',
        'project_name': 'TJ4D-temporal-shortcut-diagnostic',
    }
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)

    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().eval()

    indices = select_indices(dataset, args)
    if not indices:
        raise ValueError(
            f'no validation samples selected from start={args.start_index}')

    per_sample = []
    for index in indices:
        batch = collate_one(dataset, index, args.gpu_id)
        result = diagnose_one(model, batch)
        result['index'] = index
        result['history_validity'] = history_validity(dataset, index)
        per_sample.append(result)
        print(f'[sample {index}] residual_ratio={result["temporal_residual_ratio"]:.4f} '
              f'head_residual_ratio={result["temporal_residual_head_ratio"]:.4f} '
              f'no_history_feat={result["variants"]["no_history"]["feature_l2_ratio_vs_true"]:.4f} '
              f'no_history_head={result["variants"]["no_history"]["head_l2_ratio_vs_true"]:.4f}')

    summary = aggregate_results(per_sample)
    payload = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'indices': indices,
        'summary': summary,
        'per_sample': per_sample,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print_summary(summary)
    print(f'\n[json] {output}')


if __name__ == '__main__':
    main()
