"""Compare candidate reconstruction targets by how sample-discriminative they are.

A target whose variance is dominated by an across-sample template can be
reconstructed by a sample-invariant z, so it cannot anchor z to the current
observation no matter how well the reconstruction loss itself descends. This
tool measures that split for candidate target definitions on cached BEV
features, without training.

Usage (physical GPU 5):

    CUDA_VISIBLE_DEVICES=5 python tools/probe_recon_target.py \
        --config configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z.py \
        --checkpoint /data/lurui/work_dirs/<run>/epoch_12.pth \
        --start-index 3 --limit 16 --output /tmp/recon_target.json

For each candidate target definition we standardize exactly the way the training
loss would, then decompose per-element variance into:
  shared  = energy of the across-sample mean (a constant predictor can fit it)
  specific= residual that actually differs between samples
A target dominated by `shared` can be reconstructed to a low loss by a
sample-invariant z, so it cannot anchor z to the observation.

Region-restricted decomposition (added 2026-10-09, measurement 2 of the
foreground-only question): the anchor and difference targets are ALSO
decomposed inside GT-box foreground, its energy-matched background control, the
Doppler-measured cells and the complement, using the region definitions of
tools/bev_region_utils.py. That answers the headroom question directly -- "if
the reconstruction target were the foreground (or the frame difference), how
much of ITS variance would be sample-specific?" -- and the difference family
tests whether temporal alignment is what makes a residual target informative:

  diff_globalstd          t = feat_t - feat_{t-1}
  diff_warped_a1          t = feat_t - warp(feat_{t-1}) with the L1 analytic
                          displacement at warp_scale 1.0 / dt 0.1 (full
                          compensation), only when the fusion carries the
                          ego-warp module
  diff_warped_learned     same with the checkpoint's learned warp_scale/dt
  pool16_fgstd            pool16 target standardized with FOREGROUND-only
                          mean/std -- the "fg-only recon target" candidate --
                          decomposed inside fg (adaptive normalization, so no
                          standardizing statistics are taken from the
                          background the target is not asking for)

RAW designs (added 2026-10-10, audit follow-up): every design above is
standardized, so none of them is the tensor training actually regresses.
R4Det.py:919/930 computes ``MSE(decoder([h_t,z_t]), bev_feats)`` on the
full-resolution RAW feature with no pooling, no standardization and no
detach. These two designs measure that target:

  raw_fullres_current     t = feat_t                    (the real target)
  raw_fullres_diff        t = feat_t - feat_{t-1}       (the candidate)

The variance split is scale-invariant, so it is computed on the raw tensor
directly; the printed ``rms`` is the point of these rows -- it is the energy
scale that sets how much gradient pressure the reconstruction term carries
(measured log: loss_rssm_recon is 0.18% of the total loss by ep24).

Read-only: no training, no checkpoint writes.
"""
import argparse, json, os, sys, types, warnings
warnings.filterwarnings('ignore')
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import numpy as np, torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector
from tools.diagnose_z_utilization import (
    build_batch, history_validity, run_sequence, select_indices)
from tools import bev_region_utils as bru
from me_rssm.modality_bus import ModalityContext

CANDS = ['pool16_globalstd', 'pool16_percellstd', 'pool32_globalstd',
         'fullres_globalstd', 'pool16_channelstd']
DIFF_CANDS = ['diff_globalstd', 'diff_warped_a1', 'diff_warped_learned']
# Unstandardized, full-resolution: what the training loss regresses (current)
# and the candidate replacement. Kept out of CANDS/DIFF_CANDS so the anchor
# reproduction path stays byte-identical.
RAW_CANDS = ['raw_fullres_current', 'raw_fullres_diff']
CLASS_NAMES = ['Pedestrian', 'Cyclist', 'Car', 'Truck']
PC_RANGE = [0.0, -39.68, -4.0, 69.12, 39.68, 2.0]
ANALYTIC_SCALE = (1.0, 0.1)   # warp_scale, dt of the full-compensation case


def make_target(feat, kind):
    if kind == 'fullres_globalstd':
        t = feat
    elif kind == 'pool16_globalstd':
        t = F.adaptive_avg_pool2d(feat, (16, 16))
    elif kind == 'pool32_globalstd':
        t = F.adaptive_avg_pool2d(feat, (32, 32))
    elif kind == 'pool16_percellstd':
        # Deliberately *no* standardization: the point of this variant is to
        # test whether the shared template is introduced by the normalizer.
        return F.adaptive_avg_pool2d(feat, (16, 16))
    elif kind == 'pool16_channelstd':
        t = F.adaptive_avg_pool2d(feat, (16, 16))
        m = t.mean(dim=(2, 3), keepdim=True)
        s = t.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        return (t - m) / s
    m = t.mean(dim=(1, 2, 3), keepdim=True)
    s = t.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
    return (t - m) / s


def standardize_global(t):
    m = t.mean(dim=(1, 2, 3), keepdim=True)
    s = t.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
    return (t - m) / s


def standardize_masked(t, mask):
    """Standardize each sample with the mean/std of its masked cells only.

    `t` is (S, C, H, W), `mask` is (S, H, W) boolean on the same grid; cells
    outside the mask are zeroed so the target carries no background content.
    """
    w = mask.unsqueeze(1).expand_as(t).to(t.dtype)
    count = w.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
    mean = (t * w).sum(dim=(1, 2, 3), keepdim=True) / count
    var = ((t - mean).pow(2) * w).sum(dim=(1, 2, 3), keepdim=True) / count
    return (t - mean) / var.sqrt().clamp_min(1e-4) * w


def build_target(kind, feats, feats_prev, analytics, learned_scale):
    """(S, C, H, W) standardized target for one design, or None if unavailable."""
    if kind == 'raw_fullres_current':
        return torch.stack([f for f in feats])
    if kind == 'raw_fullres_diff':
        return torch.stack([
            feats[p] - feats_prev[p] for p in range(len(feats))])
    if kind in CANDS:
        return torch.stack([make_target(f, kind) for f in feats])
    if kind == 'diff_globalstd':
        return torch.stack([
            standardize_global(feats[p] - feats_prev[p])
            for p in range(len(feats))])
    if kind in ('diff_warped_a1', 'diff_warped_learned'):
        if kind == 'diff_warped_learned' and learned_scale is None:
            return None
        ws, dts = ANALYTIC_SCALE if kind == 'diff_warped_a1' else learned_scale
        out = []
        for p in range(len(feats)):
            if analytics[p] is None:
                return None
            v_ego, cell_x, cell_y = analytics[p]
            warped_prev = bru.warp_map_by_cells(
                feats_prev[p], v_ego[:, 0] * ws * dts / cell_x,
                v_ego[:, 1] * ws * dts / cell_y)
            out.append(standardize_global(feats[p] - warped_prev))
        return torch.stack(out)
    raise KeyError(kind)


def decompose(T, mask=None):
    """Masked or full variance split of an (S, 1, C, H, W) target."""
    if mask is None:
        total = T.pow(2).mean().item()
        mean_t = T.mean(0, keepdim=True)
        shared = mean_t.pow(2).mean().item()
        spec = (T - mean_t).pow(2).mean().item()
        return {'shape': list(T.shape[1:]), 'total_var': total,
                'shared_template_frac': shared / total if total else float('nan'),
                'sample_specific_frac': spec / total if total else float('nan'),
                'cross_sample_mse_mean': spec}
    # targets are stacked per-sample features (S, 1, C, H, W); the masked
    # decomposition wants (S, C, H, W)
    T = T.reshape(T.shape[0], -1, *T.shape[-2:])
    stats = bru.masked_variance_stats(T, mask)
    if stats is None:
        return None
    stats['shape'] = list(T.shape[1:])
    stats['cross_sample_mse_mean'] = stats.get('specific_var')
    # same schema as the unmasked branch (both tables print them side by side)
    stats['shared_template_frac'] = stats.get('shared_frac')
    stats['sample_specific_frac'] = stats.get('specific_frac')
    return stats


def class_counts_of(dataset, index):
    """{class_name: n} GT instances in one frame; {} when the frame has no GT."""
    _, labels = bru.gt_boxes_of_frame(dataset, index)
    if labels is None:
        return {}
    counts = {}
    for lab in np.asarray(labels).reshape(-1).astype(int):
        name = (CLASS_NAMES[lab] if 0 <= lab < len(CLASS_NAMES) else str(lab))
        counts[name] = counts.get(name, 0) + 1
    return counts


def select_indices_covered(dataset, args):
    """Dispersed, history-valid frames carrying enough GT of every class.

    The 2026-10-09 region window turned out to hold only Car/Cyclist, which
    left Ped/Truck outside the table entirely. Here per-class coverage is a
    selection criterion rather than something noticed afterwards, and
    ``--stride`` disperses the window instead of taking consecutive frames.
    """
    selected, index = [], args.start_index
    while index < len(dataset) and len(selected) < args.limit:
        mask = history_validity(dataset, index)
        if mask is not None and sum(mask[:-1]) >= args.min_valid_history:
            counts = class_counts_of(dataset, index)
            if all(counts.get(c, 0) >= args.require_classes
                   for c in CLASS_NAMES):
                selected.append(index)
        index += args.stride
    if not selected:
        raise ValueError(
            f'no validation frame from start-index={args.start_index} with '
            f'>= {args.min_valid_history} valid history frames and '
            f'>= {args.require_classes} GT per class (stride={args.stride}); '
            f'len(dataset)={len(dataset)}')
    return selected


def to_grid(mask, target_hw):
    """Resample a per-sample boolean mask to the target's spatial grid.

    A pooled target cell belongs to the region when it overlaps any region
    cell, which is what adaptive_max_pool2d computes. Expects a single (H, W)
    mask; a stray leading singleton dim is dropped, anything else is an error
    (a 4-D mask silently stacked per sample is how a whole probe run died).
    """
    mask = np.asarray(mask)
    while mask.ndim > 2:
        if mask.shape[0] != 1:
            raise ValueError(
                f'to_grid expects a single (H, W) mask, got {mask.shape}')
        mask = mask[0]
    if tuple(mask.shape[-2:]) == tuple(target_hw):
        return mask
    t = torch.as_tensor(mask).float()[None, None]
    t = F.adaptive_max_pool2d(t, tuple(target_hw))[0, 0]
    return t.numpy() > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--start-index', type=int, default=3)
    ap.add_argument('--limit', type=int, default=16)
    ap.add_argument('--pc-range', type=float, nargs=6, default=PC_RANGE)
    ap.add_argument('--margin-cells', type=float, default=0.5)
    ap.add_argument('--stride', type=int, default=1,
                    help='walk step between candidate frames (1 = the '
                         'byte-identical anchor path)')
    ap.add_argument('--require-classes', type=int, default=0,
                    help='keep only frames with >= N GT of EVERY class '
                         '(0 = off)')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    torch.cuda.set_device(0); torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    cfg = Config.fromfile(a.config); cfg.model.pretrained = None; cfg.model.train_cfg = None
    cfg.model['meta_info'] = {'figures_path': '/tmp/r4det_td_figs', 'project_name': 'td'}
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    cfg.data.val.pop('samples_per_gpu', None); cfg.data.val.pop('workers_per_gpu', None)
    ds = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, a.checkpoint, map_location='cpu')
    model = model.cuda().eval()
    fusion = model.temporal_fusion
    cap = []
    orig = fusion.forward
    def wrapper(self, feat, *args, **kwargs):
        cap.append(feat.detach().float().cpu())
        return orig(feat, *args, **kwargs)
    fusion.forward = types.MethodType(wrapper, fusion)
    class A: pass
    args = A(); args.require_valid_history = True
    args.start_index = a.start_index; args.limit = a.limit; args.min_valid_history = 2
    args.stride = a.stride; args.require_classes = a.require_classes
    if a.stride == 1 and a.require_classes == 0:
        idx = select_indices(ds, args)
    else:
        idx = select_indices_covered(ds, args)

    feats, feats_prev, fg_masks, dop_masks, analytics, warp_meta = [], [], [], [], [], []
    has_ego_warp = getattr(fusion, 'use_ego_warp', False) and \
        hasattr(fusion, '_solve_ego')
    learned_scale = None
    if has_ego_warp:
        learned_scale = (float(fusion.warp_scale.detach().float().item()),
                         float(fusion.dt.detach().float().item()))
    window_class_counts = {}
    for i in idx:
        cap.clear()
        run_sequence(model, build_batch(ds, [i], 0), z_mode='normal')
        cur = cap[-1][0].unsqueeze(0)        # (1, C, H, W) current frame
        prev = cap[-2][0].unsqueeze(0)       # previous frame's fusion input
        feats.append(cur); feats_prev.append(prev)
        grid_hw = tuple(cur.shape[-2:])
        boxes, labels = bru.gt_boxes_of_frame(ds, i)
        if labels is not None:
            for lab in np.asarray(labels).reshape(-1).astype(int):
                name = (CLASS_NAMES[lab] if 0 <= lab < len(CLASS_NAMES)
                        else str(lab))
                window_class_counts[name] = window_class_counts.get(name, 0) + 1
        fg_masks.append(bru.fg_masks(boxes, labels, grid_hw, list(a.pc_range),
                                    a.margin_cells))
        dop_masks.append(bru.doppler_masks(
            ModalityContext.get('velocity_bev'), grid_hw))
        # one detached analytic solve per frame; the solve only reads the
        # Doppler canvas, so this does not touch module state
        analytics.append(bru.analytic_ego_velocity(fusion, cur.cuda())
                         if has_ego_warp else None)
        if has_ego_warp and analytics[-1] is not None:
            v_ego, cell_x, cell_y = analytics[-1]
            ws, dts = ANALYTIC_SCALE
            warp_meta.append(dict(
                index=i, v_ego=v_ego.detach().cpu().tolist(),
                cell_x=cell_x, cell_y=cell_y,
                dx_a1=float(v_ego[0, 0] * ws * dts / cell_x),
                dy_a1=float(v_ego[0, 1] * ws * dts / cell_y)))
    grid_hw = tuple(feats[0].shape[-2:])

    res = {'indices': idx, 'n': len(idx), 'checkpoint': a.checkpoint,
           'config': a.config, 'grid_hw': list(grid_hw),
           'pc_range': list(a.pc_range), 'margin_cells': a.margin_cells,
           'designs': {}, 'region_designs': {},
           'stride': a.stride, 'require_classes': a.require_classes,
           'class_coverage': window_class_counts,
           'learned_warp_scale_dt': learned_scale,
           'ego_warp_frames': warp_meta or None}

    # ---- the anchor designs, unchanged (must reproduce §51/§52 numbers) -----
    for kind in CANDS:
        res['designs'][kind] = decompose(build_target(
            kind, feats, feats_prev, analytics, learned_scale))
    # ---- difference family -------------------------------------------------
    for kind in DIFF_CANDS:
        T = build_target(kind, feats, feats_prev, analytics, learned_scale)
        if T is not None:
            res['designs'][kind] = decompose(T)

    # ---- the real training target and its candidate, unstandardized --------
    for kind in RAW_CANDS:
        res['designs'][kind] = decompose(build_target(
            kind, feats, feats_prev, analytics, learned_scale))

    # ---- region-restricted decomposition ----------------------------------
    fg_stack = torch.as_tensor(np.stack([m['fg'] for m in fg_masks]))
    energy = [bru.cell_energy(f) for f in feats]
    bg_hi_stack = torch.as_tensor(np.stack([
        bru.energy_matched_bg(energy[p], fg_masks[p]['fg'])
        for p in range(len(feats))]))
    region_masks = {'fg': fg_stack, 'bg': ~fg_stack, 'bg_hi': bg_hi_stack}
    if all(d is not None for d in dop_masks):
        measured = torch.as_tensor(np.stack([d['measured'] for d in dop_masks]))
        region_masks['measured'] = measured
        region_masks['unmeasured'] = ~measured
    region_sources = ['pool16_globalstd', 'fullres_globalstd'] + [
        k for k in DIFF_CANDS if k in res['designs']] + RAW_CANDS
    for kind in region_sources:
        T = build_target(kind, feats, feats_prev, analytics, learned_scale)
        target_hw = tuple(T.shape[-2:])
        res['region_designs'].setdefault('all', {})[kind] = decompose(T)
        for region, mask in region_masks.items():
            masked = torch.as_tensor(np.stack([
                to_grid(mask[p].numpy(), target_hw)
                for p in range(mask.shape[0])]))
            stats = decompose(T, masked)
            if stats is None:
                continue
            stats['region_mass'] = float(masked.float().mean().item())
            res['region_designs'].setdefault(region, {})[kind] = stats
    # ---- the fg-only standardized candidate (adaptive normalization) -------
    # f[0] drops the per-sample batch dim: every capture is (1, C, H, W), and
    # standardize_masked/masked_variance_stats both work in (S, C, H, W).
    pooled = torch.stack([F.adaptive_avg_pool2d(f[0], (16, 16)) for f in feats])
    pooled_fg = torch.as_tensor(np.stack([
        to_grid(fg_masks[p]['fg'], (16, 16)) for p in range(len(feats))]))
    fg_target = standardize_masked(pooled, pooled_fg)
    stats = decompose(fg_target, pooled_fg)
    if stats is not None:
        stats['region_mass'] = float(pooled_fg.float().mean().item())
        res['region_designs'].setdefault('fg', {})['pool16_fgstd'] = stats
        stats_bg = decompose(fg_target, ~pooled_fg)
        if stats_bg is not None:
            stats_bg['region_mass'] = float((~pooled_fg).float().mean().item())
            res['region_designs'].setdefault('bg', {})['pool16_fgstd'] = stats_bg

    output = os.path.abspath(a.output)
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    json.dump(res, open(output, 'w'), indent=2)
    print('%-22s %-16s %8s %10s %10s' % (
        'design', 'shape', 'total', 'shared%', 'specific%'))
    for k, v in res['designs'].items():
        print('%-22s %-16s %8.3f %9.1f%% %9.1f%%'
              % (k, str(tuple(v['shape'])), v['total_var'],
                 100*v['shared_template_frac'], 100*v['sample_specific_frac']))
    print('\nclass coverage in the selected window (%d frames): %s'
          % (len(idx), ', '.join('%s=%d' % (c, window_class_counts.get(c, 0))
                                 for c in CLASS_NAMES)))
    print('\nraw (unstandardized) targets -- energy scale sets how much the '
          'reconstruction term can push:')
    print('%-22s %10s %10s %10s %9s' % (
        'design', 'rms', 'shared rms', 'spec rms', 'specific%'))
    for k in RAW_CANDS:
        v = res['designs'].get(k)
        if not v:
            continue
        t, s = v['total_var'], v['shared_template_frac']
        print('%-22s %10.5f %10.5f %10.5f %8.1f%%'
              % (k, t ** 0.5, (t * s) ** 0.5, (t * (1 - s)) ** 0.5,
                 100*v['sample_specific_frac']))
    if res['designs'].get('raw_fullres_current') and \
            res['designs'].get('raw_fullres_diff'):
        c = res['designs']['raw_fullres_current']
        d = res['designs']['raw_fullres_diff']
        print('  diff/current energy ratio = %.4f  (specific-energy ratio '
              '%.4f)' % (d['total_var'] / c['total_var'] if c['total_var'] else float('nan'),
                         (d['total_var'] * d['sample_specific_frac']) /
                         (c['total_var'] * c['sample_specific_frac'])
                         if c['total_var'] * c['sample_specific_frac'] else float('nan')))
    print('\nregion-restricted decomposition (specific% = sample-specific '
          'share of that region):')
    print('%-22s %-11s %7s %9s %10s' % (
        'design', 'region', 'mass%', 'total', 'specific%'))
    for region in ['all', 'fg', 'bg', 'bg_hi', 'measured', 'unmeasured']:
        for kind, v in res['region_designs'].get(region, {}).items():
            print('%-22s %-11s %6.2f%% %9.4f %9.1f%%'
                  % (kind, region, 100.0 * v.get('region_mass', 1.0),
                     v['total_var'], 100*v['sample_specific_frac']))
    if res['ego_warp_frames']:
        w0 = res['ego_warp_frames'][0]
        print('\nL1 analytic displacement (cell units, full compensation):')
        print('  sample %d: v_ego=(%.3f, %.3f) m/s -> dx=%.3f dy=%.3f cells '
              '(cell %.4f x %.4f m)' % (
                  w0['index'], w0['v_ego'][0][0], w0['v_ego'][0][1],
                  w0['dx_a1'], w0['dy_a1'], w0['cell_x'], w0['cell_y']))
        print('  learned warp_scale=%.6f dt=%.6f'
              % res['learned_warp_scale_dt'])
    print('\n[json]', output)

if __name__ == '__main__':
    main()
