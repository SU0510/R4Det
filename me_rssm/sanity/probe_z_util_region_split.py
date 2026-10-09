"""Region-split counterfactual z_t dependency readout (measurement 1).

Pre-registered question (written 2026-10-09, before any data from this probe):
the family-level P1 readout says z_t moves the head by ~5-27% (relative), that
zero_z is dominated by a readout bias, and that the effect is architecture-
dependent. It does NOT say WHERE in the BEV that dependence lives. The
"foreground-only RSSM" idea (docs §51/§52) only has a mechanism to stand on if
the current latent's causal effect is concentrated inside the object region
relative to an energy-matched background. If it is not, the bottleneck is the
read-out, not the state's region coverage, and cutting the state to the
foreground cannot buy AP.

Design: same intervention, rollout and z-pairing as
me_rssm/sanity/probe_z_utilization_gated.py (final-z intervention on the
current frame only; MCI/evidence families use the gated hook point, MAI and
plain-RSSM families fall back to rewriting posterior_mu). Every delta is then
reduced PER REGION via tools/bev_region_utils.py:

  fg          GT boxes rasterized on the evidence grid (0.32 m cells)
  bg          complement of fg
  bg_hi       background cells whose feature energy reaches the fg median --
              the fair control: a raw fg-vs-bg delta comparison is confounded
              by activation magnitude, which is far larger on objects
  bg_top10    background cells in the top decile of background energy, a
              sharper control when the fg median sits low
  measured    Doppler canvas occupancy > 0
  unmeasured  complement of measured
  dyn         measured & v_std > threshold (Doppler-defined dynamic proxy;
              known-broken for Pedestrian, contrast only)

The three Doppler regions need the motion chain (RadarPillarFeatureNetMotion +
PointPillarsScatterMotion). Configs without it (baseline / MAI families) have
no canvas: those runs report fg/bg/bg_hi/bg_top10 only and say so loudly.

Layout: the captured feature/head tensors are image-like (dim2 = world y,
dim3 = world x -- R4Det permutes before the head), so they are transposed once
inside region_readout; the masks are built in the (x, y) layout. With the
motion chain the evidence grid is derived from the raw Doppler canvas and every
run prints an orientation self-check, P(measured | fg) / P(measured) for the
chosen vs the transposed pairing (GT cars are radar-visible, so the correct
pairing must win) -- the axis convention is then measured, not assumed. Without
the canvas the same grid is where the two derivations must agree by the pinned
code fact (the head-facing permute), which the canvas-equipped family confirms
empirically.

Per region and variant the probe reports the region's mean |base| activation,
the mean |change|, and their ratio, for both the current-frame BEV feature and
the 3D head's class logits. The head is called with the level list (the form
`simple_test_pts` uses); the global head ratio is also printed so the readout
can be cross-checked against the previously documented P1 numbers.

Read-only: no training, no checkpoint or run-dir writes except --output.

Usage:

    CUDA_VISIBLE_DEVICES=4 python me_rssm/sanity/probe_z_util_region_split.py \
        --config me_rssm/configs/..._dcrssm1.py \
        --checkpoint /data/lurui/work_dirs/<run>/epoch_16.pth \
        --start-index 3 --limit 12 \
        --output /data/lurui/work_dirs/<run>/zutil_region_ep16.json
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

from tools import bev_region_utils as bru  # noqa: E402
from me_rssm.modality_bus import ModalityContext  # noqa: E402


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO, relative_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dz = _load('dz', os.path.join('tools', 'diagnose_z_utilization.py'))
gated = _load(
    'zutil_gated', os.path.join('me_rssm', 'sanity',
                                'probe_z_utilization_gated.py'))

PC_RANGE = [0.0, -39.68, -4.0, 69.12, 39.68, 2.0]
# (display label, intervention mode): both intervention classes name the
# shuffle variant 'replace_z' and take the donor z through replacement_z.
VARIANTS = (('zero_z', 'zero_z'), ('shuffle_z', 'replace_z'),
            ('prior_only', 'prior_only'))
VARIANT_LABELS = tuple(label for label, _ in VARIANTS)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Region-split z_t dependency readout.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--start-index', type=int, default=3)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--min-valid-history', type=int, default=2)
    parser.add_argument('--shuffle-offset', type=int, default=0)
    parser.add_argument('--pc-range', type=float, nargs=6, default=PC_RANGE)
    parser.add_argument('--margin-cells', type=float, default=0.5)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def intervention_class(fusion):
    """Gated hook point when the family has one, posterior rewrite otherwise."""
    if hasattr(fusion, '_z_intervention'):
        return gated.ZInterventionGated
    if hasattr(fusion, 'posterior_mu'):
        return dz.ZIntervention
    raise TypeError(
        f'{type(fusion).__name__} exposes neither a _z_intervention hook '
        'point nor a posterior_mu head; the probe cannot intervene on it')


def cls_logits(outs):
    """First class-logit tensor of the head outputs, whatever the nesting."""
    node = outs
    while isinstance(node, (list, tuple)):
        if not node:
            return None
        node = node[0]
    return node if torch.is_tensor(node) else None


def run_pass(model, batch, mode, intervention_cls, replacement_z=None):
    """One sequence rollout; returns CPU feature map, cls logits and z_t.

    Same frame loop as the gated probe's runner, with one deliberate
    difference: the head receives `pts_feats` (the level list `simple_test_pts`
    passes in the real test path), so the class logits come back batched
    (B, anchors*classes, H, W) instead of unbatched.
    """
    n_frames = model.seq_len
    frame_points, frame_img, frame_img_metas, frame_valid = \
        dz.build_sequence_frames(batch)
    fusion = model.temporal_fusion
    context = None
    if mode != 'normal':
        context = intervention_cls(fusion, mode, n_frames,
                                   replacement_z=replacement_z)

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
            outs = model.pts_bbox_head(feature_dict['pts_feats'])
        return feature_dict, outs

    with context if context is not None else _null_context():
        feature_dict, outs = _run()

    z_value = None
    if context is not None and context.last_z is not None:
        z_value = context.last_z.detach().float().cpu()
    elif fusion.z_state is not None:
        z_value = fusion.z_state.detach().float().cpu()
    logits = cls_logits(outs)
    return {
        'features': feature_dict['pts_feats'][0].detach().float().cpu(),
        'logits': None if logits is None else logits.detach().float().cpu(),
        'head_vector': dz.flatten_head_outputs(outs).cpu(),
        'z_t': z_value,
        'z_t_norm': None if context is None else context.last_z_norm,
        'rssm_stats': {
            key: float(value.detach().float().cpu().item())
            for key, value in (feature_dict.get('rssm_stats') or {}).items()
            if torch.is_tensor(value) and value.numel() == 1
        },
    }


class _null_context(object):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def region_definitions(map_xy, fg_masks, dop_masks):
    """Region masks for one sample, all (H_x, W_y) boolean numpy arrays.

    `map_xy` is a channel-first map ALREADY in the (x, y) layout (see the
    layout note in tools/bev_region_utils.py).
    """
    energy = bru.cell_energy(map_xy)
    regions = {'fg': fg_masks['fg'], 'bg': ~fg_masks['fg'],
               'bg_hi': bru.energy_matched_bg(energy, fg_masks['fg']),
               'bg_top10': bru.top_energy_background(energy, fg_masks['fg'])}
    if dop_masks is not None:
        measured = dop_masks['measured']
        regions['measured'] = measured
        regions['unmeasured'] = ~measured
        regions['dyn'] = dop_masks['dyn']
    for key, mask in fg_masks.items():
        if key.startswith('fg:'):
            regions[key] = mask
    return regions


def region_readout(pass_normal, pass_variant, regions):
    """feat + cls {'base','delta','rel'} per region for one variant.

    The captured tensors are image-like (dim2 = y, dim3 = x) because R4Det
    hands the head a permuted feature; transposing the last two dims once
    here puts every map in the (x, y) layout the masks are built in.
    """
    feat_base = pass_normal['features'][0].transpose(-2, -1)
    feat_val = pass_variant['features'][0].transpose(-2, -1)
    out = {'feat': {}, 'cls': {}}
    for name, mask in regions.items():
        out['feat'][name] = bru.masked_stats(feat_val, feat_base, mask)
    if pass_normal['logits'] is not None and \
            pass_variant['logits'] is not None:
        logit_base = pass_normal['logits'][0].transpose(-2, -1)
        logit_val = pass_variant['logits'][0].transpose(-2, -1)
        for name, mask in regions.items():
            out['cls'][name] = bru.masked_stats(logit_val, logit_base, mask)
    return out


def orientation_check(fg_mask, fg_mask_wrong, dop_masks):
    """P(measured | fg) / P(measured) for the chosen and the swapped grid.

    GT boxes (cars) are strongly radar-visible, so the correct axis pairing
    lifts the measured-cell rate inside the foreground well above 1 while a
    mask rasterized on the swapped grid does not. `fg_mask_wrong` must be the
    box mask built with the axes exchanged and transposed back to the (x, y)
    layout -- transposing the mask AND the map together would select exactly
    the same cells and prove nothing.
    """
    if dop_masks is None:
        return None
    measured = dop_masks['measured']
    rate = float(measured.mean())
    if rate <= 0 or not bool(fg_mask.any()):
        return None

    def _lift(mask):
        return float(measured[mask].mean()) / rate

    return {'lift_xy': _lift(fg_mask),
            'lift_swapped': _lift(fg_mask_wrong),
            'measured_rate': rate}


def diagnose(model, dataset, indices, device, pc_range, margin_cells):
    fusion = model.temporal_fusion
    intervention_cls = intervention_class(fusion)
    grid_xy = None
    batches = {}
    per_sample = {}
    normal_pass = {}
    for index in indices:
        batches[index] = dz.build_batch(dataset, [index], device)
    for index in indices:
        normal_pass[index] = run_pass(
            model, batches[index], 'normal', intervention_cls)
    # The evidence grid is (H_x, W_y) with dim2 = world x (pinned in
    # tools/bev_region_utils.py). The raw Doppler canvas states it directly;
    # without the motion chain (baseline / MAI configs) the captured feature
    # has to be reversed instead — pts_feats[0] is permuted for the head, so
    # its last two dims are (W_y, H_x). The two derivations are cross-checked
    # whenever the canvas exists.
    captured = tuple(normal_pass[indices[0]]['features'].shape[-2:])
    grid_xy = bru.grid_xy_from_canvas(ModalityContext.get('velocity_bev'))
    has_doppler = grid_xy is not None
    if grid_xy is None:
        grid_xy = tuple(reversed(captured))
        print('[note] no velocity_bev canvas (config without the motion '
              'chain):\n       measured/unmeasured/dyn regions and the '
              'orientation check are NOT EVALUABLE;\n       the (x, y) grid '
              f'is derived from the captured feature {captured} reversed -> '
              f'{tuple(grid_xy)}.')
    elif tuple(reversed(captured)) != tuple(grid_xy):
        raise RuntimeError(
            f'captured feature is {captured} but the canvas implies an '
            f'evidence grid of {tuple(reversed(grid_xy))}; the (x, y) '
            'transpose in region_readout would be wrong')

    shuffle_sources = dz.resolve_shuffle_sources(indices, 0)
    for index in indices:
        base = normal_pass[index]
        ref_index = shuffle_sources[index]
        ref_z = normal_pass[ref_index]['z_t']
        if ref_z is None:
            raise ValueError('normal pass did not return z_t')
        variants = {}
        for label, mode in VARIANTS:
            variants[label] = run_pass(
                model, batches[index], mode, intervention_cls,
                replacement_z=ref_z if label == 'shuffle_z' else None)

        boxes, labels = bru.gt_boxes_of_frame(dataset, index)
        fg_masks = bru.fg_masks(boxes, labels, grid_xy, pc_range,
                                margin_cells)
        # counterfactual mask for the orientation self-check: the same boxes
        # rasterized on the swapped grid, transposed back to (x, y)
        fg_wrong = bru.rasterize_boxes(
            boxes, (grid_xy[1], grid_xy[0]), pc_range, margin_cells)[0].T
        dop = bru.doppler_masks(ModalityContext.get('velocity_bev'), grid_xy)
        regions = region_definitions(
            base['features'][0].transpose(-2, -1), fg_masks, dop)

        record = {
            'index': index,
            'shuffle_source_index': ref_index,
            'z_t_norm': {label: variants[label]['z_t_norm']
                         for label, _ in VARIANTS},
            'n_boxes_in_range': fg_masks['n_boxes'],
            'n_boxes_total': 0 if boxes is None else int(len(boxes)),
            'orientation_check': orientation_check(
                fg_masks['fg'], fg_wrong, dop),
            'region_cells': {name: int(mask.sum())
                             for name, mask in regions.items()},
            'normal_rssm_stats': base.get('rssm_stats', {}),
            'variants': {
                label: region_readout(base, variants[label], regions)
                for label, _ in VARIANTS
            },
            'head_ratio_global': {
                label: dz.vector_l2_ratio(
                    variants[label]['head_vector'],
                    base['head_vector']) for label, _ in VARIANTS
            },
            'feature_ratio_global': {
                label: dz.vector_l2_ratio(
                    variants[label]['features'], base['features'])
                for label, _ in VARIANTS
            },
        }
        per_sample[index] = record
    return per_sample, grid_xy, has_doppler


def _mean(values):
    values = [v for v in values if v == v]  # drop NaN
    return sum(values) / len(values) if values else float('nan')


def summarize(per_sample, indices, grid_xy, checkpoint, has_doppler=True):
    regions = ['fg', 'bg', 'bg_hi', 'bg_top10']
    if has_doppler:
        regions += ['measured', 'unmeasured', 'dyn']
    for name in reversed(sorted(
            set(k for rec in per_sample.values()
                for k in rec['region_cells'] if k.startswith('fg:')))):
        regions.append(name)
    total_cells = grid_xy[0] * grid_xy[1]
    print('\n[region-split z_t dependency]')
    print(f'checkpoint: {checkpoint}')
    print(f'samples: {indices}')
    print(f'grid: {grid_xy[0]}x{grid_xy[1]} (H_x = {grid_xy[0]} world x cells, '
          f'W_y = {grid_xy[1]} world y cells, 0.32 m)')
    thin = []
    for name in regions:
        cells = _mean([rec['region_cells'].get(name, 0)
                       for rec in per_sample.values()])
        flag = ''
        if name in ('dyn',) and cells < 50:
            flag = '  <- support too thin to read; suppressed below'
            thin.append(name)
        print(f'  region {name:12s}: {100.0 * cells / total_cells:6.2f}% '
              f'of cells{flag}')
    boxes = _mean([rec['n_boxes_in_range'] for rec in per_sample.values()])
    print(f'  GT boxes in range: {boxes:.1f} per frame, '
          f'fg cells {100.0 * _mean([rec["region_cells"]["fg"] for rec in per_sample.values()]) / total_cells:.2f}%')
    lifts_xy, lifts_wrong = [], []
    for rec in per_sample.values():
        check = rec.get('orientation_check')
        if check:
            lifts_xy.append(check['lift_xy'])
            lifts_wrong.append(check['lift_swapped'])
    if lifts_xy:
        wins = sum(1 for a, b in zip(lifts_xy, lifts_wrong) if a > b)
        print(f'  orientation check P(measured|fg)/P(measured): x-y grid '
              f'{_mean(lifts_xy):.2f} vs axes-swapped mask '
              f'{_mean(lifts_wrong):.2f} (x-y must win; it does in '
              f'{wins}/{len(lifts_xy)} samples)')
    else:
        print('  orientation check: NOT EVALUABLE (no Doppler canvas; the '
              'x-y grid is\n  derived from the captured feature reversed, a '
              'code-pinned fact of the\n  shared R4Det permute — confirmed '
              'empirically on the motion-chain family)')
    print('\nglobal ratios (cross-check against the documented P1 readout):')
    for mode in VARIANT_LABELS:
        head = _mean([rec['head_ratio_global'][mode]
                      for rec in per_sample.values()])
        feat = _mean([rec['feature_ratio_global'][mode]
                      for rec in per_sample.values()])
        print(f'  {mode:11s} head_ratio={head:.4f} feature_ratio={feat:.4f}')
    header = f'{"region":12s} {"cells%":>7s}'
    for mode in VARIANT_LABELS:
        header += f' | {mode + " feat/cls rel":>26s}'
    print('\nrelative change per region (mean |delta| / region mean |base|):')
    print(header)
    summary = {}
    for name in regions:
        if name in thin:
            continue
        cells = _mean([rec['region_cells'].get(name, 0)
                       for rec in per_sample.values()])
        row = f'{name:12s} {100.0 * cells / total_cells:7.2f}'
        for mode in VARIANT_LABELS:
            feat = _mean([rec['variants'][mode]['feat'].get(name, {})
                          .get('rel', float('nan'))
                          for rec in per_sample.values()])
            cls = _mean([rec['variants'][mode]['cls'].get(name, {})
                         .get('rel', float('nan'))
                         for rec in per_sample.values()])
            row += f' | {feat:12.4f} {cls:12.4f}'
            summary.setdefault(mode, {})[name] = {'feat_rel': feat,
                                                 'cls_rel': cls}
        print(row)
    print('\npaired region contrasts (per sample, positive = fg more '
          'z-dependent):')
    for mode in ('zero_z', 'shuffle_z'):
        for metric in ('feat', 'cls'):
            for control in ('bg_hi', 'bg_top10', 'measured'):
                diffs = []
                for rec in per_sample.values():
                    node = rec['variants'][mode][metric]
                    if 'fg' not in node or control not in node:
                        continue
                    a, b = node['fg']['rel'], node[control]['rel']
                    if a == a and b == b:
                        diffs.append(a - b)
                wins = sum(1 for d in diffs if d > 0)
                print(f'  {mode:9s} {metric} fg - {control:9s}: positive in '
                      f'{wins}/{len(diffs)} samples, mean diff '
                      f'{_mean(diffs):+.4f}')
    print('\nReading: a foreground-only state can only pay off if fg dependence '
          'exceeds the\nenergy-matched background controls. fg >> bg AND '
          'fg >> bg_hi/bg_top10 is the\n"z is under-used on objects" signature '
          'the idea needs; fg on par with the\ncontrols means the bottleneck '
          'is the read-out, not the region coverage of the state.')
    return summary


def main():
    from mmcv import Config
    from mmcv.runner import load_checkpoint

    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_detector

    import numpy as np

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
        'figures_path': '/tmp/r4det_zutil_region_figures',
        'project_name': 'TJ4D-zutil-region',
    }
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    cfg.data.val.pop('samples_per_gpu', None)
    cfg.data.val.pop('workers_per_gpu', None)
    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().eval()

    class _Args(object):
        pass

    selection = _Args()
    selection.require_valid_history = True
    selection.start_index = args.start_index
    selection.limit = args.limit
    selection.min_valid_history = args.min_valid_history
    indices = dz.select_indices(dataset, selection)

    per_sample, grid_xy, has_doppler = diagnose(
        model, dataset, indices, args.gpu_id, list(args.pc_range),
        args.margin_cells)
    summary = summarize(per_sample, indices, grid_xy, args.checkpoint,
                        has_doppler)
    result = {
        'probe': 'region-split z_t dependency',
        'config': args.config,
        'checkpoint': args.checkpoint,
        'indices': indices,
        'grid_xy': list(grid_xy),
        'has_doppler_regions': bool(has_doppler),
        'pc_range': list(args.pc_range),
        'margin_cells': args.margin_cells,
        'intervention': intervention_class(
            model.temporal_fusion).__name__,
        'summary': summary,
        'per_sample': per_sample,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(f'\n[json] {output}')


if __name__ == '__main__':
    main()
