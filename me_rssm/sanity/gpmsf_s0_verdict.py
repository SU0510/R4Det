#!/usr/bin/env python
"""S0 wiring gate for the GP-MSF arm (prereg §5).

Artifact-only (no GPU, no training): reads the 450-iter smoke log, its
checkpoint, the arm config, the wiring-probe JSON and the existing baseline
front-end smokes for a throughput/memory reference.

  S0-1  smoke reached iter 450, iter_450.pth exists, no NaN in any logged loss
  S0-2  single-variable contract: only ``RCFusion`` differs from the mainline,
        the fresh arm's residual projections are zero-initialised and its
        fusion conv is shape-identical to the mainline's
  S0-3  the geometry path really ran: ``proj_r2c`` / ``proj_i2r`` left their zero
        initialisation during the smoke (they are only reached through the
        native multi-scale evidence path -- an unwired arm would keep them at
        exactly zero)
  S0-4  frozen capacity: new params 177,726 and shared fusion_conv 1,475,072

Memory / step time are reported against the existing baseline-front-end smokes
as a review flag (``S0_PERF_REVIEW``); they are not an automatic stop because a
450-iter window on a shared machine is noisy (prereg §5).
"""
import argparse
import glob
import json
import os
import sys

import torch
from mmcv import Config

ARM_CFG = ('me_rssm/configs/'
           'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_gpmsf.py')
MAIN_CFG = ('configs/r4det/'
            'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py')
FROZEN = {
    'max_iters': 450,
    'new_params': 177726,
    'fusion_conv_params': 1475072,
    'height_refs': (-1.1230436265, -0.3003779650, 0.6601446867),
    'perf_ratio_cap': 1.2,
}
REF_SMOKES = [
    '/data/lurui/work_dirs/c1valid_smoke_450iter',
    '/data/lurui/work_dirs/dcrssm1_smoke_450iter',
    '/data/lurui/work_dirs/dcrssm2_smoke_450iter',
]


def _rows(work_dir):
    f = sorted(glob.glob(os.path.join(work_dir, '*.log.json')))[-1]
    return f, [json.loads(line) for line in open(f) if line.strip()]


def _perf(work_dir):
    try:
        _, rows = _rows(work_dir)
    except Exception:                                          # noqa: BLE001
        return None
    iters, mems = [], []
    for r in rows:
        if isinstance(r.get('time'), (int, float)):
            iters.append(float(r['time']))
        if isinstance(r.get('memory'), (int, float)):
            mems.append(float(r['memory']))
    if not iters:
        return None
    tail = iters[-100:]
    return {
        'n_time_rows': len(iters),
        'mean_iter_time_s': sum(tail) / len(tail),
        'max_memory_mib': max(mems) if mems else None,
    }


def _median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    n = len(values)
    return values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    log_json, rows = _rows(args.work_dir)
    loss_keys = [k for k in rows[-1] if k.startswith('loss') or k.startswith('stat_')]
    nans, max_iter = [], 0
    for r in rows:
        if r.get('iter') is not None:
            max_iter = max(max_iter, r['iter'])
        for k in loss_keys:
            v = r.get(k)
            if isinstance(v, (int, float)) and v != v:
                nans.append((r.get('iter'), k))
    ckpt_path = os.path.join(args.work_dir, 'iter_450.pth')
    s0_1 = bool(max_iter >= FROZEN['max_iters'] and os.path.exists(ckpt_path)
                and not nans)

    arm = Config.fromfile(ARM_CFG)
    main = Config.fromfile(MAIN_CFG)
    diff = sorted(k for k in set(arm.model) | set(main.model)
                  if arm.model.get(k) != main.model.get(k))
    fusion_cfg = arm.model['RCFusion']
    height_refs = tuple(float(z) for z in fusion_cfg['height_refs'])

    import me_rssm                                           # noqa: F401
    from mmdet3d.models.builder import FUSION_LAYERS
    fresh = FUSION_LAYERS.build(dict(fusion_cfg))
    baseline = FUSION_LAYERS.build(dict(main.model['RCFusion']))
    fresh_zero = all(float(m.weight.abs().sum()) == 0.0 and float(m.bias.abs().sum()) == 0.0
                     for m in (fresh.proj_r2c, fresh.proj_i2r))
    conv_shape_ok = (list(fresh.fusion_conv.conv.weight.shape)
                     == list(baseline.fusion_conv.conv.weight.shape))
    new_params = sum(p.numel() for p in fresh.parameters()) \
        - sum(p.numel() for p in fresh.fusion_conv.parameters())
    conv_params = sum(p.numel() for p in fresh.fusion_conv.parameters())
    s0_2 = bool(diff == ['RCFusion'] and fresh_zero and conv_shape_ok
                and fusion_cfg['type'] == 'GeometryPairedMultiScaleFusion'
                and main.model['RCFusion']['type'] == 'ConcatConvFusion'
                and height_refs == FROZEN['height_refs']
                and bool(fusion_cfg.get('require_geo')))

    proj_stats, conv_moved = None, None
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location='cpu')
        sd = sd.get('state_dict', sd)
        proj_stats = {}
        for name in ('proj_r2c', 'proj_i2r'):
            for key in ('weight', 'bias'):
                t = sd.get(f'cross_attention.{name}.{key}')
                if t is not None:
                    proj_stats[f'{name}.{key}'] = {
                        'abs_mean': float(t.abs().mean()),
                        'abs_max': float(t.abs().max()),
                    }
        cv = sd.get('cross_attention.fusion_conv.conv.weight')
        if cv is not None:
            base_cv = sd.get('cross_attention.fusion_conv.conv.weight')
            conv_moved = {'abs_mean': float(base_cv.abs().mean())}
    proj_left_zero = bool(proj_stats) and all(
        v['abs_mean'] > 0.0 for k, v in proj_stats.items() if k.endswith('weight'))
    s0_3 = bool(proj_left_zero)

    s0_4 = bool(new_params == FROZEN['new_params']
                and conv_params == FROZEN['fusion_conv_params'])

    smoke_perf = _perf(args.work_dir)
    ref_perf = [_perf(d) for d in REF_SMOKES]
    ref_time = _median([p['mean_iter_time_s'] for p in ref_perf if p])
    ref_mem = _median([p['max_memory_mib'] for p in ref_perf if p])
    time_ratio = (smoke_perf['mean_iter_time_s'] / ref_time
                  if smoke_perf and ref_time else None)
    mem_ratio = (smoke_perf['max_memory_mib'] / ref_mem
                 if smoke_perf and smoke_perf.get('max_memory_mib') and ref_mem
                 else None)
    perf_review = bool((time_ratio is not None and time_ratio > FROZEN['perf_ratio_cap'])
                       or (mem_ratio is not None and mem_ratio > FROZEN['perf_ratio_cap']))

    wiring = None
    wiring_path = os.path.join(args.work_dir, 'gpmsf_wiring_stats.json')
    if os.path.exists(wiring_path):
        with open(wiring_path) as f:
            wiring = json.load(f)
        recs = [r for r in wiring.get('records', []) if r.get('stats')]
        wiring = {
            'collect_stats': wiring.get('collect_stats'),
            'require_geo': wiring.get('require_geo'),
            'n_records_with_stats': len(recs),
            'first': recs[0]['stats'] if recs else None,
            'last': recs[-1]['stats'] if recs else None,
        }

    verdict = {
        'log_json': log_json, 'max_iter': max_iter, 'rows': len(rows),
        'nan_entries': nans, 'config_diff_keys': diff,
        'fresh_residual_zero_init': fresh_zero,
        'fusion_conv_shape_ok': conv_shape_ok,
        'fusion_conv_shape_fresh': list(fresh.fusion_conv.conv.weight.shape),
        'new_params': new_params, 'fusion_conv_params': conv_params,
        'checkpoint_projection_stats': proj_stats,
        'checkpoint_fusion_conv': conv_moved,
        'height_refs_frozen': height_refs == FROZEN['height_refs'],
        'smoke_perf': smoke_perf, 'ref_perf_median': {
            'mean_iter_time_s': ref_time, 'max_memory_mib': ref_mem},
        'perf_ratio_time': time_ratio, 'perf_ratio_memory': mem_ratio,
        'S0_PERF_REVIEW': perf_review,
        'wiring_probe': wiring,
        'S0_1': s0_1, 'S0_2': s0_2, 'S0_3': s0_3, 'S0_4': s0_4,
        'S0_ALL_PASS': bool(s0_1 and s0_2 and s0_3 and s0_4),
        'frozen': FROZEN,
    }
    out = args.out or os.path.join(args.work_dir, 's0_verdict.json')
    with open(out, 'w') as f:
        json.dump(verdict, f, indent=2, sort_keys=True)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print('S0_ALL_PASS =', verdict['S0_ALL_PASS'])
    return 0 if verdict['S0_ALL_PASS'] else 1


if __name__ == '__main__':
    sys.exit(main())
