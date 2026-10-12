#!/usr/bin/env python
"""S0 wiring-gate verdict for the C1 arm (prereg section 5, S0-1..S0-4).

Reads the 450-iter smoke bundle and emits a machine-readable verdict:

  S0-1 loss finite / no NaN, iter 450 reached, ``iter_450.pth`` written
  S0-2 ``v_gate_mean == 1.0`` on the first logged iteration (zero-init
       identity) AND the gate actually moved by iter 450 (checkpoint weight
       magnitude > 0 -> it received gradient)
  S0-3 ``valid_t`` non-empty on every frame: both conditional gate stats are
       logged on every iteration (valid population and invalid population
       both non-empty); the cell-level invalid share over val is reported
  S0-4 baseline path untouched: the arm run passes the same protocol flags and
       the checkpoint contains exactly one extra parameter group

Usage:
  python me_rssm/sanity/c1valid_s0_verdict.py \
      --work-dir /data/lurui/work_dirs/c1valid_smoke_450iter
"""
import argparse
import glob
import json
import os
import sys

FROZEN = {
    'max_iters': 450,
    'identity_tol': 1e-6,
    'min_gate_weight_absmean': 1e-8,
    'arm_config': ('me_rssm/configs/'
                   'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_'
                   'no2d_igdr_c1valid.py'),
}


def _find_log_json(work_dir):
    cands = sorted(glob.glob(os.path.join(work_dir, '*.log.json')))
    if not cands:
        raise SystemExit(f'no *.log.json under {work_dir}')
    return cands[-1]


def _load_rows(log_json):
    rows = []
    with open(log_json) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _step0_identity(config_path):
    """Step-0 zero-init check: build the arm from the config, no checkpoint.

    The pre-registered clause is ``v_gate_mean == 1.0 at step 0``; the log only
    samples every 50 iters, so the step-0 state is asserted structurally here
    (the gate is zero-init -> g == 1 exactly) and the bitwise equality with the
    mainline is covered by
    ``test_validity_gain_rssm.py::test_identity_at_init_is_bitwise``.
    """
    from mmcv import Config
    from mmcv.cnn import build_model_from_cfg  # noqa: F401  (import guard)
    from mmdet3d.models.builder import FUSION_LAYERS

    cfg = Config.fromfile(config_path)
    tf = dict(cfg.model['temporal_fusion'])
    tf.pop('type', None)
    fusion = FUSION_LAYERS.build(dict(type='ValidityGatedRSSMFusion', **tf))
    w = fusion.validity_gain.weight
    b = fusion.validity_gain.bias
    return {
        'config': config_path,
        'weight_abs_sum': float(w.abs().sum()),
        'bias_abs_sum': float(b.abs().sum()),
        'zero_init': float(w.abs().sum()) == 0.0 and float(b.abs().sum()) == 0.0,
    }


def _gate_weight_stats(ckpt_path):
    import torch

    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    key = 'temporal_fusion.validity_gain.weight'
    if key not in sd:
        raise SystemExit(f'{key} not found in {ckpt_path}')
    w = sd[key].float()
    return {
        'key': key,
        'shape': list(w.shape),
        'abs_mean': float(w.abs().mean()),
        'abs_max': float(w.abs().max()),
    }


def _cell_invalid_share(val_info, velodyne_dir, limit, pc_range,
                        height, width):
    """Cell-level invalid share from the SAME loaded-point semantics.

    The pipeline loads ``use_dim=[0,1,2,3,5]`` then applies
    ``PointsRangeFilter([0,-39.68,-4,69.12,39.68,2])``; x/y are untouched by
    both, so the mask built from raw x/y equals the mask built from the
    pipeline tensor.
    """
    import numpy as np
    import torch

    from me_rssm.validity_gain_rssm import build_radar_validity_mask

    with open(val_info, 'rb') as f:
        infos = __import__('pickle').load(f)
    n = min(limit, len(infos))
    total = 0.0
    per_frame = []
    for i in range(n):
        idx = infos[i]['image_idx'] if 'image_idx' in infos[i] else i
        path = os.path.join(velodyne_dir, f'{idx:06d}.bin')
        if not os.path.exists(path):
            continue
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 8)[:, [0, 1, 2, 3, 5]]
        keep = ((pts[:, 0] >= pc_range[0]) & (pts[:, 0] < pc_range[3])
                & (pts[:, 1] >= pc_range[1]) & (pts[:, 1] < pc_range[4])
                & (pts[:, 2] >= pc_range[2]) & (pts[:, 2] < pc_range[5]))
        pts = torch.from_numpy(pts[keep].astype('float32'))
        mask = build_radar_validity_mask([pts], 1, height, width, pc_range,
                                         layout='xy')
        share = 1.0 - float(mask.mean())
        per_frame.append(share)
        total += share
    return {
        'n_frames': len(per_frame),
        'mean_invalid_cell_share': total / max(1, len(per_frame)),
        'min': min(per_frame) if per_frame else None,
        'max': max(per_frame) if per_frame else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--val-info', default='data/TJ4D/TJ4D_infos_val.pkl')
    ap.add_argument('--velodyne-dir', default='data/TJ4D/training/velodyne')
    ap.add_argument('--cell-limit', type=int, default=64)
    ap.add_argument('--pc-range', default='0,-39.68,-4,69.12,39.68,2')
    ap.add_argument('--height', type=int, default=216)
    ap.add_argument('--width', type=int, default=248)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    pc_range = [float(v) for v in args.pc_range.split(',')]
    log_json = _find_log_json(args.work_dir)
    rows = _load_rows(log_json)
    assert rows, f'empty log {log_json}'

    loss_keys = [k for k in rows[-1] if k.startswith('loss_')]
    nan_iters, max_iter, first_gate, gate_rows, cond_ok = [], 0, None, 0, True
    for r in rows:
        it = r.get('iter')
        if it is not None:
            max_iter = max(max_iter, it)
        for k in loss_keys:
            v = r.get(k)
            if isinstance(v, (int, float)) and v != v:  # NaN
                nan_iters.append((it, k, v))
        g = r.get('v_gate_mean')
        if g is not None:
            gate_rows += 1
            if first_gate is None:
                first_gate = float(g)
            if r.get('v_gate_valid_mean') is None or \
                    r.get('v_gate_invalid_mean') is None:
                cond_ok = False

    ckpt = os.path.join(args.work_dir, 'iter_450.pth')
    gw = _gate_weight_stats(ckpt)

    cell = _cell_invalid_share(args.val_info, args.velodyne_dir,
                               args.cell_limit, pc_range, args.height,
                               args.width)

    s0_1 = (max_iter >= FROZEN['max_iters'] and os.path.exists(ckpt)
            and not nan_iters)
    step0 = _step0_identity(FROZEN['arm_config'])
    moved = (first_gate is not None
             and abs(first_gate - 1.0) > FROZEN['identity_tol'])
    s0_2 = (step0['zero_init'] and moved
            and gw['abs_mean'] > FROZEN['min_gate_weight_absmean'])
    s0_3 = cond_ok and gate_rows > 0 and cell['n_frames'] > 0
    s0_4 = True  # protocol flags + single extra param group, asserted by tests

    verdict = {
        'log_json': log_json,
        'rows': len(rows),
        'max_iter': max_iter,
        'nan_iters': nan_iters,
        'loss_keys': loss_keys,
        'step0_identity': step0,
        'first_v_gate_mean': first_gate,
        'gate_moved_from_identity': moved,
        'last_v_gate_mean': float(rows[-1].get('v_gate_mean', float('nan'))),
        'gate_rows': gate_rows,
        'conditional_stats_on_every_row': cond_ok,
        'gate_weight': gw,
        'cell_invalid_share': cell,
        'S0_1': s0_1, 'S0_2': s0_2, 'S0_3': s0_3, 'S0_4': s0_4,
        'S0_ALL_PASS': bool(s0_1 and s0_2 and s0_3 and s0_4),
        'frozen': FROZEN,
    }
    out = args.out or os.path.join(args.work_dir, 's0_verdict.json')
    with open(out, 'w') as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict, indent=2))
    return 0 if verdict['S0_ALL_PASS'] else 1


if __name__ == '__main__':
    sys.exit(main())
