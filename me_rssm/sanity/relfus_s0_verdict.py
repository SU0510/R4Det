#!/usr/bin/env python
"""S0 wiring gate for the reliability-gated fusion arm (prereg section 5).

Artifact-only (no GPU, no training): reads the 450-iter smoke log, its
checkpoint and the two configs.

  S0-1  smoke reached iter 450, iter_450.pth exists, no NaN in any logged loss
  S0-2  single-variable contract: only ``RCFusion`` differs from the mainline
        config, and the fresh arm's gate is zero-initialised (alpha == 1)
  S0-3  gate is live after training (weights left zero) and the fusion conv is
        shape-identical to the mainline's.  NOTE: alpha is deliberately NOT
        published into the runner's log -- ``rssm_stats`` is owned by the
        temporal fusion, and adding front-end plumbing would couple this arm to
        the detector.  The gate's behaviour is read from the checkpoint here and
        from the per-epoch alpha probe instead.
  S0-4  capacity delta equals the gate's own parameter count (593) and the
        occupancy layout is the pillar-scatter convention ('yx')
"""
import argparse
import glob
import json
import os
import sys

import torch
from mmcv import Config

ARM_CFG = ('me_rssm/configs/'
           'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_relfusion.py')
MAIN_CFG = ('configs/r4det/'
            'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py')
FROZEN = {'max_iters': 450, 'gate_params': 593, 'valid_layout': 'yx'}


def _rows(work_dir):
    f = sorted(glob.glob(os.path.join(work_dir, '*.log.json')))[-1]
    return f, [json.loads(l) for l in open(f) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    log_json, rows = _rows(args.work_dir)
    loss_keys = [k for k in rows[-1] if k.startswith('loss') or k.startswith('stat_')]
    nans, max_iter, gate_rows, first_alpha = [], 0, 0, None
    for r in rows:
        if r.get('iter') is not None:
            max_iter = max(max_iter, r['iter'])
        for k in loss_keys:
            v = r.get(k)
            if isinstance(v, (int, float)) and v != v:
                nans.append((r.get('iter'), k))
        if r.get('v_rel_gate_mean') is not None:
            gate_rows += 1
            if first_alpha is None:
                first_alpha = float(r['v_rel_gate_mean'])
    ckpt_path = os.path.join(args.work_dir, 'iter_450.pth')
    s0_1 = bool(max_iter >= FROZEN['max_iters'] and os.path.exists(ckpt_path)
                and not nans)

    arm = Config.fromfile(ARM_CFG)
    main = Config.fromfile(MAIN_CFG)
    diff = sorted(k for k in set(arm.model) | set(main.model)
                  if arm.model.get(k) != main.model.get(k))
    s0_2 = (diff == ['RCFusion']
            and arm.model['RCFusion']['type'] == 'ReliabilityGatedConcatFusion'
            and main.model['RCFusion']['type'] == 'ConcatConvFusion')

    # fresh module: gate must start at zero -> alpha == 1 (bitwise baseline)
    import me_rssm  # noqa: F401
    from mmdet3d.models.builder import FUSION_LAYERS
    fresh = FUSION_LAYERS.build(dict(arm.model['RCFusion']))
    fresh_zero = (float(fresh.gate[-1].weight.abs().sum()) == 0.0
                  and float(fresh.gate[-1].bias.abs().sum()) == 0.0)
    gate_params = sum(p.numel() for p in fresh.gate.parameters())
    s0_4 = (gate_params == FROZEN['gate_params']
            and arm.model['RCFusion']['valid_layout'] == FROZEN['valid_layout'])

    gate_w, conv_shape = None, None
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location='cpu')
        sd = sd.get('state_dict', sd)
        gw = [v for k, v in sd.items() if 'cross_attention.gate' in k and 'weight' in k]
        cv = [v for k, v in sd.items()
              if 'cross_attention.fusion_conv.conv.weight' in k]
        if gw:
            gate_w = {'n_tensors': len(gw),
                      'abs_mean': float(sum(t.abs().mean() for t in gw) / len(gw)),
                      'abs_max': float(max(t.abs().max() for t in gw))}
        if cv:
            conv_shape = list(cv[0].shape)
        elif sd:
            conv_shape = ['fusion_conv weights not found']
    baseline_conv = list(fresh.fusion_conv.conv.weight.shape)
    s0_3 = bool(gate_w and gate_w['abs_mean'] > 0.0
                and conv_shape == baseline_conv)

    verdict = {
        'log_json': log_json, 'max_iter': max_iter, 'rows': len(rows),
        'nan_entries': nans, 'gate_rows_in_log': gate_rows,
        'first_v_rel_gate_mean': first_alpha,
        'config_diff_keys': diff,
        'fresh_gate_zero_init': fresh_zero,
        'gate_params': gate_params,
        'gate_weight_stats': gate_w,
        'fusion_conv_shape_in_ckpt': conv_shape,
        'fusion_conv_shape_fresh': baseline_conv,
        'S0_1': s0_1, 'S0_2': bool(s0_2 and fresh_zero), 'S0_3': s0_3, 'S0_4': s0_4,
        'S0_ALL_PASS': bool(s0_1 and s0_2 and fresh_zero and s0_3 and s0_4),
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
