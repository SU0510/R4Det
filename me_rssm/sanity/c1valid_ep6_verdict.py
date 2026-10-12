#!/usr/bin/env python
"""ep6 mechanism-gate verdict for the C1 arm (prereg section 5).

Same epoch / same seed / same offline harness, C1 vs the standard no2d_igdr
baseline seed0. Frozen clauses:

  M1  covered-group recall (pooled, all distances):  C1 - B0 >= 0.0 points
  M2  natural-zero-group recall:                     C1 - B0 >= -0.5 points
  M3  per-class offline-harness 3D AP non-inferiority at ep6 (class-quality
      clause; see the prereg addendum): every class >= B0 - 0.5 AP

Reported but not judged: low/mid/high recalls, frame-level vs object-level
zero split, per-class recall deltas.
"""
import argparse
import json
import sys

THRESH = {'M1_covered_min_delta_points': 0.0,
          'M2_zero_min_delta_points': -0.5,
          'M3_per_class_min_delta_ap': -0.5}

CLASS_KEYS = {
    'Car': 'pts_bbox/KITTI/Car_3D_moderate_strict',
    'Truck': 'pts_bbox/KITTI/Truck_3D_moderate_strict',
    'Pedestrian': 'pts_bbox/KITTI/Pedestrian_3D_moderate_loose',
    'Cyclist': 'pts_bbox/KITTI/Cyclist_3D_moderate_loose',
}


def _recall(probe, *path):
    cur = probe
    for p in path:
        cur = cur[p]
    return None if cur is None else 100.0 * float(cur['recall'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c1-probe', required=True)
    ap.add_argument('--b0-probe', required=True)
    ap.add_argument('--c1-ap', required=True)
    ap.add_argument('--b0-ap', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    c1 = json.load(open(args.c1_probe))
    b0 = json.load(open(args.b0_probe))
    ap_c1 = json.load(open(args.c1_ap))
    ap_b0 = json.load(open(args.b0_ap))

    out = {'thresholds': THRESH, 'recall_points': {}, 'ap': {}}
    for grp in ['zero', 'low', 'mid', 'high']:
        r_c1 = _recall(c1, 'primary', 'recall_by_coverage_group', grp)
        r_b0 = _recall(b0, 'primary', 'recall_by_coverage_group', grp)
        out['recall_points'][grp] = {
            'c1': r_c1, 'b0': r_b0,
            'delta': None if (r_c1 is None or r_b0 is None) else r_c1 - r_b0}
    # covered = the pooled population with coverage > 0
    cov_c1 = _recall(c1, 'primary', 'recall_by_zero_scope',
                     'not_applicable (coverage > 0)')
    cov_b0 = _recall(b0, 'primary', 'recall_by_zero_scope',
                     'not_applicable (coverage > 0)')
    out['recall_points']['covered'] = {
        'c1': cov_c1, 'b0': cov_b0,
        'delta': None if (cov_c1 is None or cov_b0 is None) else cov_c1 - cov_b0}

    for name, key in CLASS_KEYS.items():
        v_c1, v_b0 = ap_c1.get(key), ap_b0.get(key)
        out['ap'][name] = {
            'key': key, 'c1': v_c1, 'b0': v_b0,
            'delta': None if (v_c1 is None or v_b0 is None) else v_c1 - v_b0}

    d_cov = out['recall_points']['covered']['delta']
    d_zero = out['recall_points']['zero']['delta']
    worst_ap = min((v['delta'] for v in out['ap'].values()
                    if v['delta'] is not None), default=None)
    out['M1_covered'] = bool(d_cov is not None
                             and d_cov >= THRESH['M1_covered_min_delta_points'])
    out['M2_zero'] = bool(d_zero is not None
                          and d_zero >= THRESH['M2_zero_min_delta_points'])
    out['M3_per_class_ap'] = bool(
        worst_ap is not None and worst_ap >= THRESH['M3_per_class_min_delta_ap']
        and all(v['delta'] is not None for v in out['ap'].values()))
    out['ALL_PASS'] = bool(out['M1_covered'] and out['M2_zero']
                           and out['M3_per_class_ap'])
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0 if out['ALL_PASS'] else 1


if __name__ == '__main__':
    sys.exit(main())
