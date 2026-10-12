#!/usr/bin/env python
"""C1 training-curve check (prereg appendix A).

Two uses:

  * early check  (default ``--window 1-6``): compare C1's in-process val curve
    against B0's recorded curve for the same seed/schedule window.  Stop rule:
    C1 window mean < B0 window mean - 2.0 points => stop for review.
  * mature gate  (``--window 12-16``): frozen task gate
    Overall window mean >= 40.00, four-class non-inferiority
    (Car >= 47.34, Truck >= 29.22, Ped >= 29.93, Cyc >= 47.63),
    stop-loss if any class <= B0 same-window mean - 2.0.

Always reports the three frozen calibers: interval mean (fixed window, n
written explicitly), all-epoch peak ("value @ epoch"), best saved
("value @ epoch", epochs with an epoch_N.pth on disk only).

Read-only.  Does not touch weights.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

CLASS_KEYS = {
    'Car': 'pts_bbox/KITTI/Car_3D_moderate_strict',
    'Truck': 'pts_bbox/KITTI/Truck_3D_moderate_strict',
    'Pedestrian': 'pts_bbox/KITTI/Pedestrian_3D_moderate_loose',
    'Cyclist': 'pts_bbox/KITTI/Cyclist_3D_moderate_loose',
}
OVERALL = 'pts_bbox/KITTI/Overall_3D_moderate'
GATES = {
    'Overall_min': 40.00,
    'Car_min': 47.34,
    'Truck_min': 29.22,
    'Pedestrian_min': 29.93,
    'Cyclist_min': 47.63,
    'stop_loss_delta_points': -2.0,
    'early_stop_delta_points': -2.0,
    'early_stop_window': '1-6',
}


def _newest_json(path):
    if os.path.isfile(path) and path.endswith('.json'):
        return path
    cands = sorted(glob.glob(os.path.join(path, '*.log.json')))
    if not cands:
        raise SystemExit('no *.log.json found under %s' % path)
    # newest by mtime, then name (timestamps in the name sort fine)
    cands.sort(key=lambda p: (os.path.getmtime(p), p))
    return cands[-1]


def _load_val_rows(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = []
    for r in rows:
        if r.get('mode') == 'val' and OVERALL in r and r.get('epoch'):
            out.append(r)
    if not out:
        raise SystemExit('no val rows with %s in %s' % (OVERALL, path))
    # de-dup by epoch keeping the last occurrence
    by_ep = {}
    for r in out:
        by_ep[int(r['epoch'])] = r
    return [by_ep[e] for e in sorted(by_ep)]


def _vals(rows, epochs):
    sel = [r for r in rows if int(r['epoch']) in epochs]
    per = {k: [float(r[key]) for r in sel] for k, key in CLASS_KEYS.items()}
    per['Overall'] = [float(r[OVERALL]) for r in sel]
    return sel, {k: (statistics.mean(v) if v else None) for k, v in per.items()}


def _saved_epochs(work_dir):
    if not work_dir or not os.path.isdir(work_dir):
        return []
    eps = []
    for p in glob.glob(os.path.join(work_dir, 'epoch_*.pth')):
        m = re.search(r'epoch_(\d+)\.pth$', p)
        if m:
            eps.append(int(m.group(1)))
    return sorted(eps)


def _peak(rows):
    best = max(rows, key=lambda r: float(r[OVERALL]))
    return float(best[OVERALL]), int(best['epoch'])


def _best_saved(rows, saved):
    if not saved:
        return None, None
    sel = [r for r in rows if int(r['epoch']) in saved]
    if not sel:
        return None, None
    best = max(sel, key=lambda r: float(r[OVERALL]))
    return float(best[OVERALL]), int(best['epoch'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True,
                    help='C1 text log, .log.json, or a work dir')
    ap.add_argument('--work-dir', default=None,
                    help='C1 run dir (for best-saved epoch discovery)')
    ap.add_argument('--b0-log', required=True,
                    help='B0 .log.json or the B0 run dir')
    ap.add_argument('--window', default='1-6', help='e.g. 1-6 or 12-16')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.window.split('-'))
    epochs = list(range(lo, hi + 1))

    c1_path = _newest_json(args.log)
    b0_path = _newest_json(args.b0_log)
    c1_rows = _load_val_rows(c1_path)
    b0_rows = _load_val_rows(b0_path)

    c1_sel, c1_mean = _vals(c1_rows, epochs)
    b0_sel, b0_mean = _vals(b0_rows, epochs)

    saved = _saved_epochs(args.work_dir)
    c1_peak, c1_peak_ep = _peak(c1_rows)
    c1_bs, c1_bs_ep = _best_saved(c1_rows, saved)

    n_c1, n_b0 = len(c1_sel), len(b0_sel)
    span = ('%d-%d' % (epochs[0], epochs[-1])) if n_c1 == len(epochs) \
        else ('incomplete: observed epochs %s' % [int(r['epoch']) for r in c1_sel])

    delta = {k: (None if (c1_mean[k] is None or b0_mean[k] is None)
                 else round(c1_mean[k] - b0_mean[k], 4))
             for k in c1_mean}

    report = {
        'window': args.window,
        'c1_log_json': c1_path,
        'b0_log_json': b0_path,
        'interval_mean': {
            'caliber': 'fixed-window equal-weight mean (caliber 1/3)',
            'c1_n': n_c1, 'b0_n': n_b0, 'span': span,
            'c1': {k: (None if v is None else round(v, 4)) for k, v in c1_mean.items()},
            'b0': {k: (None if v is None else round(v, 4)) for k, v in b0_mean.items()},
            'delta_c1_minus_b0': delta,
            'c1_epochs': {int(r['epoch']): round(float(r[OVERALL]), 4) for r in c1_sel},
        },
        'all_epoch_peak': {
            'caliber': 'highest val point over all epochs (caliber 2/3)',
            'overall': c1_peak, 'epoch': c1_peak_ep,
        },
        'best_saved': {
            'caliber': 'highest val point among epochs with weights on disk (caliber 3/3)',
            'overall': c1_bs, 'epoch': c1_bs_ep, 'saved_epochs': saved,
        },
        'gates': GATES,
    }

    if lo >= 10:  # mature task gate
        checks = {'Overall_ge_40': c1_mean['Overall'] is not None and c1_mean['Overall'] >= GATES['Overall_min']}
        for k in CLASS_KEYS:
            checks['%s_non_inferior' % k] = (
                c1_mean[k] is not None and c1_mean[k] >= GATES['%s_min' % k])
            checks['%s_stop_loss' % k] = (
                None if (c1_mean[k] is None or b0_mean[k] is None)
                else bool(c1_mean[k] - b0_mean[k] > GATES['stop_loss_delta_points']))
        checks['n_complete'] = n_c1 == 5
        report['task_gate'] = checks
        report['task_gate_all_pass'] = bool(
            checks['Overall_ge_40'] and checks['n_complete']
            and all(v for k, v in checks.items()
                    if k.endswith('non_inferior') and v is not None)
            and all(v for k, v in checks.items() if k.endswith('stop_loss')))
    else:  # early curve check
        ok = (c1_mean['Overall'] is not None and b0_mean['Overall'] is not None
              and c1_mean['Overall'] - b0_mean['Overall'] >= GATES['early_stop_delta_points'])
        report['early_check'] = {
            'rule': 'C1 window mean - B0 window mean >= %s' % GATES['early_stop_delta_points'],
            'delta_overall_points': None if (c1_mean['Overall'] is None or b0_mean['Overall'] is None)
            else round(c1_mean['Overall'] - b0_mean['Overall'], 4),
            'pass': bool(ok),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2, sort_keys=True)

    # human summary
    print('[curve-check] window=%s c1_n=%d b0_n=%d span=%s' % (args.window, n_c1, n_b0, span))
    hdr = '%-12s %10s %10s %10s' % ('metric', 'C1', 'B0', 'delta')
    print(hdr)
    for k in ['Overall', 'Car', 'Truck', 'Pedestrian', 'Cyclist']:
        print('%-12s %10s %10s %10s' % (
            k,
            'n/a' if c1_mean[k] is None else '%.4f' % c1_mean[k],
            'n/a' if b0_mean[k] is None else '%.4f' % b0_mean[k],
            'n/a' if delta[k] is None else '%+.4f' % delta[k]))
    print('[caliber2 peak] %.4f @ ep%d' % (c1_peak, c1_peak_ep))
    if c1_bs is not None:
        print('[caliber3 best-saved] %.4f @ ep%d (saved=%s)' % (c1_bs, c1_bs_ep, saved))
    else:
        print('[caliber3 best-saved] n/a (no epoch_*.pth yet)')
    if 'task_gate' in report:
        print('[task-gate] all_pass=%s %s' % (
            report['task_gate_all_pass'],
            {k: v for k, v in report['task_gate'].items()}))
    if 'early_check' in report:
        print('[early-check] pass=%s delta=%+.4f' % (
            report['early_check']['pass'], report['early_check']['delta_overall_points']))
    print('wrote', args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
