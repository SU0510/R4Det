#!/usr/bin/env python
"""Merge every val row scattered across a run dir's *.log.json files and report
the three frozen calibers.

Needed because a resumed run writes a NEW timestamped log.json: ep1-6 live in the
original file, ep7+ in the resumed one.  Read-only wrt weights.

    python me_rssm/sanity/merge_val_curves.py --work-dir DIR \
        [--b0-log DIR_OR_FILE] [--out merged.log.json] [--json-out report.json]

``--out`` writes a JSONL file in the same shape as mmdet's log.json (val rows
only) so downstream read-only tools can consume the union directly.
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


def _logs(work_dir):
    out = []
    for p in glob.glob(os.path.join(work_dir, '*.log.json')):
        if os.path.basename(p).startswith('merged'):
            continue
        out.append(p)
    out.sort(key=lambda p: (os.path.getmtime(p), p))  # later file wins
    return out


def _resolve(path):
    if os.path.isfile(path) and path.endswith('.json'):
        return path
    c = sorted(glob.glob(os.path.join(path, '*.log.json')),
               key=lambda p: (os.path.getmtime(p), p))
    if not c:
        raise SystemExit('no *.log.json under %s' % path)
    return c[-1]


def curate(work_dir):
    by_ep = {}
    src = {}
    for f in _logs(work_dir):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('mode') == 'val' and OVERALL in r and r.get('epoch'):
                ep = int(r['epoch'])
                by_ep[ep] = r
                src[ep] = os.path.basename(f)
    return by_ep, src


def saved_epochs(work_dir):
    return sorted(int(m.group(1)) for m in
                  (re.search(r'epoch_(\d+)\.pth$', p)
                   for p in glob.glob(os.path.join(work_dir, 'epoch_*.pth')))
                  if m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--b0-log', default=None)
    ap.add_argument('--window', default='12-16')
    ap.add_argument('--out', default=None)
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    by_ep, src = curate(args.work_dir)
    if not by_ep:
        raise SystemExit('no val rows in %s' % args.work_dir)
    eps = sorted(by_ep)
    lo, hi = (int(x) for x in args.window.split('-'))

    if args.out:
        with open(args.out, 'w') as f:
            for e in eps:
                f.write(json.dumps(by_ep[e]) + '\n')

    def mean_of(eps_):
        return {k: (statistics.mean([by_ep[e][key] for e in eps_ if key in by_ep[e]])
                    if any(key in by_ep[e] for e in eps_) else None)
                for k, key in list(CLASS_KEYS.items()) + [('Overall', OVERALL)]}

    win = [e for e in range(lo, hi + 1) if e in by_ep]
    m = mean_of(win)
    peak_ep = max(eps, key=lambda e: by_ep[e][OVERALL])
    saved = saved_epochs(args.work_dir)
    bs = [e for e in saved if e in by_ep]
    bs_ep = max(bs, key=lambda e: by_ep[e][OVERALL]) if bs else None

    print('[merge] dir=%s  epochs=%s  (sources: %s)'
          % (args.work_dir, eps, sorted(set(src.values()))))
    print('%-6s %9s %9s %9s %9s %9s' % ('epoch', 'Overall', 'Car', 'Truck', 'Ped', 'Cyc'))
    for e in eps:
        r = by_ep[e]
        print('%-6d %9.4f %9.4f %9.4f %9.4f %9.4f'
              % (e, r[OVERALL], r[CLASS_KEYS['Car']], r[CLASS_KEYS['Truck']],
                 r[CLASS_KEYS['Pedestrian']], r[CLASS_KEYS['Cyclist']]))
    rep = {'work_dir': args.work_dir, 'epochs': eps, 'sources': src,
           'per_epoch': {e: {k: by_ep[e][key] for k, key in
                             list(CLASS_KEYS.items()) + [('Overall', OVERALL)]}
                         for e in eps}}
    n = len(win)
    span = '%d-%d' % (lo, hi) if n == (hi - lo + 1) else \
        'incomplete: observed %s' % win
    print('\n[caliber1 interval mean] window=%s n=%d span=%s' % (args.window, n, span))
    print('  ' + '  '.join('%s=%.4f' % (k, m[k]) for k in
                           ['Overall', 'Car', 'Truck', 'Pedestrian', 'Cyclist']
                           if m[k] is not None))
    rep['interval_mean'] = {'window': args.window, 'n': n, 'span': span, 'values': m,
                            'caliber': 'fixed-window equal-weight mean'}
    print('[caliber2 all-epoch peak] %.4f @ ep%d (scope: ep%d-%d)'
          % (by_ep[peak_ep][OVERALL], peak_ep, eps[0], eps[-1]))
    rep['all_epoch_peak'] = {'overall': by_ep[peak_ep][OVERALL], 'epoch': peak_ep,
                             'scope': [eps[0], eps[-1]]}
    if bs_ep is None:
        print('[caliber3 best saved] n/a (no epoch_*.pth)')
    else:
        print('[caliber3 best saved] %.4f @ ep%d (saved=%s)'
              % (by_ep[bs_ep][OVERALL], bs_ep, saved))
        rep['best_saved'] = {'overall': by_ep[bs_ep][OVERALL], 'epoch': bs_ep,
                             'saved_epochs': saved}

    if args.b0_log:
        b0f = _resolve(args.b0_log)
        b0 = {}
        for line in open(b0f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get('mode') == 'val' and OVERALL in r and r.get('epoch'):
                b0[int(r['epoch'])] = r
        bw = [e for e in range(lo, hi + 1) if e in b0]
        bm = {k: statistics.mean([b0[e][key] for e in bw])
              for k, key in list(CLASS_KEYS.items()) + [('Overall', OVERALL)]}
        print('\n[vs B0 same window] source=%s n=%d' % (os.path.basename(b0f), len(bw)))
        for k in ['Overall', 'Car', 'Truck', 'Pedestrian', 'Cyclist']:
            if m[k] is None:
                continue
            print('  %-11s C=%8.4f  B0=%8.4f  delta=%+.4f' % (k, m[k], bm[k], m[k] - bm[k]))
        rep['vs_b0'] = {'b0_log': b0f, 'n': len(bw), 'b0': bm,
                        'delta': {k: (m[k] - bm[k]) for k in bm if m[k] is not None}}
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(rep, f, indent=2, sort_keys=True)
        print('\nwrote', args.json_out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
