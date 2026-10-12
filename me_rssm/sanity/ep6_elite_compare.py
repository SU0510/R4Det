#!/usr/bin/env python
"""C1's ep1-6 curve vs the same-family runs that actually matured well (CPU read-only).

Motivation (2026-10-12): the user asked to compare C1's ep1-6 结果 against the
ep6 of the historically *good* runs.  Section 84 already ranked C1 inside the
whole same-family set; this tool answers the narrower question --- how does C1
look next to only the runs whose **mature window (ep12-16)** was strong, and what
did those runs' ep6 values actually predict?

Read-only.  Never touches weights, log files are only parsed.

Calibers (AGENTS rule 5): interval mean (fixed window, n printed), all-epoch
peak, best saved.  This tool prints the interval means and single epochs only;
peak/best-saved stay in the run's own section of docs/training_runs_full.md.
"""
import argparse
import glob
import json
import os
import statistics as st

WD_ROOT = '/data/lurui/work_dirs'
OVERALL = 'pts_bbox/KITTI/Overall_3D_moderate'
CLASS_KEYS = [
    ('Car', 'pts_bbox/KITTI/Car_3D_moderate_strict'),
    ('Truck', 'pts_bbox/KITTI/Truck_3D_moderate_strict'),
    ('Ped', 'pts_bbox/KITTI/Pedestrian_3D_moderate_loose'),
    ('Cyc', 'pts_bbox/KITTI/Cyclist_3D_moderate_loose'),
]
# frozen mature task gate (prereg appendix A / section 80.4)
GATES = {'Car': 47.34, 'Truck': 29.22, 'Ped': 29.93, 'Cyc': 47.63}


def load_curve(work_dir):
    """epoch -> last val row (log.json is JSON-lines; resumed runs split files)."""
    out = {}
    files = sorted(glob.glob(os.path.join(work_dir, '*.log.json')),
                   key=lambda p: (os.path.getmtime(p), p))
    for f in files:
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('mode') == 'val' and OVERALL in row and row.get('epoch'):
                    out[int(row['epoch'])] = row
    return out


def window_mean(curve, lo, hi, key=OVERALL):
    vals = [float(curve[e][key]) for e in range(lo, hi + 1) if e in curve]
    if not vals:
        return None, 0
    return st.mean(vals), len(vals)


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=WD_ROOT)
    ap.add_argument('--pattern', default='fgfull_N4_*24e*',
                    help='same-family work-dir glob (default: fgfull_N4_*24e*)')
    ap.add_argument('--focus', default='fgfull_N4_no2d_igdr_2x4_24e_c1valid_seed0')
    ap.add_argument('--elite-min', type=float, default=40.00)
    ap.add_argument('--neighborhood', type=float, default=2.5)
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(args.root, args.pattern))
                  if os.path.isdir(d))
    runs = {}
    for d in dirs:
        curve = load_curve(d)
        if not curve:
            continue
        early, n_early = window_mean(curve, 1, 6)
        mature, n_mature = window_mean(curve, 12, 16)
        runs[os.path.basename(d)] = dict(curve=curve, early=early, n_early=n_early,
                                         mature=mature, n_mature=n_mature)
    if args.focus not in runs:
        raise SystemExit('focus run not found: %s' % args.focus)
    focus = runs[args.focus]
    f_ep6 = float(focus['curve'][6][OVERALL]) if 6 in focus['curve'] else None

    elite = sorted([n for n, r in runs.items()
                    if r['mature'] is not None and r['mature'] >= args.elite_min],
                   key=lambda n: -runs[n]['mature'])
    others = sorted([n for n in runs
                     if n not in elite and n != args.focus and runs[n]['mature'] is not None],
                    key=lambda n: -runs[n]['mature'])
    order = [args.focus] + elite + others

    def short(n):
        s = n.replace('fgfull_N4_', '').replace('_2x4_24e', '')
        if s.startswith('no2d_igdr_'):
            s = 'std_' + s[len('no2d_igdr_'):]
        if n == args.focus:
            s += '  <= C1 focus'
        return s

    print('== per-epoch Overall (ep1-6); mature = ep12-16 equal-weight mean ==')
    print('%-28s %6s %6s %6s %6s %6s %6s | %8s | %8s' %
          ('run', 'ep1', 'ep2', 'ep3', 'ep4', 'ep5', 'ep6', 'ep1-6', 'ep12-16'))
    for n in order:
        r = runs[n]
        cells = []
        for e in range(1, 7):
            cells.append('%6.2f' % float(r['curve'][e][OVERALL]) if e in r['curve'] else '     -')
        early = '%8.4f' % r['early'] if r['early'] else '%8s' % '-'
        mat = ('%8.4f (n=%d)' % (r['mature'], r['n_mature'])) if r['mature'] else '%8s' % 'pending'
        print('%-34s %s | %s | %-18s' % (short(n), ' '.join(cells), early, mat))

    if elite:
        print()
        print('== C1 minus elite runs, per epoch (Overall) ==')
        print('%-34s %6s %6s %6s %6s %6s %6s' % ('delta vs', 'ep1', 'ep2', 'ep3', 'ep4', 'ep5', 'ep6'))
        for n in elite:
            d = ['%+6.2f' % (float(focus['curve'][e][OVERALL]) - float(runs[n]['curve'][e][OVERALL]))
                 if (e in focus['curve'] and e in runs[n]['curve']) else '     -'
                 for e in range(1, 7)]
            print('%-34s %s' % (short(n), ' '.join(d)))
        elite_mean = [st.mean([float(runs[n]['curve'][e][OVERALL]) for n in elite
                               if e in runs[n]['curve']]) for e in range(1, 7)]
        print('%-34s %s' % ('elite mean', ' '.join('%6.2f' % v for v in elite_mean)))
        print('%-34s %s' % ('C1 minus elite mean',
                            ' '.join('%+6.2f' % (float(focus['curve'][e][OVERALL]) - elite_mean[e - 1])
                                     for e in range(1, 7))))
        if f_ep6 is not None:
            vals = sorted([f_ep6] + [float(runs[n]['curve'][6][OVERALL]) for n in elite
                                     if 6 in runs[n]['curve']], reverse=True)
            print()
            print('C1 ep6 rank inside [C1 + elite] at ep6: %d / %d (value %.4f, elite span %.2f-%.2f)'
                  % (vals.index(f_ep6) + 1, len(vals), f_ep6,
                     min(v for v in vals if v != f_ep6) if len(vals) > 1 else f_ep6,
                     max(vals)))
        em = [runs[n]['early'] for n in elite if runs[n]['early']]
        print('ep1-6 window mean: C1 %.4f vs elite mean %.4f (delta %+.4f, elite span %.4f-%.4f)'
              % (focus['early'], st.mean(em), focus['early'] - st.mean(em), min(em), max(em)))
        pairs = [(float(runs[n]['curve'][6][OVERALL]), runs[n]['mature']) for n in elite
                 if 6 in runs[n]['curve']]
        if len(pairs) >= 3:
            rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
            print('Spearman(ep6, ep12-16) inside elite subset: %+.3f  (n=%d; mature span %.2f)'
                  % (rho, len(pairs), max(p[1] for p in pairs) - min(p[1] for p in pairs)))

    print()
    print('== ep6 per-class: C1 vs elite (frozen mature gates in header) ==')
    print('%-34s %7s %7s %7s %7s' % ('run', *('%s>%.2f' % (k, GATES[k]) for k, _ in CLASS_KEYS)))
    for n in [args.focus] + elite:
        r = runs[n]['curve'].get(6)
        if not r:
            continue
        cells = []
        for k, key in CLASS_KEYS:
            v = float(r[key])
            cells.append('%7.2f%s' % (v, '*' if v >= GATES[k] else ' '))
        print('%-34s %s' % (short(n), ' '.join(cells)))
    print('(* = already at/above the frozen mature non-inferiority line at ep6)')

    if f_ep6 is not None:
        lo, hi = f_ep6 - args.neighborhood, f_ep6 + args.neighborhood
        band = [(n, float(runs[n]['curve'][6][OVERALL]), runs[n]['mature'])
                for n in runs if n != args.focus and runs[n]['mature'] is not None
                and 6 in runs[n]['curve'] and lo <= float(runs[n]['curve'][6][OVERALL]) <= hi]
        if band:
            print()
            print('== neighbourhood: same-family runs with ep6 in [%.2f, %.2f] ==' % (lo, hi))
            for n, e6, mt in sorted(band, key=lambda t: -t[1]):
                print('  %-34s ep6 %6.2f -> mature %6.4f  (%+.2f)' % (short(n), e6, mt, mt - e6))
            mts = [b[2] for b in band]
            print('  mature outcomes: n=%d  min %.4f  mean %.4f  max %.4f  std %.4f'
                  % (len(mts), min(mts), st.mean(mts), max(mts), st.pstdev(mts)))
    print()
    print('== typical ep6 -> ep12-16 drift per class (family runs with both windows) ==')
    drift = {}
    for k, key in CLASS_KEYS + [('Overall', OVERALL)]:
        ds = []
        for n in runs:
            r = runs[n]
            if r['mature'] is None or 6 not in r['curve']:
                continue
            mature, cnt = window_mean(r['curve'], 12, 16, key)
            if mature is None:
                continue
            ds.append(mature - float(r['curve'][6][key]))
        drift[k] = ds
        if ds:
            print('  %-8s n=%2d  mean %+6.3f  std %5.3f  min %+6.3f  max %+6.3f'
                  % (k, len(ds), st.mean(ds), st.pstdev(ds), min(ds), max(ds)))
    print('  (sign = mature window minus ep6 single epoch; C1 Ped/Cyc sitting below the')
    print('   mature gate line at ep6 is therefore not by itself a stop signal)')

    print()
    print("== projections for C1's mature window (no claim, calibration only) ==")
    fit = [(runs[n]['early'], runs[n]['mature']) for n in runs
           if runs[n]['mature'] is not None and runs[n]['early'] and n != args.focus]
    if len(fit) >= 5 and focus['early']:
        xs = [p[0] for p in fit]
        ys = [p[1] for p in fit]
        mx, my = st.mean(xs), st.mean(ys)
        den = sum((a - mx) ** 2 for a in xs)
        b = sum((a - mx) * (c - my) for a, c in zip(xs, ys)) / den if den else 0.0
        a0 = my - b * mx
        resid = [c - (a0 + b * a) for a, c in zip(xs, ys)]
        sd = st.pstdev(resid)
        pred = a0 + b * focus['early']
        print('  A) family regression (n=%d): mature = %.4f %+.4f * ep1-6  ->  %.4f '
              '[1sigma %.4f, %.4f]' % (len(fit), a0, b, pred, pred - sd, pred + sd))
    if f_ep6 is not None and 'Overall' in drift and drift['Overall']:
        d = drift['Overall']
        print('  B) ep6 + typical drift: %.4f %+.3f (std %.3f)  ->  %.4f'
              % (f_ep6, st.mean(d), st.pstdev(d), f_ep6 + st.mean(d)))
    print('  Neither projection is a result: both bands straddle the frozen Overall >= 40.00 line.')

    if focus['mature'] is None:
        print()
        print('NOTE: C1 has no ep12-16 window yet; its mature caliber is pending '
              '(window must complete with n=5 before any task-gate statement).')


if __name__ == '__main__':
    main()
