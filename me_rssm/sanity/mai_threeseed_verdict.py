#!/usr/bin/env python3
"""MAI three-seed paired verdict (read-only, CPU-only).

The chain watcher (watch_mai_seed1_then_seed2_ep16.sh) stops seed2 at ep16
and runs its probes; the three-seed paired verdict is a manual step. This
script freezes it: for each seed it reads the ep12-16 window mean of
Overall_3D_moderate from the MAI run dir and the SAME-SEED no2d_igdr
baseline run dir, prints the paired delta and the across-seed summary.

Pre-registered reading frame (docs/todo.md / STATUS.md):
  - judge the per-seed paired window mean, never single-point BESTs;
  - paired-delta noise bound is 0.42-0.53 (from the clean multi-seed
    logs), so |delta| < 0.5 is within noise;
  - MAI seed0 = +0.8434, seed1 = -2.5063 are already recorded; seed2
    decides the three-seed conclusion.

Usage:
  python3 me_rssm/sanity/mai_threeseed_verdict.py            # default dirs
  python3 me_rssm/sanity/mai_threeseed_verdict.py --root /data/lurui/work_dirs
"""

import argparse
import glob
import json
import os
import statistics
import sys

METRIC = 'pts_bbox/KITTI/Overall_3D_moderate'
WINDOW = (12, 16)
SEEDS = (0, 1, 2)

MAI_DIR = 'fgfull_N4_no2d_igdr_mai_2x4_24e_seed{k}'
BASE_DIR = 'fgfull_N4_no2d_igdr_2x4_24e_seed{k}'

# Recorded earlier (docs/training_runs_full.md §64/65) — the script
# re-derives them from the logs and cross-checks.
RECORDED = {0: 0.8434, 1: -2.5063}


def window_mean(run_dir):
    """ep12-16 window mean of the headline metric; None if incomplete."""
    by_epoch = {}
    for path in sorted(glob.glob(os.path.join(run_dir, '*.log.json'))):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('mode') != 'val':
                    continue
                ep = rec.get('epoch')
                value = rec.get(METRIC)
                if ep is None or value is None:
                    continue
                by_epoch[int(ep)] = float(value)
    epochs = [e for e in range(WINDOW[0], WINDOW[1] + 1) if e in by_epoch]
    if len(epochs) != WINDOW[1] - WINDOW[0] + 1:
        return None, by_epoch
    return statistics.mean(by_epoch[e] for e in epochs), by_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data/lurui/work_dirs')
    args = ap.parse_args()

    rows = []
    for seed in SEEDS:
        mai_dir = os.path.join(args.root, MAI_DIR.format(k=seed))
        base_dir = os.path.join(args.root, BASE_DIR.format(k=seed))
        for d in (mai_dir, base_dir):
            if not os.path.isdir(d):
                print(f'MISSING run dir: {d}')
                sys.exit(0)
        mai_mean, mai_curve = window_mean(mai_dir)
        base_mean, base_curve = window_mean(base_dir)
        rows.append((seed, mai_mean, base_mean, mai_curve, base_curve))

    print(f'window: ep{WINDOW[0]}-ep{WINDOW[1]}  '
          f'metric: {METRIC}  (paired same-seed comparison)')
    print(f'{"seed":>4}  {"MAI mean":>10}  {"base mean":>10}  '
          f'{"delta":>9}  note')
    deltas = []
    for seed, mai_mean, base_mean, mai_curve, base_curve in rows:
        if mai_mean is None or base_mean is None:
            miss = []
            if mai_mean is None:
                miss.append('MAI')
            if base_mean is None:
                miss.append('base')
            have = sorted(set(mai_curve) & set(base_curve))
            print(f'{seed:>4}  {"--":>10}  {"--":>10}  {"--":>9}  '
                  f'INCOMPLETE window (missing {"/".join(miss)} epochs; '
                  f'common epochs so far: {have})')
            continue
        delta = mai_mean - base_mean
        deltas.append((seed, delta))
        note = ''
        if seed in RECORDED and abs(delta - RECORDED[seed]) > 0.01:
            note = f' (!) differs from recorded {RECORDED[seed]:+.4f}'
        print(f'{seed:>4}  {mai_mean:>10.4f}  {base_mean:>10.4f}  '
              f'{delta:>+9.4f}{note}')

    if len(deltas) == len(SEEDS):
        vals = [d for _, d in deltas]
        print('-' * 60)
        print(f'paired deltas: ' +
              '  '.join(f'seed{s}={d:+.4f}' for s, d in deltas))
        print(f'mean delta {statistics.mean(vals):+.4f}  '
              f'std {statistics.pstdev(vals):.4f}')
        print('reference noise frame: paired-delta bound 0.42-0.53; '
              '|delta| < 0.5 is within noise.')
        n_pos = sum(d >= 0.5 for d in vals)
        n_neg = sum(d <= -0.5 for d in vals)
        if n_pos == len(vals):
            print('verdict frame: positive on all seeds -> '
                  'MAI supported as a component.')
        elif n_neg == len(vals):
            print('verdict frame: negative on all seeds -> '
                  'MAI alone not supported.')
        else:
            print('verdict frame: mixed across seeds -> '
                  'MAI effect not seed-robust; keep it as a '
                  'mechanism-level component, not an AP claim.')
    sys.exit(0)


if __name__ == '__main__':
    main()
