#!/usr/bin/env python3
"""MCI three-seed paired verdict (ep12-16 window, same-seed baselines).

Same convention as mai_threeseed_verdict.py but for the MCI run family.
Read-only, CPU-only. Usage:
  python3 me_rssm/sanity/mci_threeseed_verdict.py [--root /data/lurui/work_dirs]
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
CAND_DIR = 'fgfull_N4_no2d_igdr_mci_2x4_24e_seed{k}'
BASE_DIR = 'fgfull_N4_no2d_igdr_2x4_24e_seed{k}'
RECORDED = {0: 0.8252, 1: -0.7967}


def window_mean(run_dir):
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
        cand = os.path.join(args.root, CAND_DIR.format(k=seed))
        base = os.path.join(args.root, BASE_DIR.format(k=seed))
        for d in (cand, base):
            if not os.path.isdir(d):
                print(f'MISSING run dir: {d}')
                sys.exit(0)
        cand_mean, cand_curve = window_mean(cand)
        base_mean, _ = window_mean(base)
        rows.append((seed, cand_mean, base_mean, cand_curve))

    print(f'window: ep{WINDOW[0]}-ep{WINDOW[1]}  metric: {METRIC}  '
          f'(paired same-seed comparison, MCI vs no2d_igdr)')
    print(f'{"seed":>4}  {"MCI mean":>10}  {"base mean":>10}  {"delta":>9}  note')
    deltas = []
    for seed, cand_mean, base_mean, cand_curve in rows:
        if cand_mean is None or base_mean is None:
            have = sorted(cand_curve)
            print(f'{seed:>4}  {"--":>10}  {"--":>10}  {"--":>9}  '
                  f'INCOMPLETE (MCI epochs so far: {have})')
            continue
        delta = cand_mean - base_mean
        deltas.append((seed, delta))
        note = ''
        if seed in RECORDED and abs(delta - RECORDED[seed]) > 0.01:
            note = f' (!) differs from recorded {RECORDED[seed]:+.4f}'
        print(f'{seed:>4}  {cand_mean:>10.4f}  {base_mean:>10.4f}  '
              f'{delta:>+9.4f}{note}')

    if len(deltas) == len(SEEDS):
        vals = [d for _, d in deltas]
        print('-' * 60)
        print('paired deltas: ' +
              '  '.join(f'seed{s}={d:+.4f}' for s, d in deltas))
        print(f'mean delta {statistics.mean(vals):+.4f}  '
              f'std {statistics.pstdev(vals):.4f}')
        print('MAI reference: +0.8434 / -2.5063 / -0.0985 '
              '(mean -0.5872, std 1.4105)')
        n_pos = sum(d >= 0 for d in vals)
        n_neg = sum(d < 0 for d in vals)
        if all(d >= 0 for d in vals):
            print('verdict frame: positive on all seeds -> MCI mainline '
                  'supported.')
        elif all(d < 0 for d in vals):
            print('verdict frame: negative on all seeds -> MCI mainline '
                  'not supported.')
        else:
            print(f'verdict frame: mixed ({n_pos} pos / {n_neg} neg) -> '
                  'report per-seed paired deltas; compare against the MAI '
                  'roll (mean -0.59, std 1.41) for the mechanism-vs-AP '
                  'discussion.')
    sys.exit(0)


if __name__ == '__main__':
    main()
