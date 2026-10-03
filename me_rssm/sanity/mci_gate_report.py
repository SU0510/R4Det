#!/usr/bin/env python3
"""Gate dynamic-range report for MCI-RSSM training logs (read-only).

Parses a work_dir's ``*.log.json`` train records and judges the
pre-registered 450-iter mechanism gate for the reliability gates
(me_rssm/docs/12 §gates):

  G1 non-degenerate spatial structure: mean over records of
     stat_gain_std > GATE_SPATIAL_STD (per-pixel spread inside a map)
  G2 non-degenerate temporal variation:
     max(stat_gain_mean) - min(stat_gain_mean) > GATE_TEMPORAL_SPREAD
  G3 not collapsed: the time-averaged mean stays inside (0.5, 1.5),
     i.e. the gate neither shut the correction off nor saturated.

Same three checks for the readout gate (stat_readout_gain_*). This tool
only prints the numbers and a PASS/FAIL per check; the verdict (continue
to ep6 or stop) stays a human/agent step.

Usage:
  python3 me_rssm/sanity/mci_gate_report.py /data/lurui/work_dirs/<run>
"""

import argparse
import glob
import json
import statistics
import sys

GATE_SPATIAL_STD = 0.005
GATE_TEMPORAL_SPREAD = 0.005
MEAN_RANGE = (0.5, 1.5)

FAMILIES = {
    'correct gate': ('stat_gain_mean', 'stat_gain_std'),
    'readout gate': ('stat_readout_gain_mean', 'stat_readout_gain_std'),
}


def load_train_records(work_dir):
    records = []
    for path in sorted(glob.glob(f'{work_dir}/*.log.json')):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('mode') == 'train':
                    records.append(rec)
    return records


def judge(name, means, stds):
    if not means:
        print(f'[{name}] no records found -> NOT EVALUABLE')
        return
    mean_of_means = statistics.mean(means)
    spread = max(means) - min(means)
    mean_std = statistics.mean(stds) if stds else float('nan')
    checks = [
        ('G1 spatial std', mean_std > GATE_SPATIAL_STD),
        ('G2 temporal spread', spread > GATE_TEMPORAL_SPREAD),
        ('G3 mean in range', MEAN_RANGE[0] < mean_of_means
         < MEAN_RANGE[1]),
    ]
    verdict = 'PASS' if all(ok for _, ok in checks) else 'FAIL'
    print(f'[{name}] {verdict}:')
    print(f'    records={len(means)}  mean(gain_mean)={mean_of_means:.4f}  '
          f'min/max=({min(means):.4f}, {max(means):.4f})  '
          f'mean(gain_std)={mean_std:.4f}')
    for label, ok in checks:
        print(f'    {label}: {"PASS" if ok else "FAIL"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('work_dir')
    args = ap.parse_args()

    records = load_train_records(args.work_dir)
    if not records:
        print(f'no train records under {args.work_dir}')
        sys.exit(0)
    print(f'work_dir: {args.work_dir}  train records: {len(records)}')
    for name, (mean_key, std_key) in FAMILIES.items():
        means = [r[mean_key] for r in records if mean_key in r]
        stds = [r[std_key] for r in records if std_key in r]
        judge(name, means, stds)
    sys.exit(0)


if __name__ == '__main__':
    main()
