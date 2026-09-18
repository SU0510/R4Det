#!/usr/bin/env python
"""Wait for a TJ4D run's ep12-16 validation window, score the Cyclist gate, act.

Used to make the Cyclist-branch seed0 comparison self-driving: it polls the
run's ``*.log.json`` until the ep12-16 validation window is present, computes
the 5-epoch means, compares them against the clean full-RSSM baseline with the
pre-registered gate, writes a human-readable verdict next to the run, and
(optionally) stops the training tmux session on failure so a failed run does
not keep burning GPUs.

The log ``epoch`` field is 1-indexed and matches the epoch labels used in
``docs/training_runs_full.md`` (verified against the shared-stem run, where
log epoch 14 -> Cyclist loose 45.3136 = documented ep14).
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

METRIC_KEYS = {
    'Overall3D': 'pts_bbox/KITTI/Overall_3D_moderate',
    'BEV3D': 'pts_bbox/KITTI/Overall_BEV_moderate',
    'Cyc_l': 'pts_bbox/KITTI/Cyclist_3D_moderate_loose',
    'Ped_l': 'pts_bbox/KITTI/Pedestrian_3D_moderate_loose',
    'Car_s': 'pts_bbox/KITTI/Car_3D_moderate_strict',
    'Trk_s': 'pts_bbox/KITTI/Truck_3D_moderate_strict',
}

# clean full-RSSM seed0 ep12-16 means (docs/training_runs_full.md sec 44).
# The earlier 46.9612 BEV value was a copy from the KL=0 ablation run; the
# clean full-RSSM seed0 value verified from the raw log is 46.5193.
CLEAN_BASELINE = {
    'Overall3D': 38.3902,
    'BEV3D': 46.5193,
    'Cyc_l': 48.6271,
    'Ped_l': 28.9160,
    'Car_s': 47.8027,
    'Trk_s': 28.2149,
}

# Pre-registered gate floors. These are intentionally frozen at the values
# registered before the run; do not retune them after correcting the clean
# baseline above. Cyclist floor encodes the "no class may drop by more than
# 1.0" rule as an absolute number (48.6271 - 1.0 -> 47.63); Overall and BEV
# floors were the originally registered 38.89 / 47.46.
CLEAN_FLOORS = {
    'Overall3D': 38.89,
    'BEV3D': 47.46,
    'Cyc_l': 47.63,
}
MAX_DROP = 1.0
MIN_UP = 0.5
NEED_UP = 2
UP_CLASSES = ('Cyc_l', 'Ped_l', 'Car_s', 'Trk_s')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--log-json', required=True, help='run *.log.json path')
    p.add_argument('--session', default='', help='tmux session to stop on fail')
    p.add_argument('--start-epoch', type=int, default=12)
    p.add_argument('--end-epoch', type=int, default=16)
    p.add_argument('--poll-sec', type=int, default=120)
    p.add_argument('--kill-on-fail', action='store_true')
    p.add_argument('--run-dir', default='',
                   help='dir for the verdict file (defaults to log dir)')
    return p.parse_args()


def read_val_records(log_json):
    """Return {epoch_1indexed: record} for mode=val lines."""
    recs = {}
    path = Path(log_json)
    if not path.exists():
        return recs
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get('mode') != 'val':
            continue
        ep = d.get('epoch')
        if ep is None:
            continue
        recs[ep] = d
    return recs


def window_complete(recs, start, end):
    for ep in range(start, end + 1):
        rec = recs.get(ep)
        if rec is None or any(k not in rec for k in METRIC_KEYS.values()):
            return False
    return True


def checkpoint_ready(run_dir, end_epoch):
    """Guard so a kill-on-fail never deletes the endpoint checkpoint.

    In this runner the epoch-N checkpoint is written before the epoch-N
    validation record (verified on the shared-stem and Cyclist-branch runs), so
    requiring the file is a cheap extra safety net before stopping the session.
    """
    return (Path(run_dir) / f'epoch_{end_epoch}.pth').exists()


def window_means(recs, start, end):
    out = {}
    n = end - start + 1
    for name, key in METRIC_KEYS.items():
        out[name] = sum(recs[ep][key] for ep in range(start, end + 1)) / n
    return out


def evaluate(means):
    floors = {name: (means[name], thr, means[name] >= thr)
              for name, thr in CLEAN_FLOORS.items()}
    drops = {name: means[name] - CLEAN_BASELINE[name]
             for name in ('Ped_l', 'Car_s', 'Trk_s')}
    drop_ok = all(v >= -MAX_DROP for v in drops.values())
    ups = [n for n in UP_CLASSES if means[n] - CLEAN_BASELINE[n] >= MIN_UP]
    up_ok = len(ups) >= NEED_UP
    passed = all(ok for _, _, ok in floors.values()) and drop_ok and up_ok
    return {'floors': floors, 'drops': drops, 'ups': ups,
            'drop_ok': drop_ok, 'up_ok': up_ok, 'passed': passed}


def render(means, res, start, end, log_json):
    lines = [
        f'Cyclist-branch gate, ep{start}-{end} mean  (log: {log_json})',
        '',
        f'{"metric":10} {"clean":>9} {"run":>9} {"delta":>9}',
        '-' * 45,
    ]
    for name in ('Overall3D', 'BEV3D', 'Cyc_l', 'Ped_l', 'Car_s', 'Trk_s'):
        lines.append(f'{name:10} {CLEAN_BASELINE[name]:9.4f} '
                     f'{means[name]:9.4f} {means[name]-CLEAN_BASELINE[name]:+9.4f}')
    lines += ['', 'checks:']
    for name, (val, thr, ok) in res['floors'].items():
        lines.append(f'  [{"PASS" if ok else "FAIL"}] {name} >= {thr:.2f} '
                     f'(got {val:.4f})')
    for name, delta in res['drops'].items():
        ok = delta >= -MAX_DROP
        lines.append(f'  [{"PASS" if ok else "FAIL"}] {name} drop <= {MAX_DROP} '
                     f'(delta {delta:+.4f})')
    lines.append(f'  [{"PASS" if res["up_ok"] else "FAIL"}] '
                 f'>={NEED_UP} classes up >= {MIN_UP}: {res["ups"]}')
    lines += ['', f'VERDICT: {"PASS" if res["passed"] else "FAIL"}',
              ('  PASS -> continue to ep24, then run seed1/2.'
               if res['passed'] else
               '  FAIL -> stop at ep16; clean RSSM stays the main model.')]
    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.log_json).parent
    verdict_path = run_dir / f'gate_ep{args.start_epoch}_{args.end_epoch}.txt'
    print(f'{time.strftime("%F %T")} watcher start; waiting for '
          f'ep{args.start_epoch}-{args.end_epoch} validation window')
    while True:
        recs = read_val_records(args.log_json)
        if window_complete(recs, args.start_epoch, args.end_epoch) \
                and checkpoint_ready(Path(args.log_json).parent,
                                     args.end_epoch):
            break
        time.sleep(args.poll_sec)

    means = window_means(recs, args.start_epoch, args.end_epoch)
    res = evaluate(means)
    text = render(means, res, args.start_epoch, args.end_epoch, args.log_json)
    verdict_path.write_text(f'{time.strftime("%F %T")}\n{text}')
    print(text)

    if not res['passed'] and args.kill_on_fail and args.session:
        print(f'{time.strftime("%F %T")} gate FAIL -> stopping tmux '
              f'session {args.session}')
        subprocess.run(['tmux', 'kill-session', '-t', args.session],
                       check=False)


if __name__ == '__main__':
    main()
