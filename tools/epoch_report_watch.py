#!/usr/bin/env python
"""Report a running TJ4D training run to a Codex thread once per finished epoch.

Watches a run's ``*.log.json`` for new ``mode=val`` records. Each new epoch is
summarised into a short message (Overall 3D moderate first, since that is the
documented primary metric) and delivered into an existing Codex conversation
with ``codex queue --thread <id>``.

The state file records the highest epoch already reported, so a restart does not
re-announce epochs that were sent before.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

MOD = 'pts_bbox/KITTI/Overall_3D_moderate'
EASY = 'pts_bbox/KITTI/Overall_3D_easy'
HARD = 'pts_bbox/KITTI/Overall_3D_hard'
BEV = 'pts_bbox/KITTI/Overall_BEV_moderate'
PARTS = {
    'Car_s': 'pts_bbox/KITTI/Car_3D_moderate_strict',
    'Trk_s': 'pts_bbox/KITTI/Truck_3D_moderate_strict',
    'Ped_l': 'pts_bbox/KITTI/Pedestrian_3D_moderate_loose',
    'Cyc_l': 'pts_bbox/KITTI/Cyclist_3D_moderate_loose',
}
WINDOW = (12, 16)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--log-json', required=True)
    p.add_argument('--thread', required=True, help='Codex thread id to report into')
    p.add_argument('--run-name', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--peer-log-json', action='append', default=[],
                   help='seed<N>=<path> for same-epoch cross-seed context')
    p.add_argument('--state', required=True)
    p.add_argument('--interval', type=float, default=60.0)
    p.add_argument('--stop-epoch', type=int, default=0,
                   help='stop after reporting this epoch (0 = run until log stalls)')
    p.add_argument('--stall-hours', type=float, default=3.0)
    p.add_argument('--codex-bin', default='codex')
    return p.parse_args()


def read_val_epochs(path):
    """Return {epoch: record} keeping the last val record per epoch."""
    out = {}
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get('mode') == 'val' and 'epoch' in rec:
                    out[int(rec['epoch'])] = rec
    except FileNotFoundError:
        pass
    return out


def read_peers(specs):
    peers = {}
    for spec in specs:
        try:
            label, path = spec.split('=', 1)
        except ValueError:
            continue
        peers[label] = read_val_epochs(path)
    return peers


def load_state(path):
    try:
        with open(path) as fh:
            return int(json.load(fh).get('last_reported', 0))
    except (OSError, ValueError):
        return 0


def save_state(path, epoch):
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as fh:
        json.dump({'last_reported': epoch}, fh)
    os.replace(tmp, path)


def window_mean(epochs, key, lo, hi):
    vals = [epochs[e][key] for e in range(lo, hi + 1)
            if e in epochs and key in epochs[e]]
    if not vals:
        return None
    return sum(vals) / len(vals), len(vals)


def build_message(run_name, seed, epoch, rec, epochs, peers, stop_epoch, total):
    mod = rec.get(MOD)
    best_ep, best = max(
        ((e, r[MOD]) for e, r in epochs.items() if MOD in r),
        key=lambda kv: kv[1], default=(0, 0.0))
    peak_note = '' if best_ep == epoch else f'，本 run 最高 {best:.2f} @ep{best_ep}'

    lines = [f'【{run_name} · ep{epoch}/{total} 出分】',
             f'Overall 3D moderate: {mod:.4f}{peak_note}']

    extras = []
    for label, key in (('3D easy', EASY), ('3D hard', HARD), ('BEV moderate', BEV)):
        if key in rec:
            extras.append(f'{label} {rec[key]:.2f}')
    if extras:
        lines.append(' | '.join(extras))

    parts = [f'{name} {rec[key]:.2f}' for name, key in PARTS.items() if key in rec]
    if parts:
        lines.append(' | '.join(parts))

    peer_bits = []
    for label in sorted(peers):
        r = peers[label].get(epoch)
        if r and MOD in r:
            peer_bits.append(f'{label} {r[MOD]:.2f}')
        else:
            peer_bits.append(f'{label} —')
    if peer_bits:
        lines.append(f'同期同 epoch: seed{seed} {mod:.2f} | ' + ' | '.join(peer_bits))

    lo, hi = WINDOW
    wm = window_mean(epochs, MOD, lo, hi)
    if wm:
        mean, n = wm
        span = f'ep{lo}-{hi}' if n == hi - lo + 1 else f'ep{lo}-{hi} 中已出 {n} 点'
        lines.append(f'区间均值（{span}，等权）: {mean:.4f}')

    if stop_epoch and epoch >= stop_epoch:
        lines.append(f'已到预设截断点 ep{stop_epoch}，可 kill 训练进程；'
                     f'本 watcher 自动退出。')
    return '\n'.join(lines)


def deliver(codex_bin, thread, message):
    env = dict(os.environ, TERM='xterm')
    proc = subprocess.run(
        [codex_bin, 'queue', '--thread', thread, '--message', message],
        capture_output=True, text=True, env=env)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def main():
    args = parse_args()
    total = 24
    last = load_state(args.state)
    if last == 0:
        existing = read_val_epochs(args.log_json)
        last = max(existing) if existing else 0
        save_state(args.state, last)

    stalled_since = time.time()
    while True:
        epochs = read_val_epochs(args.log_json)
        newest = max(epochs) if epochs else 0
        if newest > last:
            peers = read_peers(args.peer_log_json)
            for epoch in sorted(e for e in epochs if e > last):
                msg = build_message(args.run_name, args.seed, epoch, epochs[epoch],
                                    epochs, peers, args.stop_epoch, total)
                ok, out = deliver(args.codex_bin, args.thread, msg)
                if ok:
                    last = epoch
                    save_state(args.state, last)
                    stalled_since = time.time()
                    if args.stop_epoch and epoch >= args.stop_epoch:
                        return
                else:
                    # Keep the epoch pending and retry on the next poll.
                    break
        elif newest < last:
            # Log rotated or truncated; re-sync without spamming.
            last = newest
            save_state(args.state, last)
        else:
            if time.time() - stalled_since > args.stall_hours * 3600:
                return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
