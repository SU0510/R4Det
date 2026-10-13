#!/usr/bin/env python3
"""Strict artifacts/curve helpers for the GP-MSF queue (stdlib, no GPU)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

PRIMARY = 'pts_bbox/KITTI/Overall_3D_moderate'
CLASSES = {
    'Car': 'pts_bbox/KITTI/Car_3D_moderate_strict',
    'Truck': 'pts_bbox/KITTI/Truck_3D_moderate_strict',
    'Pedestrian': 'pts_bbox/KITTI/Pedestrian_3D_moderate_loose',
    'Cyclist': 'pts_bbox/KITTI/Cyclist_3D_moderate_loose',
}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def source_hashes():
    # Freeze all research code/config dependencies, not only the entry config.
    root = Path(__file__).resolve().parents[2]
    paths = [root / 'tools/train_vod.py', root / 'tools/dist_train.sh',
             root / 'training_old_version.py',
             root / 'checkpoints/pretrained_tj4d.pth']
    for name in ('mmdet3d', 'me_rssm', 'configs'):
        paths.extend((root / name).rglob('*.py'))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths)) if p.is_file() and '__pycache__' not in p.parts}


def start_phase(directory, config):
    path = Path(directory) / 'phase.json'
    if path.exists() or list(Path(directory).glob('*.log.json')):
        raise ValueError('refusing stale phase artifacts: ' + str(directory))
    write_json(path, dict(started_at=time.time(), config=config,
                         source_sha=source_hashes(), exit_code=None,
                         gpus=[5, 6, 7], world_size=3, seed=0))


def finish_phase(directory):
    path = Path(directory) / 'phase.json'
    value = json.loads(path.read_text())
    if value['source_sha'] != source_hashes():
        raise ValueError('research source changed while phase was running')
    value.update(exit_code=0, finished_at=time.time())
    write_json(path, value)


def completed_phase(directory):
    value = json.loads((Path(directory) / 'phase.json').read_text())
    if value.get('exit_code') != 0 or value.get('finished_at', 0) < value['started_at']:
        raise ValueError('phase did not exit successfully')
    if value['source_sha'] != source_hashes():
        raise ValueError('phase source does not match current source')
    if value.get('gpus') != [5, 6, 7] or value.get('world_size') != 3:
        raise ValueError('phase GPU/rank contract mismatch')
    return value


def numeric_leaves(value, prefix=''):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from numeric_leaves(item, prefix + '.' + str(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from numeric_leaves(item, prefix + '.' + str(i))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def read_rows(directory, live=False, since=None):
    paths = sorted((p for p in Path(directory).glob('*.log.json')
                    if not p.name.startswith('merged')),
                   key=lambda p: (p.stat().st_mtime, str(p)))
    rows = []
    for path in paths:
        if since is not None and path.stat().st_mtime < since:
            raise ValueError('stale training log: ' + str(path))
        data = path.read_bytes()
        lines = data.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if live and i == len(lines) - 1 and not line.endswith(b'\n'):
                    continue  # Only an unfinished trailing write is tolerated.
                raise ValueError(f'malformed log {path}:{i + 1}')
            bad = [key for key, value in numeric_leaves(row) if not math.isfinite(value)]
            if bad:
                raise ValueError(f'non-finite log {path}:{i + 1}: {bad}')
            rows.append(row)
    return rows


def performance(rows):
    train = [r for r in rows if r.get('mode') == 'train' and r.get('iter', 0) > 50]
    times = [r['time'] for r in train if isinstance(r.get('time'), (int, float))]
    memory = [r['memory'] for r in train if isinstance(r.get('memory'), (int, float))]
    if len(times) < 5 or not memory or min(times) <= 0 or max(memory) <= 0:
        raise ValueError('missing/invalid matched smoke time or memory measurements')
    return dict(n_time_rows=len(times), mean_iter_time_s=statistics.mean(times[-100:]),
                max_memory_mib=max(memory))


def curve_report(directory, window, baseline=None, expected_epochs=None, live=False):
    rows = read_rows(directory, live=live)
    val = {}
    for row in rows:
        if row.get('mode') == 'val' and PRIMARY in row:
            val[int(row['epoch'])] = row
    lo, hi = window
    epochs = list(range(lo, hi + 1))
    missing = sorted(set(epochs) - val.keys())
    if missing:
        raise ValueError('incomplete frozen window; missing epochs ' + str(missing))
    if expected_epochs and set(range(1, expected_epochs + 1)) - val.keys():
        raise ValueError('incomplete full validation curve')
    required = [PRIMARY] + list(CLASSES.values())
    for epoch in val:
        if any(key not in val[epoch] for key in required):
            raise ValueError('missing class metrics at epoch ' + str(epoch))
    metrics = sorted({key for row in val.values() for key, value in row.items()
                      if key.startswith('pts_bbox/') and isinstance(value, (int, float))})
    means = {key: statistics.mean(val[e][key] for e in epochs)
             for key in metrics if all(key in val[e] for e in epochs)}
    peaks = {}
    saved = {}
    for key in metrics:
        candidates = [e for e in val if key in val[e]]
        epoch = max(candidates, key=lambda e: val[e][key])
        peaks[key] = dict(value=val[epoch][key], epoch=epoch)
        candidates = [e for e in candidates if (Path(directory) / f'epoch_{e}.pth').is_file()]
        if candidates:
            epoch = max(candidates, key=lambda e: val[e][key])
            saved[key] = dict(value=val[epoch][key], epoch=epoch)
    result = dict(window=dict(start=lo, end=hi, n=len(epochs), complete=True, mean=means),
                  all_epoch_peak=peaks, best_saved=saved,
                  full_epoch_curve=[val[e] for e in sorted(val)],
                  attribution='single-seed direction only; grouped/paired mechanism evidence pending')
    if baseline:
        ref = curve_report(baseline, window)
        result['vs_b0'] = dict(mean=ref['window']['mean'], delta={
            k: v - ref['window']['mean'][k] for k, v in means.items()
            if k in ref['window']['mean']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'finish', 'curve', 'ready'])
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--config')
    parser.add_argument('--window', default='12-16')
    parser.add_argument('--b0')
    parser.add_argument('--expected-epochs', type=int)
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--out')
    args = parser.parse_args()
    try:
        if args.action == 'start':
            if not args.config:
                raise ValueError('--config required')
            start_phase(args.work_dir, args.config)
        elif args.action == 'finish':
            finish_phase(args.work_dir)
        elif args.action == 'ready':
            rows = read_rows(args.work_dir, live=True)
            return 0 if any(r.get('mode') == 'val' and
                            r.get('epoch') == args.expected_epochs and PRIMARY in r
                            for r in rows) else 3
        else:
            window = tuple(map(int, args.window.split('-')))
            report = curve_report(args.work_dir, window, args.b0, args.expected_epochs, args.live)
            if args.out:
                write_json(args.out, report)
            if args.early_stop:
                if window != (1, 6) or not args.b0:
                    raise ValueError('early gate requires complete ep1-6 and B0')
                delta = report['vs_b0']['delta'][PRIMARY]
                print('ep1-6 delta vs B0 =', delta)
                return 2 if delta < -2.0 else 0
            print('complete curve/window report:', args.out)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print('CHECK FAILED:', exc, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
