#!/usr/bin/env python3
"""Summarize a training run's validation metrics with reproducibility stats.

Reads all ``*.log.json`` files under a work_dir, keeps only the last
``mode=val`` record per epoch (matching the docs convention to avoid
double-counting mid-run restarts), and reports the two comparison
quantities fixed by the training-record convention:

* fixed-window mean over ``--window`` (default ep12-16)
* all-epoch peak over every val epoch, whether or not its checkpoint exists

It also reports ``best saved`` (highest val epoch with an actual
``epoch_<N>.pth`` on disk) and ``last`` for checkpoint bookkeeping.
"""

import argparse
import glob
import json
import os
import statistics


METRIC = "pts_bbox/KITTI/Overall_3D_moderate"

# Default comparison window. See docs/training_runs_full.md "统一记录口径".
DEFAULT_WINDOW_START = 12
DEFAULT_WINDOW_END = 16

# Constituent metrics of Overall_3D_moderate (see kitti_utils/eval.py):
#   (Car_strict + Truck_strict + Ped_loose + Cyc_loose) / 4
CONSTITUENTS = {
    "Car_strict": "pts_bbox/KITTI/Car_3D_moderate_strict",
    "Truck_strict": "pts_bbox/KITTI/Truck_3D_moderate_strict",
    "Ped_loose": "pts_bbox/KITTI/Pedestrian_3D_moderate_loose",
    "Cyc_loose": "pts_bbox/KITTI/Cyclist_3D_moderate_loose",
}

# Training-set class frequency priors (excluding ~0.5% "Other"), renormalized.
# Use with --weighted to compute a frequency-weighted Overall instead of the
# default 1/4 equal-weight used by upstream kitti_utils/eval.py.
CLASS_PRIORS = {
    "Car": 0.4837,
    "Truck": 0.1646,
    "Ped": 0.1353,
    "Cyc": 0.2164,
}


def load_eval_series(paths):
    """Return {epoch: val_record} using the last val record per epoch."""
    by_epoch = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("mode") == "val":
                    by_epoch[rec["epoch"]] = rec
    return by_epoch


def series(records, key):
    return [records[e][key] for e in sorted(records) if key in records[e]]


def fmt(records, key):
    vals = series(records, key)
    if not vals:
        return "n/a"
    return ", ".join(f"{v:.2f}" for v in vals)


def overall(records, weighted=False):
    """Return list of per-epoch Overall values (equal or frequency weighted)."""
    vals = []
    for e in sorted(records):
        cs = records[e]["pts_bbox/KITTI/Car_3D_moderate_strict"]
        ts = records[e]["pts_bbox/KITTI/Truck_3D_moderate_strict"]
        pl = records[e]["pts_bbox/KITTI/Pedestrian_3D_moderate_loose"]
        cl = records[e]["pts_bbox/KITTI/Cyclist_3D_moderate_loose"]
        if weighted:
            v = (CLASS_PRIORS["Car"] * cs + CLASS_PRIORS["Truck"] * ts
                 + CLASS_PRIORS["Ped"] * pl + CLASS_PRIORS["Cyc"] * cl)
        else:
            v = (cs + ts + pl + cl) / 4
        vals.append(v)
    return vals


def parse_window(text):
    """Parse 'START-END' (inclusive) into a two-tuple of ints."""
    try:
        start, end = text.split("-", 1)
        start, end = int(start), int(end)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid window {text!r}; expected START-END") from exc
    if start > end:
        raise argparse.ArgumentTypeError(f"invalid window {text!r}; START must be <= END")
    return start, end


def saved_epochs(work_dir):
    """Return epochs that have an actual epoch_<N>.pth file on disk."""
    saved = []
    for path in glob.glob(os.path.join(work_dir, "epoch_*.pth")):
        stem = os.path.basename(path)[len("epoch_"):-len(".pth")]
        if stem.isdigit():
            saved.append(int(stem))
    return sorted(saved)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("work_dir", help="directory containing *.log.json")
    ap.add_argument("--tail", type=int, default=5,
                    help="last N epochs used for the supplementary platform mean/std (default 5)")
    ap.add_argument("--window", type=parse_window, default=(DEFAULT_WINDOW_START, DEFAULT_WINDOW_END),
                    metavar="START-END",
                    help="fixed comparison window, inclusive (default 12-16)")
    ap.add_argument("--metric", default=METRIC,
                    help="metric key to summarize (default Overall_3D_moderate)")
    ap.add_argument("--weighted", action="store_true",
                    help="report frequency-weighted Overall (class-prior weighted)")
    ap.add_argument("--key", action="append", dest="extra_keys", default=[],
                    help="additional metric key(s) to show per epoch (repeatable)")
    args = ap.parse_args()

    logfiles = sorted(glob.glob(os.path.join(args.work_dir, "*.log.json")))
    if not logfiles:
        print(f"No *.log.json found under {args.work_dir}")
        return 1

    records = load_eval_series(logfiles)
    epochs = sorted(records)
    if not epochs:
        print(f"No mode=val records in {len(logfiles)} log file(s)")
        return 1

    if args.weighted:
        vals = overall(records, weighted=True)
    else:
        vals = [records[e][args.metric] for e in epochs]
    peak_ep = epochs[vals.index(max(vals))]
    last_ep = epochs[-1]

    win_start, win_end = args.window
    window_epochs = [e for e in epochs if win_start <= e <= win_end]
    if args.weighted:
        by_epoch = dict(zip(epochs, vals))
        window_vals = [by_epoch[e] for e in window_epochs]
    else:
        window_vals = [records[e][args.metric] for e in window_epochs]

    tail_epochs = epochs[-args.tail:]
    if args.weighted:
        tail_vals = overall(records, weighted=True)[-args.tail:]
    else:
        tail_vals = [records[e][args.metric] for e in tail_epochs]
    mean = statistics.mean(tail_vals)
    std = statistics.stdev(tail_vals) if len(tail_vals) > 1 else 0.0

    saved = saved_epochs(args.work_dir)
    saved_with_val = [e for e in saved if e in records]
    if saved_with_val:
        if args.weighted:
            saved_vals = [dict(zip(epochs, vals))[e] for e in saved_with_val]
        else:
            saved_vals = [records[e][args.metric] for e in saved_with_val]
        saved_ep = saved_with_val[saved_vals.index(max(saved_vals))]
        saved_value = max(saved_vals)
    else:
        saved_ep = None
        saved_value = None

    print(f"run: {args.work_dir}")
    mode = "frequency-weighted" if args.weighted else "equal-weight (log)"
    print(f"metric: {args.metric}")
    print(f"weighting: {mode}")
    print(f"epochs: {epochs[0]}..{epochs[-1]} ({len(epochs)} val records)")
    if window_vals:
        window_mean = statistics.mean(window_vals)
        window_std = statistics.stdev(window_vals) if len(window_vals) > 1 else 0.0
        print(f"window ep{win_start}-{win_end} = {window_mean:.2f} +- {window_std:.2f} "
              f"({len(window_vals)} epochs)")
    else:
        print(f"window ep{win_start}-{win_end} = n/a (no matching val epochs)")
    print(f"peak (all epochs) = {max(vals):.2f} @ ep{peak_ep}"
          f"{'' if peak_ep in saved else '  [checkpoint not saved]'}")
    if saved_ep is None:
        print("best saved = n/a (no epoch_<N>.pth with matching val)")
    else:
        print(f"best saved = {saved_value:.2f} @ ep{saved_ep}")
    last_val = vals[-1] if args.weighted else records[last_ep][args.metric]
    print(f"last = {last_val:.2f} @ ep{last_ep}")
    print(f"platform (last {args.tail}, supplementary) = {mean:.2f} +- {std:.2f}")
    print(f"platform range = {min(tail_vals):.2f} .. {max(tail_vals):.2f}")
    print()

    print(f"constituents (window ep{win_start}-{win_end} | peak | last):")
    for name, key in CONSTITUENTS.items():
        cvals = series(records, key)
        if not cvals:
            continue
        cwindow = [records[e][key] for e in window_epochs]
        cwin = f"{statistics.mean(cwindow):5.2f}" if cwindow else "  n/a"
        print(f"  {name:12s} win={cwin} "
              f"peak={max(cvals):5.2f} "
              f"last={records[last_ep][key]:5.2f}")
    print()

    if args.extra_keys:
        print("per-epoch (extra keys):")
        header = "ep  " + "  ".join(k.split("/")[-1] for k in args.extra_keys)
        print(header)
        for e in epochs:
            row = [f"{e:2d}"] + [f"{records[e][k]:.2f}" for k in args.extra_keys]
            print("  ".join(row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
