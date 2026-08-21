#!/usr/bin/env python3
"""Summarize a training run's validation metrics with reproducibility stats.

Reads all ``*.log.json`` files under a work_dir, keeps only the last
``mode=val`` record per epoch (matching the docs convention to avoid
double-counting mid-run restarts), and reports the headline metrics as:

* BEST / LAST
* platform mean +- std over the last ``--tail`` epochs

This encodes the "compare platform mean, not single-point BEST" convention.
"""

import argparse
import glob
import json
import os
import statistics


METRIC = "pts_bbox/KITTI/Overall_3D_moderate"

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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("work_dir", help="directory containing *.log.json")
    ap.add_argument("--tail", type=int, default=5,
                    help="last N epochs used for platform mean/std (default 5)")
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
    best_ep = epochs[vals.index(max(vals))]
    last_ep = epochs[-1]

    tail_epochs = epochs[-args.tail:]
    if args.weighted:
        tail_vals = overall(records, weighted=True)[-args.tail:]
    else:
        tail_vals = [records[e][args.metric] for e in tail_epochs]
    mean = statistics.mean(tail_vals)
    std = statistics.stdev(tail_vals) if len(tail_vals) > 1 else 0.0

    print(f"run: {args.work_dir}")
    mode = "frequency-weighted" if args.weighted else "equal-weight (log)"
    print(f"metric: {args.metric}")
    print(f"weighting: {mode}")
    print(f"epochs: {epochs[0]}..{epochs[-1]} ({len(epochs)} val records)")
    print(f"BEST  = {max(vals):.2f} @ ep{best_ep}")
    last_val = vals[-1] if args.weighted else records[last_ep][args.metric]
    print(f"LAST  = {last_val:.2f} @ ep{last_ep}")
    print(f"platform (last {args.tail}) = {mean:.2f} +- {std:.2f}")
    print(f"platform range = {min(tail_vals):.2f} .. {max(tail_vals):.2f}")
    print()

    print("constituents (BEST | LAST | platform mean+-std):")
    for name, key in CONSTITUENTS.items():
        cvals = series(records, key)
        if not cvals:
            continue
        ctail = [records[e][key] for e in tail_epochs]
        cmean = statistics.mean(ctail)
        cstd = statistics.stdev(ctail) if len(ctail) > 1 else 0.0
        print(f"  {name:12s} best={max(cvals):5.2f} "
              f"last={records[last_ep][key]:5.2f} "
              f"plat={cmean:5.2f}+-{cstd:4.2f}")
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
