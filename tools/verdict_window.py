#!/usr/bin/env python3
"""Paired multi-seed window verdicts + training-signal stat extraction.

Two independent modes (new tool; touches no existing file):

1. Window verdict (default): for pairs of run dirs sharing seeds
   (``--base DIR`` and ``--cand DIR``, each containing ``seed_<k>``
   subdirs), compute the ep12-16-style window means of the headline
   metrics per seed, the paired deltas (cand - base), and the across-seed
   summary. Encodes the pre-registered gating convention: judge window
   means per seed, never single-point BESTs.

2. Stat curves (``--stats``): scan one run dir's ``*.log.json`` train
   records (last record per epoch) and print the RSSM/ME-RSSM diagnostic
   channels (stat_gain_mean, stat_conf_cam_mean, stat_conf_mot_mean,
   stat_dyn_ratio, stat_speed_mean, stat_clamped_ratio, ...) per epoch --
   the monitoring input for the 06_second_layer P2/C6 decisions.

Examples:
  python3 tools/verdict_window.py \
      --base work_dirs/run10_headv2_multiseed \
      --cand work_dirs/cyccls_branch_N4_2x4_24e_multiseed \
      --window 12 16
  python3 tools/verdict_window.py --stats /data/lurui/work_dirs/me_rssm_N4_2x4_24e_seed0
"""

import argparse
import glob
import json
import os
import statistics

METRIC = "pts_bbox/KITTI/Overall_3D_moderate"
BEV = "pts_bbox/KITTI/Overall_BEV_moderate"
# (Car_strict + Truck_strict + Ped_loose + Cyc_loose) / 4
CONSTITUENTS = {
    "Car_strict": "pts_bbox/KITTI/Car_3D_moderate_strict",
    "Truck_strict": "pts_bbox/KITTI/Truck_3D_moderate_strict",
    "Ped_loose": "pts_bbox/KITTI/Pedestrian_3D_moderate_loose",
    "Cyc_loose": "pts_bbox/KITTI/Cyclist_3D_moderate_loose",
}
STAT_KEYS = ("stat_gain_mean", "stat_conf_cam_mean", "stat_conf_mot_mean",
             "stat_dyn_ratio", "stat_speed_mean", "stat_kl_raw_mean",
             "stat_kl_effective_mean", "stat_clamped_ratio",
             "stat_mu_diff_sq", "stat_posterior_std", "stat_prior_std")


def load_records(path_glob, mode):
    """{epoch: last record of `mode`} across all matching log files."""
    by_epoch = {}
    for p in sorted(glob.glob(path_glob)):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("mode") == mode:
                    by_epoch[rec["epoch"]] = rec
    return by_epoch


def window_mean(records, key, lo, hi):
    """Mean over the window; returns (mean, n_epochs_covered)."""
    vals = [records[e][key] for e in sorted(records)
            if lo <= e <= hi and key in records[e]]
    return (statistics.mean(vals) if vals else None), len(vals)


def seed_dirs(base):
    return {d.split("seed_")[-1]: d for d in sorted(glob.glob(
        os.path.join(base, "seed_*"))) if os.path.isdir(d)}


def run_window_mode(args):
    base_seeds = seed_dirs(args.base)
    cand_seeds = seed_dirs(args.cand)
    seeds = sorted(set(base_seeds) & set(cand_seeds))
    if not seeds:
        raise SystemExit(f"no shared seed_* subdirs under {args.base} / "
                         f"{args.cand}")
    keys = [("Overall3D", METRIC), ("OverallBEV", BEV)]
    keys += list(CONSTITUENTS.items())

    print(f"window ep{args.window[0]}-{args.window[1]}, "
          f"paired per-seed deltas (cand - base). "
          f"Full window = {args.window[1] - args.window[0] + 1} epochs; "
          f"PARTIAL windows are noise-prone and not a verdict.")
    header = ["seed"] + [n for n, _ in keys]
    print(" ".join(f"{h:>12}" for h in header))
    per_metric_delta = {n: [] for n, _ in keys}
    for seed in seeds:
        rec_b = load_records(os.path.join(base_seeds[seed], "*.log.json"),
                             "val")
        rec_c = load_records(os.path.join(cand_seeds[seed], "*.log.json"),
                             "val")
        epochs_b = {e for e in rec_b if args.window[0] <= e <= args.window[1]}
        epochs_c = {e for e in rec_c if args.window[0] <= e <= args.window[1]}
        coverage = len(epochs_b & epochs_c)
        tag = f"seed{seed}" + ("" if coverage ==
                               args.window[1] - args.window[0] + 1
                               else f" PARTIAL({coverage})")
        row = [tag]
        for name, key in keys:
            mb, nb = window_mean(rec_b, key, *args.window)
            mc, nc = window_mean(rec_c, key, *args.window)
            if mb is None or mc is None or not (nb == nc ==
                                                args.window[1] - args.window[0] + 1):
                row.append(f"{'--':>12}")
                continue
            d = mc - mb
            per_metric_delta[name].append(d)
            row.append(f"{d:+.4f}")
        print(" ".join(f"{c:>12}" for c in row))

    print("\nacross-seed paired delta over COMPLETE windows only "
          "(mean +- std; direction-consistent if signs match):")
    for name, _ in keys:
        ds = per_metric_delta[name]
        if not ds:
            continue
        signs = "consistent" if all(d > 0 for d in ds) or \
            all(d < 0 for d in ds) else "MIXED"
        print(f"  {name:>12}: {statistics.mean(ds):+.4f} +- "
              f"{statistics.stdev(ds) if len(ds) > 1 else 0.0:.4f} "
              f"({signs}, n={len(ds)})")


def run_stats_mode(args):
    rec = load_records(os.path.join(args.stats, "*.log.json"), "train")
    if not rec:
        raise SystemExit(f"no train records under {args.stats}")
    found = [k for k in STAT_KEYS
             if any(k in r for r in rec.values())]
    print(f"epochs={min(rec)}..{max(rec)}  stat channels found: {found}")
    hdr = ["ep"] + [k.replace("stat_", "").replace("_mean", "")
                    for k in found]
    print(" ".join(f"{h:>14}" for h in hdr))
    for e in sorted(rec):
        row = [str(e)]
        for k in found:
            v = rec[e].get(k)
            row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "--")
        print(" ".join(f"{c:>14}" for c in row))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", help="baseline multiseed dir (seed_* subdirs)")
    ap.add_argument("--cand", help="candidate multiseed dir")
    ap.add_argument("--window", type=int, nargs=2, default=(12, 16),
                    metavar=("LO", "HI"))
    ap.add_argument("--stats", help="run dir: dump stat_* train curves")
    args = ap.parse_args()
    if args.stats:
        run_stats_mode(args)
    elif args.base and args.cand:
        run_window_mode(args)
    else:
        ap.error("provide --base and --cand, or --stats")


if __name__ == "__main__":
    main()
