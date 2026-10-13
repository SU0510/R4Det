#!/usr/bin/env python3
"""CPU-only reread of saved A2-R/B0 curves and ep6 mechanism artifacts.

Does not import the model, run evaluation, use GPUs, or change frozen gates.
A max epoch makes conclusions stable while a source training run is live.
"""
import argparse
import datetime
import hashlib
import json
import math
import statistics
from pathlib import Path

OVERALL = "pts_bbox/KITTI/Overall_3D_moderate"
TRAIN_KEYS = (
    "loss_rssm_sem", "stat_sem_hit1", "stat_sem_thr50",
    "stat_kl_raw_mean", "stat_kl_effective_mean", "stat_clamped_ratio",
    "stat_mu_diff_sq", "stat_posterior_std", "stat_prior_std",
)
PROBES = (
    "semreadout_ep6_gate.json", "semA_rank_train_ep6.json",
    "semA_rank_val_ep6.json", "semA_m4floor_ep6.json",
    "semA_zcontract_ep6_n20.json", "post_source_std_ep6_stride700.json",
    "semA_bnmode_ep6.json", "semA_bnmode_ep6_trainsplit.json",
)


def mean(values):
    return statistics.mean(values) if values else None


def window(records, epochs):
    present = [e for e in epochs if e in records]
    keys = sorted({k for e in present for k in records[e] if k.startswith("pts_bbox/")})
    return {
        "requested_epochs": list(epochs), "actual_epochs": present, "n": len(present),
        "means": {k: mean([records[e][k] for e in present if k in records[e]]) for k in keys},
    }


def read_run(root, max_epoch, read_snapshot):
    vals, train, val_sources, ignored_partial = {}, {}, {}, []
    for path in sorted(root.glob("*.log.json")):
        data = read_snapshot(path)
        lines = data.splitlines()
        for number, line in enumerate(lines, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A live logger can leave its final record incomplete; never
                # hide a malformed record in the middle of a historical log.
                if number == len(lines) and not data.endswith(b"\n"):
                    ignored_partial.append({"file": str(path), "line": number})
                    continue
                raise
            if "epoch" not in row:
                continue
            epoch = int(row["epoch"])
            if max_epoch is not None and epoch > max_epoch:
                continue
            if row.get("mode") == "val" and OVERALL in row:
                vals[epoch] = row  # chronological filenames, final val wins
                val_sources[epoch] = {"file": str(path), "line": number}
            elif row.get("mode") == "train":
                train.setdefault(epoch, []).append(row)
    if not vals:
        raise ValueError(f"No validation records in {root}")
    peak = max(vals, key=lambda e: vals[e][OVERALL])
    saved = sorted(int(p.stem.split("_")[-1]) for p in root.glob("epoch_*.pth"))
    available = [e for e in saved if e in vals]
    best = max(available, key=lambda e: vals[e][OVERALL]) if available else None
    return {
        "work_dir": str(root), "max_epoch_filter": max_epoch,
        "val_records": vals, "val_sources": val_sources,
        "all_observed_peak": {"value": vals[peak][OVERALL], "epoch": peak},
        "saved_epochs_at_snapshot": saved,
        "best_saved": None if best is None else {"value": vals[best][OVERALL], "epoch": best},
        "windows": {"fixed_ep12_16": window(vals, range(12, 17)),
                    "ep1_observed": window(vals, range(1, max(vals) + 1)),
                    "resume_ep7_observed": window(vals, range(7, max(vals) + 1))},
        "train_logger_means": {
            e: {"n_logged_rows": len(rows),
                **{k: mean([r[k] for r in rows if k in r]) for k in TRAIN_KEYS}}
            for e, rows in sorted(train.items())
        },
        "aggregation_note": "Train means equally weight existing logger rows, not individual examples or gradient norms.",
        "ignored_partial_tail_records": ignored_partial,
    }


def rank_summary(probe):
    rows = probe["per_clip"]
    recalls = [r["recall_at_top_k_mean"] for r in rows]
    distances = [r["argmax_cityblock_dist_cells"] for r in rows]
    return {
        "n_clips": len(rows), "summary": probe["summary"],
        "topk_recall_median": statistics.median(recalls),
        "zero_topk_recall_clips": sum(x == 0 for x in recalls),
        "argmax_distance_median_cells": statistics.median(distances),
        "clipmean_distance_gt10_cells": sum(x > 10 for x in distances),
        "auc_ge_0_9_and_zero_topk_clips": sum(
            r["auc_mean_over_occupied_channels"] >= .9 and r["recall_at_top_k_mean"] == 0
            for r in rows),
    }


def mechanism_summary(probes):
    gate = probes["semreadout_ep6_gate.json"]
    m3, m4 = gate["M3_readout_toggle"]["per_clip"], gate["M4_class_information"]["per_clip"]
    n_cells = sum(r["cells"] for r in m3)
    m4_cells = sum(r["cells"] for r in m4)
    contract = probes["semA_zcontract_ep6_n20.json"]
    deploy, sampled = (contract["arms"][k] for k in ("deploy_mu_q", "train_sampled_z"))
    return {
        "frozen_gates_unchanged": gate["gates"],
        "rank": {split: rank_summary(probes[f"semA_rank_{split}_ep6.json"])
                 for split in ("train", "val")},
        "m3": {
            "n_clips": len(m3), "n_gt_cells": n_cells,
            "wins": sum(r["delta_off_minus_on"] > 0 for r in m3),
            "clipmean_ce_off_minus_on": mean([r["delta_off_minus_on"] for r in m3]),
            "cellweighted_ce_off_minus_on": sum(r["cells"] * r["delta_off_minus_on"] for r in m3) / n_cells,
            "ce_on_clipmean": mean([r["ce_on"] for r in m3]),
            "ce_off_clipmean": mean([r["ce_off"] for r in m3]),
            "harmful_clip_indices": [r["index"] for r in m3 if r["delta_off_minus_on"] <= 0],
        },
        "m4_matched_aggregation": {
            "n_gt_cells": m4_cells,
            "cellweighted_own_hit": sum(r["cells"] * r["own_hit"] for r in m4) / m4_cells,
            "pooled_prior_floor": gate["M4_class_information"]["pooled_prior_floor"],
            "clipmean_own_hit": mean([r["own_hit"] for r in m4]),
            "clipmean_prior_floor": mean([r["prior_floor"] for r in m4]),
            "note": "Report matched units; do not change preregistered M4 gate or compare macro own_hit directly with pooled floor as a causal gain.",
        },
        "bn_mode": {
            split: {
                "eval_thr50": probes[name]["arms"]["eval_running_stats"]["mean_thr50"],
                "z_path_bn_delta_thr50": probes[name]["paired_delta_vs_eval"]["z_path_batch_stats"]["mean_delta_thr50"],
                "n_z_path_bn_modules": probes[name]["bn_counts"]["z_path"],
                "note": "Read raw arms: the artifact reference/gate-reproduction metadata retains parent-A values, not A2-R. This is only a local four-BN intervention.",
            }
            for split, name in (("val", "semA_bnmode_ep6.json"),
                                ("train", "semA_bnmode_ep6_trainsplit.json"))
        },
        "sampling_contract": {
            "single_seed_only": True,
            "deploy_hit1": deploy["mean_hit1"], "sampled_hit1": sampled["mean_hit1"],
            "deploy_thr50": deploy["mean_thr50"], "sampled_thr50": sampled["mean_thr50"],
            "mean_rel_abs_z_movement": mean([r["rel_delta_z"] for r in contract["intervention_check"]]),
            "mean_rel_abs_sem_map_movement": mean([r["rel_delta_sem_map"] for r in contract["intervention_check"]]),
            "note": "Both arms use eval network/val data; changing latent sampling is not a full train/eval-BN or multiple-seed test. rel_delta uses mean absolute values, not RMS.",
        },
    }


def analytical_checks():
    """Counterexamples/identities only; these are NOT model measurements."""
    negative_cells, higher_scored_negatives = 216 * 248 - 1, 3750
    radius = 2
    sigma = (2 * radius + 1) / 6
    kernel = [math.exp(-(x * x + y * y) / (2 * sigma * sigma))
              for x in range(-radius, radius + 1)
              for y in range(-radius, radius + 1)]
    logits, shared_offset = [-2, 1, 4], 3
    original = [a - b for a in logits for b in logits]
    shifted = [(a + shared_offset) - (b + shared_offset) for a in logits for b in logits]
    assert original == shifted
    return {
        "synthetic_one_positive_auc": (negative_cells - higher_scored_negatives) / negative_cells,
        "synthetic_top1_recall": 0,
        "synthetic_binary_ranking_ap_not_kitti_ap": 1 / (higher_scored_negatives + 1),
        "synthetic_note": "One positive below 3750 of 53567 negatives: high pairwise AUC, zero top1 recall. Not actual model data.",
        "radius2_gaussian_positive_support": sum(x > 0 for x in kernel),
        "radius2_gaussian_exact_one_count": sum(x == 1 for x in kernel),
        "shared_cell_class_offset_preserves_anchor_logit_differences": original == shifted,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a2r-dir", type=Path, required=True)
    parser.add_argument("--b0-dir", type=Path, required=True)
    parser.add_argument("--max-a2r-epoch", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sources, cache = [], {}

    def snapshot(path):
        key = str(path)
        if key not in cache:
            data = path.read_bytes()
            cache[key] = data
            sources.append({"path": key, "snapshot_bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest()})
        return cache[key]

    a = read_run(args.a2r_dir, args.max_a2r_epoch, snapshot)
    b = read_run(args.b0_dir, None, snapshot)
    probes = {name: json.loads(snapshot(args.a2r_dir / name)) for name in PROBES}
    paired = {}
    for name, epochs in (("ep1_observed", range(1, args.max_a2r_epoch + 1)),
                         ("resume_ep7_observed", range(7, args.max_a2r_epoch + 1))):
        aw, bw = window(a["val_records"], epochs), window(b["val_records"], epochs)
        if aw["actual_epochs"] != bw["actual_epochs"]:
            raise ValueError(f"Paired epoch coverage differs for {name}")
        paired[name] = {"a2r": aw, "b0": bw,
                        "delta": {k: aw["means"][k] - bw["means"][k]
                                  for k in aw["means"].keys() & bw["means"].keys()}}
    snapshot(Path(__file__))
    report = {
        "snapshot_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "CPU reread only; no new training/evaluation/GPUs; live A2-R evidence capped by epoch.",
        "a2r": a, "b0": b, "paired_windows": paired,
        "mechanisms": mechanism_summary(probes), "original_probe_artifacts": probes,
        "analytical_checks_not_model_measurements": analytical_checks(),
        "sources": sources,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    last_epoch = max(a["val_records"])
    mature_n = a["windows"]["fixed_ep12_16"]["n"]
    print(f"wrote {args.out}; A2-R val ep1-{last_epoch}; fixed mature n={mature_n}")


if __name__ == "__main__":
    main()
