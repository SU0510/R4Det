#!/usr/bin/env bash
# C1 staging gate (prereg section 5 + addendum A). Read-only wrt weights.
#
#   Stage A (ep6)  early check: C1's ep1-6 in-process val curve vs B0's logged
#                  ep1-6 curve (same seed/schedule). > 2.0 points below B0
#                  in the window mean => stop for review.
#   Stage B (ep12-16) mature task gate: Overall window mean vs 39.2809 and the
#                  four-class non-inferiority lines (read from the log).
#   Print both and exit; the grouped-recall mechanism gate needs B0's ep14
#   dump (already frozen) plus C1's ep14 dump (built here, GPU 5).
#
# Usage: setsid bash me_rssm/sanity/watch_c1valid_ep6.sh &
set -u
cd /home/lurui/workspace/R4Det
source .envrc

C1_WD=${C1_WD:-/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_c1valid_seed0}
B0_WD=${B0_WD:-/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed0}
C1_CFG=${C1_CFG:-me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_c1valid.py}
B0_CFG=${B0_CFG:-configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py}
LOG=${LOG:-/tmp/c1valid_24e.log}
COV="$B0_WD/c_coverage_groups_20261012.json"
MOT="$B0_WD/c_motion_axis_20261012.json"
OUT="$C1_WD/watch_c1valid.log"

mkdir -p "$C1_WD"
echo "[watch] $(date -u '+%F %T') armed" >> "$OUT"

while [ ! -f "$C1_WD/epoch_6.pth" ] || ! grep -q 'Epoch(val) \[6\]' "$LOG" 2>/dev/null; do
  sleep 60
done
echo "[watch] $(date -u '+%F %T') ep6 present" >> "$OUT"

# ---- Stage A: same-epoch early curve -------------------------------------
python me_rssm/sanity/c1valid_curve_check.py --log "$LOG" \
  --work-dir "$C1_WD" --b0-log "$B0_WD" --out "$C1_WD/c1valid_ep6_early_check.json" >> "$OUT" 2>&1

# ---- Stage B: ep14 paired group-recall (C1 dump only; B0 frozen) ---------
if [ ! -f "$C1_WD/c1valid_ep14_dump.pkl" ]; then
  while [ ! -f "$C1_WD/epoch_14.pth" ]; do sleep 60; done
  CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/tool_c_masked_inference.py \
    --config "$C1_CFG" --checkpoint "$C1_WD/epoch_14.pth" \
    --coverage-groups "$COV" --motion-axis "$MOT" --mode identity \
    --out "$C1_WD/c1valid_ep14_dump.pkl" --gpu-id 0 >> "$OUT" 2>&1
fi
if [ ! -f "$C1_WD/c1valid_ep14_probe.json" ]; then
  python me_rssm/sanity/probe_c_failure_enrichment.py \
    --preds "$C1_WD/c1valid_ep14_dump.pkl" --config "$C1_CFG" \
    --coverage-groups "$COV" --motion-axis "$MOT" \
    --output "$C1_WD/c1valid_ep14_probe.json" >> "$OUT" 2>&1
fi
if [ ! -f "$C1_WD/c1valid_ep14_offline_ap.json" ]; then
  python me_rssm/sanity/eval_dump_c.py --config "$C1_CFG" \
    --dump "$C1_WD/c1valid_ep14_dump.pkl" \
    --out "$C1_WD/c1valid_ep14_offline_ap.json" >> "$OUT" 2>&1
fi
python me_rssm/sanity/c1valid_ep6_verdict.py \
  --c1-probe "$C1_WD/c1valid_ep14_probe.json" \
  --b0-probe "$B0_WD/c_failure_enrichment_ep14.json" \
  --c1-ap "$C1_WD/c1valid_ep14_offline_ap.json" \
  --b0-ap "$B0_WD/c_failpred_ep14_harness_ap.json" \
  --out "$C1_WD/c1valid_ep14_mechanism_verdict.json" >> "$OUT" 2>&1

# ---- Stage C: ep12-16 mature window --------------------------------------
while [ ! -f "$C1_WD/epoch_16.pth" ] && [ ! -f "$C1_WD/epoch_24.pth" ]; do
  grep -q 'Epoch(val) \[16\]' "$LOG" 2>/dev/null && break
  sleep 60
done
python me_rssm/sanity/c1valid_curve_check.py --log "$LOG" \
  --work-dir "$C1_WD" --b0-log "$B0_WD" --window 12-16 \
  --out "$C1_WD/c1valid_ep16_window_check.json" >> "$OUT" 2>&1

echo "[watch] $(date -u '+%F %T') done; decisions remain manual" >> "$OUT"
