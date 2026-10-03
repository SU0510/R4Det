#!/usr/bin/env bash
# MCI-RSSM ep12 gate watchdog (prepared 2026-10-02; launch is a user decision).
#
# Waits for epoch 12 validation + `epoch_12.pth` (with an mtime stale-guard so
# a leftover checkpoint from a previous attempt cannot trigger it), interrupts
# the training session, waits for the DDP workers to release the GPUs (up to
# 30 min — the seed1 chain incident showed torch distributed teardown can
# exceed 10 min), then freezes the ep12 gate numbers:
#   - three-window counterfactual probe (shuffle_z/zero_z head ratios)
#   - posterior source probe (specific-energy decomposition)
# The gate DECISION stays a human/agent step; this script only stops training
# and collects evidence. It never launches a follow-up run.
#
# Usage:
#   tmux new-session -d -s mci_ep12 \
#     'CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh \
#      me_rssm/sanity/mci_smoke... '   # NO — see docs/12 runbook for the
#                                      # exact 24e launch line, then:
#   tmux new-session -d -s mci_ep12_watch 'bash me_rssm/sanity/watch_mci_ep12.sh'
set -u

cd /home/lurui/workspace/R4Det

TRAIN_SESSION=mci_ep12
LOG=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mci_2x4_24e_seed0/stdout_ep12.log
WORK_DIR=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mci_2x4_24e_seed0
CFG=me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py
CKPT="$WORK_DIR/epoch_12.pth"
WATCH_LOG=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mci_2x4_24e_seed0/watch_ep12.log

START_TS=$(date +%s)
echo "[watch] $(date -u '+%F %T') waiting for ep12 val + checkpoint (start ts $START_TS) ..." >> "$WATCH_LOG"
while true; do
  if [ -f "$CKPT" ] && grep -q 'Epoch(val) \[12\]' "$LOG" 2>/dev/null; then
    CKPT_TS=$(stat -c %Y "$CKPT")
    if [ "$CKPT_TS" -gt "$START_TS" ]; then
      echo "[watch] $(date -u '+%F %T') gate hit (fresh ckpt); stopping $TRAIN_SESSION" >> "$WATCH_LOG"
      tmux send-keys -t "$TRAIN_SESSION" C-c 2>/dev/null
      break
    fi
    echo "[watch] $(date -u '+%F %T') stale epoch_12.pth ignored (mtime $CKPT_TS <= start $START_TS)" >> "$WATCH_LOG"
  fi
  sleep 30
done

# Wait for the DDP workers to release the GPUs (up to 30 min). GPU-idle is
# the stop criterion, not process exit — the 2026-10-01 seed1 incident.
for _ in $(seq 1 180); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    -i 5,6,7 2>/dev/null | awk -F', ' '$2 > 2000 {n++} END {print n+0}')
  if [ "$busy" = "0" ]; then
    break
  fi
  sleep 10
done
echo "[watch] $(date -u '+%F %T') GPUs released; running probes" >> "$WATCH_LOG"

# shellcheck disable=SC1091
source .envrc

CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_z_utilization_gated.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --limit 4 --batch-size 4 --shuffle-offset 2 \
  --output /data/lurui/work_dirs/fgfull_N4_no2d_igdr_mci_2x4_24e_seed0/z_util_gated_ep12_w3.json >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=6 python me_rssm/sanity/probe_z_utilization_gated.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 100 --limit 4 --batch-size 4 --shuffle-offset 2 \
  --output /data/lurui/work_dirs/fgfull_N4_no2d_igdr_mci_2x4_24e_seed0/z_util_gated_ep12_w100.json >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=7 python me_rssm/sanity/probe_z_utilization_gated.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset -1 \
  --output /data/lurui/work_dirs/fgfull_N4_no2d_igdr_mci_2x4_24e_seed0/z_util_gated_ep12_w1800.json >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_posterior_source_gated.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --limit 4 --stride 700 \
  --output /data/lurui/work_dirs/fgfull_N4_no2d_igdr_mci_2x4_24e_seed0/post_source_gated_ep12_stride700.json >> "$WATCH_LOG" 2>&1

echo "[watch] $(date -u '+%F %T') probes done; gate decision is a manual step" >> "$WATCH_LOG"
