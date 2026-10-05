#!/usr/bin/env bash
# Parameterized MCI seed gate watchdog: wait for epoch_16 val+ckpt of a
# given seed, stop the training session, run the gated probes into the
# seed's work_dir. Launch is a user-approved operation.
#
# Usage:
#   SEED=1 SESSION=mci_seed1 tmux new-session -d -s mci_seed1_watch \
#     'bash me_rssm/sanity/watch_mci_seed_ep16.sh'
set -u

cd /home/lurui/workspace/R4Det

SEED="${SEED:-1}"
SESSION="${SESSION:-mci_seed${SEED}}"
LOG="${LOG:-/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mci_2x4_24e_seed${SEED}/stdout.log}"
WORK_DIR="${WORK_DIR:-/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mci_2x4_24e_seed${SEED}}"
CFG=me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py
CKPT="$WORK_DIR/epoch_16.pth"
WATCH_LOG="$WORK_DIR/watch_seed${SEED}_ep16.log"

START_TS=$(date +%s)
echo "[watch] $(date -u '+%F %T') seed${SEED}: waiting for ep16 val + checkpoint (start ts $START_TS) ..." >> "$WATCH_LOG"
while true; do
  if [ -f "$CKPT" ] && grep -q 'Epoch(val) \[16\]' "$LOG" 2>/dev/null; then
    CKPT_TS=$(stat -c %Y "$CKPT")
    if [ "$CKPT_TS" -gt "$START_TS" ]; then
      echo "[watch] $(date -u '+%F %T') gate hit (fresh ckpt); stopping $SESSION" >> "$WATCH_LOG"
      tmux send-keys -t "$SESSION" C-c 2>/dev/null
      break
    fi
    echo "[watch] $(date -u '+%F %T') stale epoch_16.pth ignored" >> "$WATCH_LOG"
  fi
  sleep 30
done

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
  --output "$WORK_DIR/z_util_gated_ep16_w3.json" >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=6 python me_rssm/sanity/probe_z_utilization_gated.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 100 --limit 4 --batch-size 4 --shuffle-offset 2 \
  --output "$WORK_DIR/z_util_gated_ep16_w100.json" >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=7 python me_rssm/sanity/probe_z_utilization_gated.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset -1 \
  --output "$WORK_DIR/z_util_gated_ep16_w1800.json" >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_posterior_source_gated.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --limit 4 --stride 700 \
  --output "$WORK_DIR/post_source_gated_ep16_stride700.json" >> "$WATCH_LOG" 2>&1

echo "[watch] $(date -u '+%F %T') probes done; gate decision is a manual step" >> "$WATCH_LOG"
