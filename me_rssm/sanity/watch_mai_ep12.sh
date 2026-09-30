#!/usr/bin/env bash
# MAI ep12 gate watchdog: same contract as watch_mai_ep6.sh, one epoch later.
#
# Waits for epoch 12 validation and `epoch_12.pth`, interrupts training, waits
# for GPUs 5/6/7 to be released, then runs the three-window counterfactual
# probe and the posterior source probe the ep12 gate is defined on.
set -u

cd /home/lurui/workspace/R4Det

TRAIN_SESSION=mai_ep12
LOG=/tmp/mai_ep12.log
WORK_DIR=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mai_2x4_24e_seed0
CFG=configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mai.py
CKPT="$WORK_DIR/epoch_12.pth"
WATCH_LOG=/tmp/mai_ep12_watch.log

echo "[watch] $(date -u '+%F %T') waiting for ep12 val + checkpoint ..." >> "$WATCH_LOG"
while true; do
  if grep -q 'Epoch(val) \[12\]' "$LOG" 2>/dev/null && [ -f "$CKPT" ]; then
    echo "[watch] $(date -u '+%F %T') gate hit; stopping $TRAIN_SESSION" >> "$WATCH_LOG"
    tmux send-keys -t "$TRAIN_SESSION" C-c 2>/dev/null
    break
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

CUDA_VISIBLE_DEVICES=5 python tools/diagnose_z_utilization.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --limit 4 --batch-size 4 --shuffle-offset 2 \
  --output /tmp/z_util_mai_ep12_w3.json >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=6 python tools/diagnose_z_utilization.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 100 --limit 4 --batch-size 4 --shuffle-offset 2 \
  --output /tmp/z_util_mai_ep12_w100.json >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=7 python tools/diagnose_z_utilization.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset -1 \
  --output /tmp/z_util_mai_ep12_w1800.json >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=5 python tools/probe_posterior_source.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --limit 4 --stride 700 \
  --output /tmp/post_source_mai_ep12_stride700.json >> "$WATCH_LOG" 2>&1

echo "[watch] $(date -u '+%F %T') probes done" >> "$WATCH_LOG"
