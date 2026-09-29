#!/usr/bin/env bash
# MAI ep6 gate watchdog.
#
# Waits for epoch 6 to finish validation and land `epoch_6.pth`, interrupts the
# training session, waits for the DDP workers to release the GPUs, then runs
# the three-window counterfactual probe plus the posterior source probe that
# the ep6 gate is defined on. Writing the gate decision stays a human/agent
# step; this script only freezes the numbers.
#
# Usage: tmux new-session -d -s mai_ep6_watch 'bash me_rssm/sanity/watch_mai_ep6.sh'
set -u

cd /home/lurui/workspace/R4Det

TRAIN_SESSION=mai_ep6
LOG=/tmp/mai_ep6.log
WORK_DIR=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mai_2x4_24e_seed0
CFG=configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mai.py
CKPT="$WORK_DIR/epoch_6.pth"
WATCH_LOG=/tmp/mai_ep6_watch.log

echo "[watch] $(date -u '+%F %T') waiting for ep6 val + checkpoint ..." >> "$WATCH_LOG"
while true; do
  if grep -q 'Epoch(val) \[6\]' "$LOG" 2>/dev/null && [ -f "$CKPT" ]; then
    echo "[watch] $(date -u '+%F %T') gate hit; stopping $TRAIN_SESSION" >> "$WATCH_LOG"
    tmux send-keys -t "$TRAIN_SESSION" C-c 2>/dev/null
    break
  fi
  sleep 30
done

# Wait for the DDP workers to release the GPUs. Checking free memory is more
# robust than matching process names: the probe needs GPU 5/6/7 to itself.
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
  --output /tmp/z_util_mai_ep6_w3.json >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=6 python tools/diagnose_z_utilization.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 100 --limit 4 --batch-size 4 --shuffle-offset 2 \
  --output /tmp/z_util_mai_ep6_w100.json >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=7 python tools/diagnose_z_utilization.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset -1 \
  --output /tmp/z_util_mai_ep6_w1800.json >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=5 python tools/probe_posterior_source.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --limit 4 --stride 700 \
  --output /tmp/post_source_mai_ep6_stride700.json >> "$WATCH_LOG" 2>&1

echo "[watch] $(date -u '+%F %T') probes done" >> "$WATCH_LOG"
