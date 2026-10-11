#!/usr/bin/env bash
# A2 ep6 gate watchdog (pre-registered S1 stop point), A2-S and A2-R.
#
# Same discipline as watch_semA_ep6.sh, with three A2-specific changes:
#   1. the gate reads the FROZEN 20-clip contract (--stride 100 --limit 20,
#      prereg section 5) so the sampling cannot be re-chosen after the fact;
#   2. the stop signal is the training PID (nohup/setsid launch), not tmux;
#   3. the pre-registered M1'/M1-loc readings come from probe_semA_rank.py on
#      BOTH splits, plus the 48-channel coverage reading for the A2-R arm.
#
# The DECISION stays manual: this script only stops training and freezes
# evidence. It never launches a follow-up run.
#
# Usage:
#   ARM=A2S setsid nohup bash me_rssm/sanity/watch_semA2_ep6.sh &
set -u

cd /home/lurui/workspace/R4Det

ARM="${ARM:-A2S}"
case "$ARM" in
  A2S) CFG_DEFAULT=me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semA2S_24e.py ;;
  A2R) CFG_DEFAULT=me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semA2R_24e.py ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac

WORK_DIR="${WORK_DIR:-/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_sem${ARM}_seed0}"
LOG="${LOG:-/tmp/sem${ARM}_24e.log}"
CFG="${CFG:-$CFG_DEFAULT}"
CKPT="$WORK_DIR/epoch_6.pth"
WATCH_LOG="$WORK_DIR/watch_sem${ARM}_ep6.log"
STOP_TIMEOUT_SEC="${STOP_TIMEOUT_SEC:-1800}"
TRAIN_PIDFILE="${TRAIN_PIDFILE:-/tmp/sem${ARM}_24e.pid}"

mkdir -p "$WORK_DIR"
START_TS=$(date +%s)
echo "[watch-$ARM] $(date -u '+%F %T') waiting for ep6 val + fresh checkpoint (ts $START_TS)" >> "$WATCH_LOG"

while true; do
  if [ -f "$CKPT" ] && grep -q 'Epoch(val) \[6\]' "$LOG" 2>/dev/null; then
    CKPT_TS=$(stat -c %Y "$CKPT")
    if [ "$CKPT_TS" -gt "$START_TS" ]; then
      echo "[watch-$ARM] $(date -u '+%F %T') gate hit (fresh ckpt)" >> "$WATCH_LOG"
      break
    fi
    echo "[watch-$ARM] $(date -u '+%F %T') stale epoch_6.pth ignored" >> "$WATCH_LOG"
  fi
  sleep 30
done

# Stop training. torchrun/DDP teardown is not instantaneous, so wait for the
# GPUs to actually drain: running the battery on busy GPUs only yields OOM.
if [ -f "$TRAIN_PIDFILE" ]; then
  PID=$(cat "$TRAIN_PIDFILE")
  echo "[watch-$ARM] $(date -u '+%F %T') sending SIGINT to training pid $PID" >> "$WATCH_LOG"
  kill -INT "$PID" 2>/dev/null || true
fi
pkill -INT -f "sem${ARM}_24e.py" 2>/dev/null || true

waited=0
while [ "$waited" -lt "$STOP_TIMEOUT_SEC" ]; do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    -i 5,6,7 2>/dev/null | awk -F', ' '$2 > 2000 {n++} END {print n+0}')
  [ "$busy" = "0" ] && break
  sleep 15
  waited=$((waited + 15))
done
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  -i 5,6,7 2>/dev/null | awk -F', ' '$2 > 2000 {n++} END {print n+0}')
if [ "$busy" != "0" ]; then
  echo "[watch-$ARM] $(date -u '+%F %T') ABORT: $busy/3 GPUs busy after ${waited}s" >> "$WATCH_LOG"
  exit 1
fi
echo "[watch-$ARM] $(date -u '+%F %T') GPUs free after ${waited}s; running probes" >> "$WATCH_LOG"

# shellcheck disable=SC1091
source .envrc
if ! command -v python >/dev/null 2>&1 || ! python -c 'import torch, mmcv, mmdet3d' >/dev/null 2>&1; then
  echo "[watch-$ARM] $(date -u '+%F %T') ABORT: python/torch/mmcv/mmdet3d unavailable" >> "$WATCH_LOG"
  exit 1
fi

# --- the pre-registered S1 gate: M1-M4 on the frozen 20-clip contract ----
CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_semreadout_ep6.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --limit 20 --stride 100 --shuffle-offset 1 \
  --output "$WORK_DIR/semreadout_ep6_gate.json" >> "$WATCH_LOG" 2>&1

# --- M1' / M1-loc: scale-free + localisation readings, BOTH splits -------
CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_semA_rank.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --stride 100 --limit 20 \
  --output "$WORK_DIR/semA_rank_val_ep6.json" >> "$WATCH_LOG" 2>&1
CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_semA_rank.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --stride 100 --limit 20 \
  --split-ann-file data/TJ4D/TJ4D_infos_train.pkl \
  --output "$WORK_DIR/semA_rank_train_ep6.json" >> "$WATCH_LOG" 2>&1

# --- M4 corrected reading (prior floor + scale curve + flip fraction) ----
CUDA_VISIBLE_DEVICES=6 python me_rssm/sanity/probe_semA_m4floor.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --stride 100 --limit 20 \
  --output "$WORK_DIR/semA_m4floor_ep6.json" >> "$WATCH_LOG" 2>&1

# --- split attribution (unchanged from A; needed for cross-generation) ---
CUDA_VISIBLE_DEVICES=6 python me_rssm/sanity/probe_semA_bnmode.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --stride 100 --limit 20 \
  --output "$WORK_DIR/semA_bnmode_ep6.json" >> "$WATCH_LOG" 2>&1

# --- family-standard counterfactual readout (cross-generation comparison) -
CUDA_VISIBLE_DEVICES=7 python me_rssm/sanity/probe_z_utilization_standard.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --limit 4 --batch-size 4 --shuffle-offset 2 \
  --output "$WORK_DIR/z_util_std_ep6_w3.json" >> "$WATCH_LOG" 2>&1
CUDA_VISIBLE_DEVICES=7 python me_rssm/sanity/probe_z_utilization_standard.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 100 --limit 4 --batch-size 4 --shuffle-offset 2 \
  --output "$WORK_DIR/z_util_std_ep6_w100.json" >> "$WATCH_LOG" 2>&1

# --- A2-R only: does writing all 48 channels raise the flip fraction? -----
if [ "$ARM" = "A2R" ]; then
  CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_semA_rank.py \
    --config "$CFG" --checkpoint "$CKPT" \
    --start-index 3 --stride 100 --limit 20 \
    --output "$WORK_DIR/semA_rank_val_ep6_a2r.json" >> "$WATCH_LOG" 2>&1
fi

echo "[watch-$ARM] $(date -u '+%F %T') probes done; gate decision is a manual step" >> "$WATCH_LOG"
