#!/usr/bin/env bash
# PREPARED, NOT AUTOMATIC: continue the DC-RSSM-2 24e seed0 schedule from
# ep16 to ep24, but ONLY after the pre-registered ep16 gate returns GO.
#
# Why this exists: the ep16 watchdog stops the run so the mechanism probes
# get the whole GPU. If the AP gate passes, the schedule must go back up
# immediately (the family convention: ep16 gate -> continue), otherwise
# GPUs 5/6/7 sit idle while the verdict is being written. This script
# removes that gap without deciding anything on its own.
#
# Guards (all must hold, none is inferable from training state alone):
#   1. $WORK_DIR/gate_decision.txt exists and contains GO -- written by
#      the operator after reading the pre-registered gate result. Absent
#      or REJECT -> the script exits without launching anything.
#   2. the ep16 watchdog has finished its probe phase ("probes done" in
#      watch_seed0_ep16.log) -- the continuation must not fight the probes
#      for GPU memory.
#   3. GPUs 5/6/7 are free (< 2000 MiB used each).
#
# Usage (operator, after the gate verdict):
#   echo GO > /data/lurui/work_dirs/<dcrssm2 run>/gate_decision.txt
#   bash me_rssm/sanity/continue_dcrssm2_ep16.sh          # in tmux/screen
#
# Resume mechanics: --resume-from restores the epoch counter, so the
# CosineAnnealing schedule continues from ep16 instead of restarting, and
# new epochs append to the same work_dir log/curves (uniform ep1-24
# record). The dataloader shuffle order is NOT restored from a
# mid-schedule resume (mmcv restores sampler state only for --resume with
# the latest checkpoint meta) -- documented, and irrelevant to window
# means but it means ep17+ is not bitwise the uninterrupted schedule.
set -u
cd /home/lurui/workspace/R4Det

CFG=${CFG:-me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_dcrssm2.py}
WORK_DIR=${WORK_DIR:-/data/lurui/work_dirs/fgfull_N4_no2d_igdr_dcrssm2_2x4_24e_seed0}
WATCH_LOG=$WORK_DIR/watch_seed0_ep16.log
CKPT=$WORK_DIR/epoch_16.pth
LOG=$WORK_DIR/continue_ep16.log
SESSION=${SESSION:-dcrssm2_cont}

log() { echo "[cont] $(date -u '+%F %T') $*" | tee -a "$LOG"; }

mkdir -p "$WORK_DIR"
log "requested: resume $CFG from $CKPT to ep24"

if [ ! -f "$WORK_DIR/gate_decision.txt" ]; then
  log "REFUSING: $WORK_DIR/gate_decision.txt not found (write GO after the gate verdict)"
  exit 1
fi
if ! grep -qw GO "$WORK_DIR/gate_decision.txt"; then
  log "REFUSING: gate_decision.txt does not contain GO ($(cat "$WORK_DIR/gate_decision.txt"))"
  exit 1
fi
if [ ! -f "$CKPT" ]; then
  log "REFUSING: checkpoint $CKPT not found"
  exit 1
fi
log "waiting for the ep16 probe phase to finish (GUARD 2)"
for _ in $(seq 1 720); do
  grep -q 'probes done' "$WATCH_LOG" 2>/dev/null && break
  sleep 20
done
grep -q 'probes done' "$WATCH_LOG" 2>/dev/null \
  || { log "REFUSING: probes never reported done in $WATCH_LOG"; exit 1; }
log "probe phase done; waiting for GPUs 5/6/7 (GUARD 3)"
for _ in $(seq 1 180); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    -i 5,6,7 2>/dev/null | awk -F', ' '$2 > 2000 {n++} END {print n+0}')
  [ "$busy" = "0" ] && break
  sleep 20
done
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  -i 5,6,7 2>/dev/null | awk -F', ' '$2 > 2000 {n++} END {print n+0}')
if [ "$busy" != "0" ]; then
  log "REFUSING: GPUs 5/6/7 still busy ($busy of 3)"
  exit 1
fi

log "launching continuation ep16 -> ep24 (tmux $SESSION)"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "cd /home/lurui/workspace/R4Det && source .envrc && CUDA_VISIBLE_DEVICES=5,6,7 \
   bash tools/dist_train.sh $CFG 3 --seed 0 --deterministic \
   --resume-from $CKPT --work-dir $WORK_DIR >> $WORK_DIR/stdout.log 2>&1"
log "launched; watch with: tmux attach -t $SESSION"
