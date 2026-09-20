#!/usr/bin/env bash
# FG-FULL N=3 launcher on GPUs 0/1/2 (unattended-safe).
#
# The N=3 preflight has already passed on this config; this script only
# launches the run into its own work_dir. On crash: retry up to 3 times,
# resuming from latest.pth when one exists.
#
# Everything is logged to /data/lurui/work_dirs/fgfull_n3_gpu012_queue.log

set -u
REPO=/home/lurui/workspace/R4Det
N3_CFG=configs/r4det/TJ4D-R4Det_fgfull_N3_2x4_24e_pretrained_v2_head.py
WORK_DIR=/data/lurui/work_dirs/fgfull_N3_2x4_24e_seed0
GPU_LIST=0,1,2
NPROC=3
LOG=/data/lurui/work_dirs/fgfull_n3_gpu012_queue.log
MAX_ATTEMPTS=3

# r4det conda env must be on PATH: dist_train.sh calls bare `python`
export PATH="/home/lurui/envs/miniforge3/envs/r4det/bin:$PATH"

log() { echo "[fgfull-n3-gpu012] $(date '+%F %T') $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1
mkdir -p "$WORK_DIR"
log "launcher started (pid $$) on GPUs $GPU_LIST"

attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    RESUME_ARGS=""
    if [ -e "$WORK_DIR/latest.pth" ]; then
        RESUME_ARGS="--resume-from $WORK_DIR/latest.pth"
        log "attempt $attempt/$MAX_ATTEMPTS: resuming from $WORK_DIR/latest.pth"
    else
        log "attempt $attempt/$MAX_ATTEMPTS: fresh start"
    fi
    CUDA_VISIBLE_DEVICES=$GPU_LIST bash tools/dist_train.sh "$N3_CFG" $NPROC \
        --seed 0 --deterministic $RESUME_ARGS \
        --work-dir "$WORK_DIR" >> "$LOG" 2>&1
    rc=$?
    log "training exited rc=$rc"
    [ $rc -eq 0 ] && { log "training finished normally"; break; }
    attempt=$((attempt+1))
    [ $attempt -le $MAX_ATTEMPTS ] && { log "sleeping 180s before retry"; sleep 180; }
done
log "launcher done"
