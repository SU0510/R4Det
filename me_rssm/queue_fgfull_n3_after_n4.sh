#!/usr/bin/env bash
# FG-FULL N=3 relay queue (unattended-safe).
#
# 1. Wait for the currently running FG-FULL N=4 process to exit.
# 2. Drain GPUs 5/6/7.
# 3. Run the N=3 CPU/preflight validation once the GPU slot is free.
# 4. Launch the N=3 run into its own work_dir. On crash: retry up to 3
#    times, resuming from latest.pth when one exists.
#
# Everything is logged to /data/lurui/work_dirs/fgfull_n3_queue.log

set -u
REPO=/home/lurui/workspace/R4Det
SOURCE_PATTERN='configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py'
N3_CFG=configs/r4det/TJ4D-R4Det_fgfull_N3_2x4_24e_pretrained_v2_head.py
WORK_DIR=/data/lurui/work_dirs/fgfull_N3_2x4_24e_seed0
LOG=/data/lurui/work_dirs/fgfull_n3_queue.log
MAX_ATTEMPTS=3

# r4det conda env must be on PATH: dist_train.sh calls bare `python`
export PATH="/home/lurui/envs/miniforge3/envs/r4det/bin:$PATH"

log() { echo "[fgfull-n3-queue] $(date '+%F %T') $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1
log "queue started (pid $$); waiting for N=4 training to exit"

# ---- Phase 1: wait for the N4 training process to exit ----
while pgrep -f "train_vod.py --config $SOURCE_PATTERN" >/dev/null 2>&1; do
    sleep 120
done
log "N=4 training process exited"

# ---- Phase 2: wait for GPUs 5/6/7 to drain (max 30 min) ----
drained=0
for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5,6,7 | sort -n | tail -1)
    if [ "${used:-99999}" -lt 2500 ]; then
        log "GPUs 5/6/7 drained (max ${used} MiB)"
        drained=1
        break
    fi
    sleep 30
done
[ "$drained" = "1" ] || log "WARNING: GPUs not fully drained after 30 min -> launching anyway (retry loop will handle OOM)"

# ---- Phase 3: pre-flight ----
mkdir -p "$WORK_DIR"
if ! CUDA_VISIBLE_DEVICES=5 python \
        me_rssm/sanity/test_fgfull_config.py "$N3_CFG" > "$WORK_DIR/preflight.log" 2>&1; then
    log "FATAL: N=3 pre-flight validation failed -> NOT launching. See $WORK_DIR/preflight.log"
    exit 1
fi
log "N=3 pre-flight validation passed"

# ---- Phase 4: launch with retry/resume ----
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    RESUME_ARGS=""
    if [ -e "$WORK_DIR/latest.pth" ]; then
        RESUME_ARGS="--resume-from $WORK_DIR/latest.pth"
        log "attempt $attempt/$MAX_ATTEMPTS: resuming from $WORK_DIR/latest.pth"
    else
        log "attempt $attempt/$MAX_ATTEMPTS: fresh start"
    fi
    CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh "$N3_CFG" 3 \
        --seed 0 --deterministic $RESUME_ARGS \
        --work-dir "$WORK_DIR" >> "$LOG" 2>&1
    rc=$?
    log "training exited rc=$rc"
    [ $rc -eq 0 ] && { log "training finished normally"; break; }
    attempt=$((attempt+1))
    [ $attempt -le $MAX_ATTEMPTS ] && { log "sleeping 180s before retry"; sleep 180; }
done
log "queue done"
