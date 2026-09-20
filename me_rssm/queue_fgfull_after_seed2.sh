#!/usr/bin/env bash
# fgfull relay queue (unattended-safe).
#
# 1. Wait for cyccls seed2 to save epoch_18.pth (epoch-18 val done + ckpt on
#    disk, per user request: no need to run to ep24). If seed2 dies before
#    that, proceed anyway (nothing to wait for).
# 2. Kill seed2 (pattern matches only its config name), drain GPUs 5/6/7.
# 3. Pre-flight: rebuild + validate the fgfull config (fail fast instead of
#    launching a broken run).
# 4. Launch fg-full training on GPUs 5,6,7 with the run10-seed_0 protocol
#    (seed 0, deterministic, same schedule). On crash: retry up to 3 times,
#    resuming from latest.pth when one exists (OOM at iter 0 has no ckpt ->
#    fresh retry; mid-run crash resumes without losing epochs).
#
# Everything is logged to /data/lurui/work_dirs/fgfull_queue.log

set -u
REPO=/home/lurui/workspace/R4Det
SEED2_DIR=/data/lurui/work_dirs/cyccls_branch_N4_2x4_24e_multiseed/seed_2
WORK_DIR=/data/lurui/work_dirs/fgfull_N4_2x4_24e_seed0
CFG=configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py
LOG=/data/lurui/work_dirs/fgfull_queue.log
KILL_PATTERN="pretrained_v2_head_cyccls.py"
SEED2_TMUX=cyccls_seed2
MAX_ATTEMPTS=3

# r4det conda env must be on PATH: dist_train.sh calls bare `python`
export PATH="/home/lurui/envs/miniforge3/envs/r4det/bin:$PATH"

log() { echo "[fgfull-queue] $(date '+%F %T') $*" | tee -a "$LOG"; }

log "queue started (pid $$); waiting for seed2 epoch_18 checkpoint"

# ---- Phase 1: wait for seed2 ep18 ----
while true; do
    if [ -f "$SEED2_DIR/epoch_18.pth" ]; then
        sleep 30
        if [ -f "$SEED2_DIR/epoch_18.pth" ]; then
            log "seed2 epoch_18.pth on disk (val+ckpt saved) -> stop seed2"
            break
        fi
    fi
    if ! tmux has-session -t "$SEED2_TMUX" 2>/dev/null; then
        sleep 60
        if ! tmux has-session -t "$SEED2_TMUX" 2>/dev/null; then
            log "WARNING: seed2 tmux gone before epoch_18 (crashed or finished?) -> proceeding"
            break
        fi
    fi
    sleep 120
done

# ---- Phase 2: stop seed2 ----
if pgrep -f "$KILL_PATTERN" >/dev/null 2>&1; then
    pkill -f "$KILL_PATTERN"
    sleep 20
    pkill -9 -f "$KILL_PATTERN" 2>/dev/null
    log "seed2 processes killed"
else
    log "no seed2 process found to kill"
fi
tmux kill-session -t "$SEED2_TMUX" 2>/dev/null && log "tmux $SEED2_TMUX killed" \
    || log "tmux $SEED2_TMUX already gone"

# ---- Phase 3: wait for GPUs 5/6/7 to drain (max 30 min) ----
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

# ---- Phase 4: pre-flight ----
cd "$REPO" || exit 1
mkdir -p "$WORK_DIR"
if ! CUDA_VISIBLE_DEVICES=5 /home/lurui/envs/miniforge3/envs/r4det/bin/python \
        me_rssm/sanity/test_fgfull_config.py > "$WORK_DIR/preflight.log" 2>&1; then
    log "FATAL: pre-flight validation failed -> NOT launching. See $WORK_DIR/preflight.log"
    exit 1
fi
log "pre-flight validation passed"

# ---- Phase 5: launch with retry/resume ----
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    RESUME_ARGS=""
    if [ -e "$WORK_DIR/latest.pth" ]; then
        RESUME_ARGS="--resume-from $WORK_DIR/latest.pth"
        log "attempt $attempt/$MAX_ATTEMPTS: resuming from $WORK_DIR/latest.pth"
    else
        log "attempt $attempt/$MAX_ATTEMPTS: fresh start"
    fi
    CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh "$CFG" 3 \
        --seed 0 --deterministic $RESUME_ARGS \
        --work-dir "$WORK_DIR" >> "$LOG" 2>&1
    rc=$?
    log "training exited rc=$rc"
    [ $rc -eq 0 ] && { log "training finished normally"; break; }
    attempt=$((attempt+1))
    [ $attempt -le $MAX_ATTEMPTS ] && { log "sleeping 180s before retry"; sleep 180; }
done
log "queue done"
