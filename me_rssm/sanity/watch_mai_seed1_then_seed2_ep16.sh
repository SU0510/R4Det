#!/usr/bin/env bash
# MAI ep16 chain watchdog:
#   seed1 ep16 gate -> stop seed1 -> seed1 probes -> verify GPUs idle
#   -> start seed2 -> seed2 ep16 gate -> stop seed2 -> seed2 probes.
#
# Why one chained watchdog instead of two independent watchdogs
# -------------------------------------------------------------
# `watch_mai_ep16.sh` stops a run at ep16 and then holds GPU 5/6/7 while it runs
# the three-window counterfactual probe and the posterior-source probe. If a
# second watchdog were started independently, it could see the GPUs as free and
# launch seed2 while the seed1 probes were still running. This script serializes
# the whole hand-off and only starts seed2 after seed1's training session has
# exited, its probes have finished, and all three GPUs are idle again.
#
# It also survives the unattended-night failure modes:
#   - the training session dying before ep16 -> resume from the latest
#     `epoch_*.pth` in that run's work_dir (up to MAX_ATTEMPTS total attempts);
#   - a stalled session (log mtime older than STALL_TIMEOUT_SEC) -> SIGINT and
#     resume from the latest checkpoint;
#   - a checkpoint that is still being written -> wait until the file has a
#     stable mtime and a plausible size before treating the gate as hit;
#   - probe failures -> logged as warnings, never block the next phase.
#
# Default mode is a read-only dry run. Set RUN_CHAIN=1 to execute.
#
# Production launch (env must be inside the tmux command string):
#   tmux new-session -d -s mai_seed1_seed2_chain \
#     'cd /home/lurui/workspace/R4Det && RUN_CHAIN=1 \
#      bash me_rssm/sanity/watch_mai_seed1_then_seed2_ep16.sh'
#
# Dry run:
#   bash me_rssm/sanity/watch_mai_seed1_then_seed2_ep16.sh
#
# Rehearsal-only overrides (defaults are the production values and must stay
# unchanged for the real run): TRAIN_ENTRY, TRAIN_ENV, PROBE_Z_CMD,
# PROBE_SRC_CMD, GPU_QUERY_CMD, CKPT_MIN_BYTES, CKPT_SETTLE_SEC, EPOCH.
set -u

cd /home/lurui/workspace/R4Det

RUN_CHAIN=${RUN_CHAIN:-0}
SOURCE_ONLY=${SOURCE_ONLY:-0}
GPU_LIST=${GPU_LIST:-5,6,7}
CFG=${CFG:-configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mai.py}
EPOCH=${EPOCH:-16}

# Phase 1: seed1 (already training when this watcher starts)
SEED1_SESSION=${SEED1_SESSION:-mai_seed1_ep16}
SEED1_LOG=${SEED1_LOG:-/tmp/mai_seed1_ep16.log}
SEED1_WORK_DIR=${SEED1_WORK_DIR:-/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mai_2x4_24e_seed1}
SEED1_TAG=${SEED1_TAG:-mai_seed1_ep16}
SEED1_GATE_TIMEOUT_SEC=${SEED1_GATE_TIMEOUT_SEC:-43200}   # 12 h from chain start

# Phase 2: seed2 (started by this watcher)
SEED2_SESSION=${SEED2_SESSION:-mai_seed2_ep16}
SEED2_LOG=${SEED2_LOG:-/tmp/mai_seed2_ep16.log}
SEED2_WORK_DIR=${SEED2_WORK_DIR:-/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mai_2x4_24e_seed2}
SEED2_TAG=${SEED2_TAG:-mai_seed2_ep16}
SEED2_SEED=${SEED2_SEED:-2}
SEED2_GATE_TIMEOUT_SEC=${SEED2_GATE_TIMEOUT_SEC:-129600}  # 36 h from seed2 start

CHAIN_LOG=${CHAIN_LOG:-/tmp/mai_seed1_then_seed2_ep16_chain.log}
RELEASE_TIMEOUT_SEC=${RELEASE_TIMEOUT_SEC:-1800}
STOP_TIMEOUT_SEC=${STOP_TIMEOUT_SEC:-600}
STALL_TIMEOUT_SEC=${STALL_TIMEOUT_SEC:-2700}
POLL_SEC=${POLL_SEC:-30}
RESTART_DELAY_SEC=${RESTART_DELAY_SEC:-60}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
CKPT_MIN_BYTES=${CKPT_MIN_BYTES:-100000000}
CKPT_SETTLE_SEC=${CKPT_SETTLE_SEC:-20}
PROBE_TIMEOUT_SEC=${PROBE_TIMEOUT_SEC:-1800}
GPU_BUSY_MIB=${GPU_BUSY_MIB:-500}

# Rehearsal-only overrides. Production defaults below are the unmodified tools.
TRAIN_ENTRY=${TRAIN_ENTRY:-tools/dist_train.sh}
TRAIN_ENV=${TRAIN_ENV:-}
PROBE_Z_CMD=${PROBE_Z_CMD:-python tools/diagnose_z_utilization.py}
PROBE_SRC_CMD=${PROBE_SRC_CMD:-python tools/probe_posterior_source.py}
GPU_QUERY_CMD=${GPU_QUERY_CMD:-nvidia-smi}

SEED1_CKPT="$SEED1_WORK_DIR/epoch_${EPOCH}.pth"
SEED2_CKPT="$SEED2_WORK_DIR/epoch_${EPOCH}.pth"
WATCH_START=$(date +%s)
FINAL_RC=0

mkdir -p "$(dirname "$CHAIN_LOG")"
touch "$CHAIN_LOG"

log() {
  echo "[chain] $(date -u '+%F %T') $*" >> "$CHAIN_LOG"
}

die() {
  FINAL_RC=1
  log "FATAL: $*"
  exit 1
}

require_file() {
  [ -f "$1" ] || die "missing file: $1"
}

session_exists() {
  tmux has-session -t "$1" 2>/dev/null
}

training_process_running() {
  local session=$1 work_dir=$2 expected_seed=$3
  session_exists "$session" \
    && ps -eo cmd \
      | grep -F -- "$work_dir" \
      | grep -F -- "--seed $expected_seed" \
      | grep -v grep >/dev/null
}

latest_checkpoint() {
  local work_dir=$1 f size mtime now
  now=$(date +%s)
  for f in "$work_dir"/epoch_*.pth; do
    [ -f "$f" ] || continue
    size=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$size" -ge "$CKPT_MIN_BYTES" ] || continue
    mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    [ $((now - mtime)) -ge "$CKPT_SETTLE_SEC" ] || continue
    printf '%s\n' "$f"
  done | sort -V | tail -1
}

gpu_busy_count() {
  local out
  out=$($GPU_QUERY_CMD --query-gpu=memory.used --format=csv,noheader,nounits \
    -i "$GPU_LIST" 2>/dev/null) || return 9
  [ -n "$out" ] || return 9
  printf '%s\n' "$out" \
    | awk -v limit="$GPU_BUSY_MIB" '$1 > limit {n++} END {print n+0}'
}

wait_for_gpus_idle() {
  local timeout=$1 deadline busy
  deadline=$(( $(date +%s) + timeout ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if busy=$(gpu_busy_count); then
      [ "$busy" = "0" ] && return 0
    fi
    sleep 10
  done
  return 1
}

gate_hit() {
  local log_file=$1 ckpt=$2 since=$3 log_start_line=$4 mtime size now
  tail -n +$((log_start_line + 1)) "$log_file" 2>/dev/null \
    | grep -q "Epoch(val) \[$EPOCH\]" || return 1
  [ -f "$ckpt" ] || return 1
  mtime=$(stat -c %Y "$ckpt" 2>/dev/null || echo 0)
  size=$(stat -c %s "$ckpt" 2>/dev/null || echo 0)
  now=$(date +%s)
  [ "$mtime" -ge $((since - 5)) ] || return 1
  [ "$size" -ge "$CKPT_MIN_BYTES" ] || return 1
  [ $((now - mtime)) -ge "$CKPT_SETTLE_SEC" ] || return 1
  return 0
}

gate_hit_any_time() {
  local log_file=$1 ckpt=$2
  grep -q "Epoch(val) \[$EPOCH\]" "$log_file" 2>/dev/null || return 1
  [ -f "$ckpt" ] || return 1
  return 0
}

audit_config() {
  python -c "
from mmcv import Config
c = Config.fromfile('$CFG')
print(c.model.temporal_fusion.type)
print(c.runner.max_epochs)
print(c.evaluation.interval)
print(c.checkpoint_config.interval)
print(getattr(c.data.val, 'samples_per_gpu', 1))
" 2>/dev/null
}

check_prerequisites() {
  local ok=1 audit fusion maxe evali ckpti valbs
  require_file "$CFG" || ok=0
  require_file "$TRAIN_ENTRY" || ok=0
  require_file tools/diagnose_z_utilization.py || ok=0
  require_file tools/probe_posterior_source.py || ok=0

  [ "$RUN_CHAIN" = "1" ] || log "dry-run: RUN_CHAIN is not 1; no stop/start will be performed"

  if session_exists "$SEED2_SESSION"; then
    log "FATAL: tmux session already exists: $SEED2_SESSION"
    ok=0
  fi
  if [ -e "$SEED2_CKPT" ]; then
    log "FATAL: seed2 epoch${EPOCH} checkpoint already exists: $SEED2_CKPT"
    ok=0
  fi
  if [ -d "$SEED2_WORK_DIR" ] && [ "$RUN_CHAIN" = "1" ]; then
    log "FATAL: seed2 workdir already exists: $SEED2_WORK_DIR"
    ok=0
  fi
  if [ "$RUN_CHAIN" = "1" ] \
    && ! session_exists "$SEED1_SESSION" \
    && [ ! -d "$SEED1_WORK_DIR" ]; then
    log "FATAL: seed1 session and workdir are both missing"
    ok=0
  fi
  if [ -e "$SEED2_LOG" ]; then
    log "FATAL: seed2 log already exists: $SEED2_LOG"
    ok=0
  fi

  if ! source .envrc >/dev/null 2>&1; then
    log "FATAL: could not source .envrc"
    ok=0
  fi

  if [ "$ok" = "1" ]; then
    if ! audit=$(audit_config); then
      log "FATAL: could not load config for audit: $CFG"
      ok=0
    else
      fusion=$(printf '%s\n' "$audit" | sed -n '1p')
      maxe=$(printf '%s\n' "$audit" | sed -n '2p')
      evali=$(printf '%s\n' "$audit" | sed -n '3p')
      ckpti=$(printf '%s\n' "$audit" | sed -n '4p')
      valbs=$(printf '%s\n' "$audit" | sed -n '5p')
      log "config audit: fusion=$fusion max_epochs=$maxe eval_interval=$evali ckpt_interval=$ckpti val_bs=$valbs"
      [ "$fusion" = "MotionAlignedInnovationRSSMFusion" ] || { log "FATAL: unexpected fusion type: $fusion"; ok=0; }
      [ "$maxe" = "24" ] || { log "FATAL: unexpected max_epochs: $maxe"; ok=0; }
      [ "$evali" = "1" ] || { log "FATAL: unexpected eval interval: $evali"; ok=0; }
      [ $((EPOCH % ckpti)) -eq 0 ] || { log "FATAL: epoch ${EPOCH} not covered by ckpt interval ${ckpti}"; ok=0; }
    fi
  fi

  log "seed1 session=$SEED1_SESSION log=$SEED1_LOG workdir=$SEED1_WORK_DIR"
  log "seed2 session=$SEED2_SESSION log=$SEED2_LOG workdir=$SEED2_WORK_DIR seed=$SEED2_SEED"
  log "config=$CFG epoch=$EPOCH gpus=$GPU_LIST mode=$([ "$RUN_CHAIN" = "1" ] && echo RUN || echo DRY_RUN)"

  [ "$ok" = "1" ] || exit 1
}

wait_for_training_stopped() {
  local session=$1 work_dir=$2 seed=$3 timeout=$4 deadline
  deadline=$(( $(date +%s) + timeout ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    training_process_running "$session" "$work_dir" "$seed" || return 0
    sleep 5
  done
  return 1
}

stop_session() {
  local label=$1 session=$2 work_dir=$3 seed=$4
  if training_process_running "$session" "$work_dir" "$seed"; then
    log "$label sending SIGINT to $session"
    tmux send-keys -t "$session" C-c 2>/dev/null || true
  fi
  if ! wait_for_training_stopped "$session" "$work_dir" "$seed" "$STOP_TIMEOUT_SEC"; then
    return 1
  fi
  if session_exists "$session"; then
    log "$label training stopped; killing lingering tmux session $session"
    tmux kill-session -t "$session" 2>/dev/null || true
  fi
  return 0
}

spawn_training() {
  local label=$1 session=$2 log_file=$3 work_dir=$4 seed=$5
  local resume_from=${6:-} append=$7 cmd redir deadline
  mkdir -p "$work_dir"
  cmd="cd /home/lurui/workspace/R4Det && source .envrc && CUDA_VISIBLE_DEVICES=$GPU_LIST SEED=$seed"
  [ -n "$TRAIN_ENV" ] && cmd="$cmd $TRAIN_ENV"
  cmd="$cmd bash $TRAIN_ENTRY $CFG 3 --seed $seed --deterministic"
  [ -n "$resume_from" ] && cmd="$cmd --resume-from $resume_from"
  cmd="$cmd --work-dir $work_dir"
  if [ "$append" = "1" ]; then redir=">>"; else redir=">"; fi
  cmd="$cmd $redir $log_file 2>&1"
  log "$label launch: $cmd"
  tmux new-session -d -s "$session" "$cmd" || return 1
  deadline=$(( $(date +%s) + 180 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    training_process_running "$session" "$work_dir" "$seed" && return 0
    session_exists "$session" || return 1
    sleep 5
  done
  return 1
}

reach_gate() {
  # args: label session log_file work_dir seed since deadline attach_first log_start_line
  local label=$1 session=$2 log_file=$3 work_dir=$4 seed=$5
  local since=$6 deadline=$7 attach_first=$8 log_start_line=$9
  local ckpt="$work_dir/epoch_${EPOCH}.pth"
  local attempt=0 rc latest resume attempt_started last_mtime now

  while true; do
    attempt=$((attempt + 1))
    if gate_hit "$log_file" "$ckpt" "$since" "$log_start_line"; then return 0; fi
    if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then return 3; fi
    [ "$(date +%s)" -lt "$deadline" ] || return 2

    if [ "$attempt" -eq 1 ] && [ "$attach_first" = "1" ]; then
      log "$label attaching to running session $session"
      attempt_started=$(date +%s)
    else
      if session_exists "$session"; then
        log "$label clearing lingering session $session before resume"
        tmux kill-session -t "$session" 2>/dev/null || true
      fi
      latest=$(latest_checkpoint "$work_dir")
      resume=$latest
      if [ -z "$resume" ]; then
        log "$label no checkpoint available; starting from scratch"
      else
        log "$label resuming from $resume"
      fi
      if ! wait_for_gpus_idle "$RELEASE_TIMEOUT_SEC"; then
        log "$label GPUs not idle before launch"
        return 4
      fi
      attempt_started=$(date +%s)
      if ! spawn_training "$label" "$session" "$log_file" "$work_dir" "$seed" \
        "$resume" "$([ "$attempt" -eq 1 ] && echo 0 || echo 1)"; then
        log "$label launch failed on attempt $attempt"
        [ "$attempt" -ge "$MAX_ATTEMPTS" ] && return 4
        sleep "$RESTART_DELAY_SEC"
        continue
      fi
    fi

    while true; do
      if gate_hit "$log_file" "$ckpt" "$since" "$log_start_line"; then return 0; fi
      if ! training_process_running "$session" "$work_dir" "$seed"; then
        log "$label training process exited before ep${EPOCH} (attempt $attempt)"
        break
      fi
      now=$(date +%s)
      if [ "$now" -ge "$deadline" ]; then return 2; fi
      if [ $((now - attempt_started)) -ge "$STALL_TIMEOUT_SEC" ]; then
        last_mtime=$(stat -c %Y "$log_file" 2>/dev/null || echo "$now")
        if [ $((now - last_mtime)) -ge "$STALL_TIMEOUT_SEC" ]; then
          log "$label log stalled for $((now - last_mtime))s; restarting from checkpoint"
          tmux send-keys -t "$session" C-c 2>/dev/null || true
          wait_for_training_stopped "$session" "$work_dir" "$seed" "$STOP_TIMEOUT_SEC" || true
          break
        fi
      fi
      sleep "$POLL_SEC"
    done

    if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then return 3; fi
    wait_for_gpus_idle "$RELEASE_TIMEOUT_SEC" || return 4
    sleep "$RESTART_DELAY_SEC"
  done
}

run_probes() {
  local tag=$1 ckpt=$2 probe_log=$3 rc=0
  log "running $tag probes"
  timeout "$PROBE_TIMEOUT_SEC" env CUDA_VISIBLE_DEVICES=5 $PROBE_Z_CMD \
    --config "$CFG" --checkpoint "$ckpt" \
    --start-index 3 --limit 4 --batch-size 4 --shuffle-offset 2 \
    --output "/tmp/z_util_${tag}_w3.json" >> "$probe_log" 2>&1 \
    || { log "WARN: $tag probe w3 failed"; rc=1; }

  timeout "$PROBE_TIMEOUT_SEC" env CUDA_VISIBLE_DEVICES=6 $PROBE_Z_CMD \
    --config "$CFG" --checkpoint "$ckpt" \
    --start-index 100 --limit 4 --batch-size 4 --shuffle-offset 2 \
    --output "/tmp/z_util_${tag}_w100.json" >> "$probe_log" 2>&1 \
    || { log "WARN: $tag probe w100 failed"; rc=1; }

  timeout "$PROBE_TIMEOUT_SEC" env CUDA_VISIBLE_DEVICES=7 $PROBE_Z_CMD \
    --config "$CFG" --checkpoint "$ckpt" \
    --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset -1 \
    --output "/tmp/z_util_${tag}_w1800.json" >> "$probe_log" 2>&1 \
    || { log "WARN: $tag probe w1800 failed"; rc=1; }

  timeout "$PROBE_TIMEOUT_SEC" env CUDA_VISIBLE_DEVICES=5 $PROBE_SRC_CMD \
    --config "$CFG" --checkpoint "$ckpt" \
    --start-index 3 --limit 4 --stride 700 \
    --output "/tmp/post_source_${tag}_stride700.json" >> "$probe_log" 2>&1 \
    || { log "WARN: $tag source probe failed"; rc=1; }

  for f in \
    "/tmp/z_util_${tag}_w3.json" \
    "/tmp/z_util_${tag}_w100.json" \
    "/tmp/z_util_${tag}_w1800.json" \
    "/tmp/post_source_${tag}_stride700.json"; do
    if [ -f "$f" ]; then
      log "$tag probe output: $f ($(stat -c %s "$f") bytes)"
    else
      log "WARN: $tag probe output missing: $f"
      rc=1
    fi
  done
  return "$rc"
}

handoff_seed1() {
  local rc=0
  local seed1_deadline=$(( $(date +%s) + SEED1_GATE_TIMEOUT_SEC ))
  log "waiting for seed1 ep${EPOCH} val + checkpoint"
  SEED1_LOG_START_LINE=$([ -f "$SEED1_LOG" ] && wc -l < "$SEED1_LOG" 2>/dev/null || echo 0)
  if reach_gate "seed1" "$SEED1_SESSION" "$SEED1_LOG" "$SEED1_WORK_DIR" 1 \
    "$WATCH_START" "$seed1_deadline" 1 "$SEED1_LOG_START_LINE"; then
    log "seed1 ep${EPOCH} gate hit"
  else
    rc=$?
    log "WARN: seed1 did not reach ep${EPOCH} gate (rc=$rc)"
    if session_exists "$SEED1_SESSION" || ! wait_for_gpus_idle 60; then
      die "seed1 is unresolved and GPUs are not free; refusing to start seed2"
    fi
    log "WARN: seed1 exited without gate but GPUs are free; continuing to seed2"
    FINAL_RC=1
    return 0
  fi

  stop_session "seed1" "$SEED1_SESSION" "$SEED1_WORK_DIR" 1 \
    || die "seed1 did not stop cleanly"
  wait_for_gpus_idle "$RELEASE_TIMEOUT_SEC" \
    || die "GPUs busy after seed1 stop: $GPU_LIST"
  log "seed1 exited and GPUs are idle"

  if ! run_probes "$SEED1_TAG" "$SEED1_CKPT" "/tmp/${SEED1_TAG}_watch.log"; then
    log "WARN: seed1 probes incomplete; continuing to seed2"
    FINAL_RC=1
  fi
  wait_for_gpus_idle "$RELEASE_TIMEOUT_SEC" \
    || die "GPUs busy after seed1 probes: $GPU_LIST"
  log "seed1 probes done and GPUs are idle"
}

start_seed2() {
  local rc=0
  local seed2_start seed2_deadline
  seed2_start=$(date +%s)
  seed2_deadline=$((seed2_start + SEED2_GATE_TIMEOUT_SEC))
  SEED2_LOG_START_LINE=0
  log "starting seed2 phase (seed=$SEED2_SEED workdir=$SEED2_WORK_DIR)"
  if reach_gate "seed2" "$SEED2_SESSION" "$SEED2_LOG" "$SEED2_WORK_DIR" "$SEED2_SEED" \
    "$seed2_start" "$seed2_deadline" 0 "$SEED2_LOG_START_LINE"; then
    log "seed2 ep${EPOCH} gate hit"
  else
    rc=$?
    log "FATAL: seed2 did not reach ep${EPOCH} gate (rc=$rc); stopping residual session"
    tmux send-keys -t "$SEED2_SESSION" C-c 2>/dev/null || true
    wait_for_training_stopped "$SEED2_SESSION" "$SEED2_WORK_DIR" "$SEED2_SEED" "$STOP_TIMEOUT_SEC" || true
    if session_exists "$SEED2_SESSION"; then
      tmux kill-session -t "$SEED2_SESSION" 2>/dev/null || true
    fi
    touch "/tmp/${SEED2_TAG}_failed"
    die "seed2 did not reach ep${EPOCH} gate (rc=$rc)"
  fi

  stop_session "seed2" "$SEED2_SESSION" "$SEED2_WORK_DIR" "$SEED2_SEED" \
    || die "seed2 did not stop cleanly"
  wait_for_gpus_idle "$RELEASE_TIMEOUT_SEC" \
    || die "GPUs busy after seed2 stop: $GPU_LIST"
  log "seed2 exited and GPUs are idle"

  if ! run_probes "$SEED2_TAG" "$SEED2_CKPT" "/tmp/${SEED2_TAG}_watch.log"; then
    log "WARN: seed2 probes incomplete"
    FINAL_RC=2
  fi
  wait_for_gpus_idle "$RELEASE_TIMEOUT_SEC" \
    || log "WARN: GPUs still busy after seed2 probes"
  touch "/tmp/${SEED2_TAG}_done"
  log "seed2 probes done"
}

GUARD=${SOURCE_ONLY:-0}
if [ "$GUARD" = "1" ]; then
  return 0 2>/dev/null || exit 0
fi

on_exit() {
  log "chain exit rc=$FINAL_RC"
}
trap on_exit EXIT

check_prerequisites
log "chain started (watch_start=$WATCH_START)"

if [ "$RUN_CHAIN" = "1" ]; then
  handoff_seed1
  start_seed2
  touch "/tmp/${SEED1_TAG}_done"
  log "chain finished rc=$FINAL_RC"
  exit "$FINAL_RC"
fi

log "dry-run: checking seed1 gate state"
if gate_hit_any_time "$SEED1_LOG" "$SEED1_CKPT"; then
  log "dry-run: seed1 ep${EPOCH} gate already present"
elif session_exists "$SEED1_SESSION"; then
  log "dry-run: seed1 session exists; chain is ready to wait"
else
  die "seed1 session is missing and the ep${EPOCH} gate is not present"
fi
log "dry-run complete: prerequisites OK"
