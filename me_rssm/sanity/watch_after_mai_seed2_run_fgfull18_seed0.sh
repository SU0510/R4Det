#!/usr/bin/env bash
# Run the complete 18-epoch MotionAlignedRSSMFusion baseline after the MAI
# seed2 replay has fully finished.
#
# Contract:
#   seed2 training/session exits
#   -> seed2 probes are written and /tmp/mai_seed2_ep16_done exists
#   -> GPU 5/6/7 are idle
#   -> this script starts the 18e run with seed 0.
#
# The 18e run uses its own config with max_epochs=18, an 18e cosine horizon
# and a 0->8 epoch KL warmup.  It therefore exits by itself at ep18; there is
# no ep18 SIGINT gate and no hand-off race.
#
# Default mode is a dry run.  Set RUN_WATCH=1 to execute.
#
# Production launch:
#   tmux new-session -d -s fgfull18_seed0_watch \
#     'cd /home/lurui/workspace/R4Det && RUN_WATCH=1 \
#      bash me_rssm/sanity/watch_after_mai_seed2_run_fgfull18_seed0.sh'
set -u

cd /home/lurui/workspace/R4Det

# The watcher uses the project's Python environment for checkpoint and JSON
# validation.  Source it in the parent shell as well as inside config audits.
if [ -f .envrc ]; then
  source .envrc >/dev/null 2>&1 || true
fi

RUN_WATCH=${RUN_WATCH:-0}
GPU_LIST=${GPU_LIST:-5,6,7}
GPU_BUSY_MIB=${GPU_BUSY_MIB:-500}

UPSTREAM_SESSION=${UPSTREAM_SESSION:-mai_seed2_ep16}
UPSTREAM_LOG=${UPSTREAM_LOG:-/tmp/mai_seed2_ep16.log}
UPSTREAM_DONE=${UPSTREAM_DONE:-/tmp/mai_seed2_ep16_done}
UPSTREAM_WORK_DIR=${UPSTREAM_WORK_DIR:-/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mai_2x4_24e_seed2}
UPSTREAM_SEED=${UPSTREAM_SEED:-2}
UPSTREAM_EPOCH=${UPSTREAM_EPOCH:-16}
UPSTREAM_PROBE_DIR=${UPSTREAM_PROBE_DIR:-/tmp}
UPSTREAM_PROBES=(
  "$UPSTREAM_PROBE_DIR/z_util_mai_seed2_ep16_w3.json"
  "$UPSTREAM_PROBE_DIR/z_util_mai_seed2_ep16_w100.json"
  "$UPSTREAM_PROBE_DIR/z_util_mai_seed2_ep16_w1800.json"
  "$UPSTREAM_PROBE_DIR/post_source_mai_seed2_ep16_stride700.json"
)

CFG=${CFG:-configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_18e_pretrained_v2_head_no2d_igdr.py}
SESSION=${SESSION:-fgfull18_seed0_ep18}
LOG=${LOG:-/tmp/fgfull18_seed0_ep18.log}
WORK_DIR=${WORK_DIR:-/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_18e_seed0}
SEED=${SEED:-0}
EXPECTED_FUSION=${EXPECTED_FUSION:-MotionAlignedRSSMFusion}
EXPECTED_MAX_EPOCHS=${EXPECTED_MAX_EPOCHS:-18}
EXPECTED_KL_WARMUP_END=${EXPECTED_KL_WARMUP_END:-8}

WATCH_LOG=${WATCH_LOG:-/tmp/fgfull18_seed0_ep18_watch.log}
DONE_MARKER=${DONE_MARKER:-/tmp/fgfull18_seed0_ep18_done}
FAILED_MARKER=${FAILED_MARKER:-/tmp/fgfull18_seed0_ep18_failed}
POLL_SEC=${POLL_SEC:-30}
# Upstream seed2 still has several epochs to run when this watcher starts;
# do not reuse the short GPU-release timeout for the upstream wait.
UPSTREAM_TIMEOUT_SEC=${UPSTREAM_TIMEOUT_SEC:-129600}   # 36 h
GPU_RELEASE_TIMEOUT_SEC=${GPU_RELEASE_TIMEOUT_SEC:-1800}
RUN_TIMEOUT_SEC=${RUN_TIMEOUT_SEC:-172800}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
RESTART_DELAY_SEC=${RESTART_DELAY_SEC:-60}
CKPT_MIN_BYTES=${CKPT_MIN_BYTES:-100000000}
CKPT_SETTLE_SEC=${CKPT_SETTLE_SEC:-20}
ALLOW_EXISTING_WORKDIR=${ALLOW_EXISTING_WORKDIR:-0}
TRAIN_ENTRY=${TRAIN_ENTRY:-tools/dist_train.sh}
GPU_QUERY_CMD=${GPU_QUERY_CMD:-nvidia-smi}
# Rehearsal-only override: when set, this exact command is launched instead of
# building the production dist_train command.  Production leaves it empty.
TRAIN_CMD_OVERRIDE=${TRAIN_CMD_OVERRIDE:-}

WATCH_START=$(date +%s)
FINAL_RC=0

mkdir -p "$(dirname "$WATCH_LOG")"
touch "$WATCH_LOG"

log() {
  echo "[watch] $(date -u '+%F %T') $*" >> "$WATCH_LOG"
}

die() {
  FINAL_RC=1
  log "FATAL: $*"
  touch "$FAILED_MARKER"
  exit 1
}

session_exists() {
  tmux has-session -t "$1" 2>/dev/null
}

training_process_running() {
  local session=$1 work_dir=$2 seed=$3
  if session_exists "$session" \
    && ps -eo cmd \
      | grep -F -- "$work_dir" \
      | grep -E -- "--seed[ =]$seed( |$)" \
      | grep -v grep >/dev/null; then
    return 0
  fi
  # Also catch a detached DDP process whose tmux session is already gone.
  ps -eo cmd \
    | grep -F -- "$work_dir" \
    | grep -E -- "--seed[ =]$seed( |$)" \
    | grep -v grep >/dev/null
}

gpu_busy_count() {
  local out
  out=$($GPU_QUERY_CMD --query-gpu=memory.used --format=csv,noheader,nounits \
    -i "$GPU_LIST" 2>/dev/null) || return 9
  [ -n "$out" ] || return 9
  # Accept both real nvidia-smi output ("5, 10") and the rehearsal shim's
  # query-independent numeric lines ("10").  The memory value is the final
  # comma-separated field in both forms.
  printf '%s\n' "$out" \
    | awk -F',' -v limit="$GPU_BUSY_MIB" \
        '$NF+0 > limit {n++} END {print n+0}'
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

latest_checkpoint() {
  local work_dir=$1 f size mtime now
  now=$(date +%s)
  [ -d "$work_dir" ] || return 0
  for f in "$work_dir"/epoch_*.pth; do
    [ -f "$f" ] || continue
    size=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$size" -ge "$CKPT_MIN_BYTES" ] || continue
    mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    [ $((now - mtime)) -ge "$CKPT_SETTLE_SEC" ] || continue
    checkpoint_valid "$f" || continue
    printf '%s\n' "$f"
  done | sort -V | tail -1
}

checkpoint_valid() {
  local checkpoint=$1
  # Verify the checkpoint the same way training will consume it.  mmcv
  # writes torch.save archives whose pickle lives under archive/data.pkl,
  # and older torch formats may not be zip archives at all, so a root-level
  # zipfile name check is both false-negative and too weak.
  python - "$checkpoint" <<'PY' >/dev/null 2>&1
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location='cpu')
assert isinstance(checkpoint, dict)
assert 'state_dict' in checkpoint
assert 'meta' in checkpoint
assert checkpoint['meta'].get('epoch') is not None
PY
}

json_nonempty() {
  local path=$1
  [ -f "$path" ] || return 1
  [ "$(stat -c %s "$path" 2>/dev/null || echo 0)" -gt 0 ] || return 1
  python - "$path" <<'PY' >/dev/null 2>&1
import json
import sys

with open(sys.argv[1]) as handle:
    value = json.load(handle)
assert isinstance(value, dict)
assert value
PY
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

stop_own_session() {
  if wait_for_training_stopped "$SESSION" "$WORK_DIR" "$SEED" "$GPU_RELEASE_TIMEOUT_SEC"; then
    log "training process exited"
  else
    log "WARN: training process did not disappear within ${GPU_RELEASE_TIMEOUT_SEC}s"
    tmux send-keys -t "$SESSION" C-c 2>/dev/null || true
    wait_for_training_stopped "$SESSION" "$WORK_DIR" "$SEED" "$GPU_RELEASE_TIMEOUT_SEC" || true
  fi
  if session_exists "$SESSION"; then
    log "killing lingering tmux session $SESSION"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
  fi
}

audit_config() {
  source .envrc >/dev/null 2>&1 || return 9
  python - "$CFG" <<'PY' 2>/dev/null
import sys
from mmcv import Config
c = Config.fromfile(sys.argv[1])
print(c.model.temporal_fusion.type)
print(c.runner.max_epochs)
print(c.get('custom_hooks')[0].get('end_epoch'))
print(c.optimizer.lr)
PY
}

upstream_complete_evidence() {
  local probe
  [ -f "$UPSTREAM_LOG" ] || return 1
  grep -q "Epoch(val) \[$UPSTREAM_EPOCH\]" "$UPSTREAM_LOG" 2>/dev/null || return 1
  [ -f "$UPSTREAM_WORK_DIR/epoch_${UPSTREAM_EPOCH}.pth" ] || return 1
  checkpoint_valid "$UPSTREAM_WORK_DIR/epoch_${UPSTREAM_EPOCH}.pth" || return 1
  for probe in "${UPSTREAM_PROBES[@]}"; do
    json_nonempty "$probe" || return 1
  done
  ! training_process_running "$UPSTREAM_SESSION" "$UPSTREAM_WORK_DIR" "$UPSTREAM_SEED" || return 1
  ! session_exists "$UPSTREAM_SESSION" || return 1
  return 0
}

wait_for_upstream_exit() {
  local timeout=$1 deadline
  deadline=$(( $(date +%s) + timeout ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if ! training_process_running "$UPSTREAM_SESSION" "$UPSTREAM_WORK_DIR" "$UPSTREAM_SEED" \
      && ! session_exists "$UPSTREAM_SESSION"; then
      return 0
    fi
    sleep "$POLL_SEC"
  done
  return 1
}

wait_for_upstream() {
  local deadline=$(( $(date +%s) + UPSTREAM_TIMEOUT_SEC ))
  log "waiting for upstream MAI seed2 hand-off: session=$UPSTREAM_SESSION done=$UPSTREAM_DONE"

  while [ "$(date +%s)" -lt "$deadline" ]; do
    if [ ! -f "$UPSTREAM_DONE" ]; then
      sleep "$POLL_SEC"
      continue
    fi
    # A marker older than this watcher can still be valid when the watcher
    # itself was restarted after seed2 completed.  In that case accept it
    # only if the full upstream evidence is intact and no seed2 process or
    # session remains; otherwise refuse rather than risk a double launch.
    marker_mtime=$(stat -c %Y "$UPSTREAM_DONE" 2>/dev/null || echo 0)
    if [ "$marker_mtime" -lt "$WATCH_START" ]; then
      log "upstream done marker predates this watch start; validating takeover evidence"
      if ! wait_for_upstream_exit "$GPU_RELEASE_TIMEOUT_SEC"; then
        die "upstream completion marker is old but seed2 process/session is still alive"
      fi
      if ! upstream_complete_evidence; then
        die "stale upstream completion marker without intact hand-off evidence: $UPSTREAM_DONE"
      fi
      log "owning pre-existing upstream completion marker after evidence check"
      return 0
    fi
    if training_process_running "$UPSTREAM_SESSION" "$UPSTREAM_WORK_DIR" "$UPSTREAM_SEED"; then
      sleep "$POLL_SEC"
      continue
    fi
    if session_exists "$UPSTREAM_SESSION"; then
      log "upstream tmux session still exists; waiting for torch.distributed to exit"
      sleep "$POLL_SEC"
      continue
    fi
    if ! grep -q "Epoch(val) \[$UPSTREAM_EPOCH\]" "$UPSTREAM_LOG" 2>/dev/null; then
      die "upstream marker exists but ep${UPSTREAM_EPOCH} validation line is missing"
    fi
    if ! [ -f "$UPSTREAM_WORK_DIR/epoch_${UPSTREAM_EPOCH}.pth" ]; then
      die "upstream marker exists but epoch_${UPSTREAM_EPOCH}.pth is missing"
    fi
    for probe in "${UPSTREAM_PROBES[@]}"; do
      if ! json_nonempty "$probe"; then
        die "upstream marker exists but probe output is missing/empty: $probe"
      fi
    done
    if ! wait_for_gpus_idle "$GPU_RELEASE_TIMEOUT_SEC"; then
      die "GPUs are still busy after upstream completion"
    fi
    log "upstream MAI seed2 is fully finished and GPUs are idle"
    return 0
  done
  die "timed out waiting for upstream MAI seed2 completion"
}

prepare_workdir() {
  local resume_from=$1
  mkdir -p "$WORK_DIR"
  if [ -n "$resume_from" ]; then
    log "resuming from $resume_from"
  else
    log "no checkpoint available; starting from scratch"
  fi
}

spawn_run() {
  local resume_from=$1 append=$2 cmd deadline
  mkdir -p "$WORK_DIR"
  if [ -n "$TRAIN_CMD_OVERRIDE" ]; then
    cmd="$TRAIN_CMD_OVERRIDE"
    [ -n "$resume_from" ] && cmd="$cmd --resume-from $resume_from"
    cmd="$cmd --work-dir $WORK_DIR"
  else
    cmd="cd /home/lurui/workspace/R4Det && source .envrc && CUDA_VISIBLE_DEVICES=$GPU_LIST SEED=$SEED bash $TRAIN_ENTRY $CFG 3 --seed $SEED --deterministic"
    [ -n "$resume_from" ] && cmd="$cmd --resume-from $resume_from"
    cmd="$cmd --work-dir $WORK_DIR"
  fi
  if [ "$append" = "1" ]; then redir=">>"; else redir=">"; fi
  cmd="$cmd $redir $LOG 2>&1"
  log "launch: $cmd"
  tmux new-session -d -s "$SESSION" "$cmd" || return 1
  deadline=$(( $(date +%s) + 180 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    training_process_running "$SESSION" "$WORK_DIR" "$SEED" && return 0
    session_exists "$SESSION" || return 1
    sleep 5
  done
  return 1
}

wait_for_run() {
  local attempt=$1 deadline=$(( $(date +%s) + RUN_TIMEOUT_SEC )) ckpt="$WORK_DIR/epoch_18.pth"
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if grep -q "Epoch(val) \[18\]" "$LOG" 2>/dev/null \
      && [ -f "$ckpt" ] \
      && checkpoint_valid "$ckpt"; then
      log "attempt $attempt reached ep18 val + checkpoint"
      return 0
    fi
    if ! training_process_running "$SESSION" "$WORK_DIR" "$SEED"; then
      log "attempt $attempt training process exited before ep18 completion"
      return 1
    fi
    sleep "$POLL_SEC"
  done
  log "attempt $attempt timed out before ep18 completion"
  return 1
}

check_prerequisites() {
  [ "$RUN_WATCH" = "1" ] || log "dry-run: RUN_WATCH is not 1; no launch will be performed"
  [ -f "$CFG" ] || die "missing config: $CFG"
  [ -f "$TRAIN_ENTRY" ] || die "missing training entry: $TRAIN_ENTRY"
  [ -f checkpoints/pretrained_tj4d.pth ] || die "missing pretrained checkpoint"
  if session_exists "$SESSION"; then
    die "target tmux session already exists: $SESSION"
  fi
  if [ "$RUN_WATCH" = "1" ] && [ -d "$WORK_DIR" ] && [ -n "$(ls -A "$WORK_DIR" 2>/dev/null)" ]; then
    if [ "$ALLOW_EXISTING_WORKDIR" != "1" ]; then
      die "target workdir already exists and is not empty (set ALLOW_EXISTING_WORKDIR=1 to resume): $WORK_DIR"
    fi
    log "existing target workdir allowed: $WORK_DIR"
  fi
  audit=$(audit_config) || die "could not load/audit config: $CFG"
  fusion=$(printf '%s\n' "$audit" | sed -n '1p')
  maxe=$(printf '%s\n' "$audit" | sed -n '2p')
  klend=$(printf '%s\n' "$audit" | sed -n '3p')
  lr=$(printf '%s\n' "$audit" | sed -n '4p')
  log "config audit: fusion=$fusion max_epochs=$maxe kl_warmup_end=$klend lr=$lr"
  [ "$fusion" = "$EXPECTED_FUSION" ] || die "unexpected fusion type: $fusion"
  [ "$maxe" = "$EXPECTED_MAX_EPOCHS" ] || die "unexpected max_epochs: $maxe"
  [ "$klend" = "$EXPECTED_KL_WARMUP_END" ] || die "unexpected KL warmup end: $klend"
}

on_exit() {
  log "watch exit rc=$FINAL_RC"
}
trap on_exit EXIT

check_prerequisites

if [ "$RUN_WATCH" != "1" ]; then
  if [ -f "$UPSTREAM_DONE" ]; then
    log "dry-run: upstream completion marker exists: $UPSTREAM_DONE"
  else
    log "dry-run: upstream completion marker not present yet"
  fi
  log "dry-run complete: prerequisites OK"
  exit 0
fi

rm -f "$DONE_MARKER" "$FAILED_MARKER"
wait_for_upstream

attempt=0
while true; do
  attempt=$((attempt + 1))
  resume_from=$(latest_checkpoint "$WORK_DIR")
  prepare_workdir "$resume_from"
  if ! wait_for_gpus_idle "$GPU_RELEASE_TIMEOUT_SEC"; then
    die "GPUs are busy before attempt $attempt"
  fi
  if ! spawn_run "$resume_from" "$([ "$attempt" -eq 1 ] && echo 0 || echo 1)"; then
    log "launch failed on attempt $attempt"
    [ "$attempt" -ge "$MAX_ATTEMPTS" ] && die "training failed to launch after $MAX_ATTEMPTS attempts"
    sleep "$RESTART_DELAY_SEC"
    continue
  fi
  if wait_for_run "$attempt"; then
    break
  fi
  if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
    die "training did not complete ep18 after $MAX_ATTEMPTS attempts"
  fi
  if session_exists "$SESSION"; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
  fi
  sleep "$RESTART_DELAY_SEC"
done

# The config exits naturally at ep18; wait for torch.distributed and the tmux
# session to finish, then wait for the GPUs before declaring completion.
stop_own_session
if ! wait_for_gpus_idle "$GPU_RELEASE_TIMEOUT_SEC"; then
  die "GPUs remained busy after the ep18 run"
fi
touch "$DONE_MARKER"
log "complete: ep18 run finished, GPUs idle, marker=$DONE_MARKER"
