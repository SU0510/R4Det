#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TMP=$(mktemp -d /tmp/fgfull18_watch_rehearsal.XXXXXX)
trap 'rm -rf "$TMP"' EXIT
source "$ROOT/.envrc" >/dev/null 2>&1
SHIM="$TMP/bin"; mkdir -p "$SHIM" "$TMP/upstream_work" "$TMP/run_work" "$TMP/tmux_state"

write_checkpoint() {
  python - "$1" <<'PY'
import sys
import torch

torch.save(
    {
        'meta': {'epoch': 1, 'iter': 1},
        'state_dict': {'weight': torch.zeros(1)},
    },
    sys.argv[1],
)
PY
}

cat > "$SHIM/tmux" <<'SH'
#!/usr/bin/env bash
set -u
state=${REHEARSE_TMUX_STATE:?}
cmd=${1:-}
case "$cmd" in
  has-session) [ -f "$state/session" ] ;;
  new-session)
    shift
    # The launch command is always the final argv element.  Capture it
    # before consuming option/value pairs; otherwise the generic `*)` case
    # drops it and `${!#}` falls back to the shell's own $0.
    launch=${!#}
    session=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        -s) session=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    [ -n "$session" ] || exit 1
    printf '%s\n' "$session" > "$state/session"
    # The watcher's tmux string is the final argv element.  Keep this shim
    # eval-safe by refusing anything unexpected.
    # Match `tmux new-session -d`: detach from this shell's stdout/stderr so
    # the watcher's command substitution cannot be held open by the child.
    eval "$launch" </dev/null >/dev/null 2>&1 &
    child=$!
    ( wait "$child"; rm -f "$state/session" ) </dev/null >/dev/null 2>&1 &
    printf '%s\n' "$child" > "$state/pid"
    ;;
  send-keys) ;;
  kill-session)
    [ -f "$state/pid" ] && kill "$(cat "$state/pid")" 2>/dev/null || true
    rm -f "$state/session"
    ;;
esac
SH
chmod +x "$SHIM/tmux"

cat > "$TMP/fake_train.sh" <<'SH'
#!/usr/bin/env bash
set -u
work_dir="" seed=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --work-dir) work_dir=$2; shift 2 ;;
    --seed) seed=$2; shift 2 ;;
    *) shift ;;
  esac
done
[ -n "$work_dir" ] || exit 2
mkdir -p "$work_dir"
printf 'launch seed=%s work=%s\n' "$seed" "$work_dir" >> "$REHEARSE_RUN_LOG"
if [ ! -f "$REHEARSE_CRASH_DONE" ]; then
  touch "$REHEARSE_CRASH_DONE"
  exit 7
fi
printf 'Epoch [18][1902/1902]\nEpoch(val) [18][170]\n' >> "$work_dir/../run.log"
python - "$work_dir/epoch_18.pth" <<'PY'
import sys
import torch

torch.save(
    {
        'meta': {'epoch': 18, 'iter': 34236},
        'state_dict': {'weight': torch.zeros(1)},
    },
    sys.argv[1],
)
PY
SH
chmod +x "$TMP/fake_train.sh"

cat > "$TMP/fake_gpu.sh" <<'SH'
#!/usr/bin/env bash
printf '5, 10\n6, 10\n7, 10\n'
exit 0
SH
chmod +x "$TMP/fake_gpu.sh"
chmod +x "$TMP/fake_gpu.sh"

export PATH="$SHIM:$PATH"
export REHEARSE_TMUX_STATE="$TMP/tmux_state"
export REHEARSE_RUN_LOG="$TMP/run_invocations.log"
export REHEARSE_CRASH_DONE="$TMP/crash_done"

export UPSTREAM_DONE="$TMP/upstream_done"
export UPSTREAM_LOG="$TMP/upstream.log"
export UPSTREAM_WORK_DIR="$TMP/upstream_work"
export UPSTREAM_PROBE_DIR="$TMP/upstream_probes"
mkdir -p "$UPSTREAM_PROBE_DIR"
export UPSTREAM_SESSION=rehearse_upstream
export UPSTREAM_SEED=2
export SESSION=rehearse_run
export LOG="$TMP/run.log"
export WORK_DIR="$TMP/run_work"
export WATCH_LOG="$TMP/watch.log"
export DONE_MARKER="$TMP/done"
export FAILED_MARKER="$TMP/failed"
export RUN_WATCH=1
export POLL_SEC=1
export UPSTREAM_TIMEOUT_SEC=30
export GPU_RELEASE_TIMEOUT_SEC=10
export RUN_TIMEOUT_SEC=30
export MAX_ATTEMPTS=3
export RESTART_DELAY_SEC=1
export CKPT_MIN_BYTES=1000
export CKPT_SETTLE_SEC=0
export GPU_QUERY_CMD="$TMP/fake_gpu.sh"
export TRAIN_CMD_OVERRIDE="bash $TMP/fake_train.sh configs/r4det/fake.py 3 --seed=0 --deterministic"

# 1. An old marker without intact upstream evidence must be refused.
touch "$UPSTREAM_DONE"
sleep 1
if "$ROOT/me_rssm/sanity/watch_after_mai_seed2_run_fgfull18_seed0.sh"; then
  echo 'FAIL unsupported stale marker accepted'; exit 1
fi
grep -q 'stale upstream completion marker without intact' "$WATCH_LOG" \
  || { echo 'FAIL unsupported stale marker not logged'; cat "$WATCH_LOG"; exit 1; }

# 2. An old marker with intact evidence must be taken over after a watcher
# restart, without relaunching the upstream seed2 job.
printf 'Epoch(val) [16][170]\n' > "$UPSTREAM_LOG"
write_checkpoint "$UPSTREAM_WORK_DIR/epoch_16.pth"
for probe in "$UPSTREAM_PROBE_DIR"/*.json; do
  [ -e "$probe" ] || continue
  printf '{"ok": true}\n' > "$probe"
done
for probe in \
  "$UPSTREAM_PROBE_DIR/z_util_mai_seed2_ep16_w3.json" \
  "$UPSTREAM_PROBE_DIR/z_util_mai_seed2_ep16_w100.json" \
  "$UPSTREAM_PROBE_DIR/z_util_mai_seed2_ep16_w1800.json" \
  "$UPSTREAM_PROBE_DIR/post_source_mai_seed2_ep16_stride700.json"; do
  printf '{"ok": true}\n' > "$probe"
done

rm -f "$WATCH_LOG" "$FAILED_MARKER" "$DONE_MARKER" "$REHEARSE_RUN_LOG" "$REHEARSE_TMUX_STATE/session"

# 3. A missing probe output must block the hand-off even when done exists.
mv "$UPSTREAM_PROBE_DIR/z_util_mai_seed2_ep16_w100.json" "$TMP/probe.hidden"
if "$ROOT/me_rssm/sanity/watch_after_mai_seed2_run_fgfull18_seed0.sh"; then
  echo 'FAIL missing probe accepted'; exit 1
fi
grep -q 'stale upstream completion marker without intact' "$WATCH_LOG" \
  || { echo 'FAIL missing probe not logged'; cat "$WATCH_LOG"; exit 1; }
mv "$TMP/probe.hidden" "$UPSTREAM_PROBE_DIR/z_util_mai_seed2_ep16_w100.json"
rm -f "$WATCH_LOG" "$FAILED_MARKER" "$DONE_MARKER" "$REHEARSE_RUN_LOG"

# 4. Full hand-off with a transient launch failure on the first attempt;
# watcher retries from scratch and completes ep18.
if ! "$ROOT/me_rssm/sanity/watch_after_mai_seed2_run_fgfull18_seed0.sh"; then
  echo 'FAIL watcher nonzero'; cat "$WATCH_LOG"; exit 1
fi
[ -f "$DONE_MARKER" ] || { echo 'FAIL done missing'; cat "$WATCH_LOG"; exit 1; }
[ -f "$WORK_DIR/epoch_18.pth" ] || { echo 'FAIL ckpt missing'; cat "$WATCH_LOG"; exit 1; }
grep -q 'launch failed on attempt 1' "$WATCH_LOG" \
  || { echo 'FAIL first-attempt launch failure not reported'; cat "$WATCH_LOG"; exit 1; }
grep -q 'launch failed on attempt 1' "$WATCH_LOG" \
  || { echo 'FAIL first-attempt crash not reported'; cat "$WATCH_LOG"; exit 1; }
grep -q 'complete: ep18 run finished' "$WATCH_LOG" || { echo 'FAIL completion missing'; cat "$WATCH_LOG"; exit 1; }
echo '[PASS] isolated rehearsal: unsupported stale rejected, evidence takeover verified, missing probe blocked, crash resume verified, ep18 completion verified'
