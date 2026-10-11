#!/usr/bin/env bash
# Idempotent ep6 finisher for the A2 arms.
#
# WHY THIS EXISTS
# ---------------
# The first revision of `watch_semA2_ep6.sh` (committed 2026-10-11 09:06) ran
# the A2-specific probes and then STOPPED: it never produced the four
# family-standard artifacts that `semA_ep6_verdict.py`/`semA_ep6_record.py`
# list as required evidence (three z-utilization windows + one
# posterior-source attribution), and it never invoked the verdict or the
# section-72 draft renderer. An arm that finished under that revision would
# have been reported as an INCOMPLETE bundle.
#
# `watch_semA2_ep6.sh` is now fixed, but the A2-S job was already running when
# the fix landed, so this script re-checks the bundle after the training stops
# and produces ONLY what is missing. Safe to run twice: every step is guarded
# by an existence check on its own output file.
#
# Read-only with respect to weights: it never writes a checkpoint and never
# launches training. It only runs CPU/GPU probes against epoch_6.pth.
#
# Usage:
#   ARM=A2S nohup setsid bash me_rssm/sanity/finish_semA2_ep6.sh &
set -u

cd /home/lurui/workspace/R4Det

ARM="${ARM:-A2S}"
case "$ARM" in
  A2S) CFG_DEFAULT=me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semA2S_24e.py ;;
  A2R) CFG_DEFAULT=me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semA2R_24e.py ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac

WORK_DIR="${WORK_DIR:-/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_sem${ARM}_seed0}"
CFG="${CFG:-$CFG_DEFAULT}"

# The training log is DERIVED from the run dir, not guessed from $ARM.
#
# The first revision defaulted to /tmp/sem${ARM}_24e.log, i.e.
# /tmp/semA2S_24e.log for A2S -- a path NOTHING creates (the job was launched
# with the log at /tmp/a2s_24e.log and the run's own log is
# $WORK_DIR/YYYYmmdd_HHMMSS.log). The wait loop below greps that path, so it
# could never be satisfied: the finisher would hang forever and silently never
# repair anything. The A generation never hit this because there
# `sem${ARM}_24e.log` happens to equal the real name; the A2 arms renamed the
# log, and the derivation was never re-checked.
#
# The run's own timestamped log is authoritative and always present once
# training starts, so prefer it; keep the /tmp path as a last-resort fallback
# for a bundle whose run dir was cleaned.
if [ -z "${LOG:-}" ]; then
  LOG=$(ls -1t "$WORK_DIR"/[0-9]*.log 2>/dev/null | head -n1)
  [ -n "$LOG" ] || LOG="/tmp/sem${ARM}_24e.log"
fi

# The A2 watchdog script is SHARED by both arms (watch_semA2_ep6.sh), so
# "watch_sem${ARM}_ep6.sh" would expand to watch_semA2S_ep6.sh and never match
# the running process. A dead guard meant the finisher would not wait for the
# watchdog battery to finish before starting its own probes on the same GPUs
# (the double-run that caused an earlier A2 OOM).
case "$ARM" in
  A2S|A2R) WATCHDOG_SCRIPT_NAME=watch_semA2_ep6.sh ;;
  *)       WATCHDOG_SCRIPT_NAME=watch_semA_ep6.sh ;;
esac

CKPT="$WORK_DIR/epoch_6.pth"
WATCH_LOG="$WORK_DIR/finish_sem${ARM}_ep6.log"

mkdir -p "$WORK_DIR"
echo "[finish-$ARM] $(date -u '+%F %T') armed" >> "$WATCH_LOG"

# 1) wait for the ep6 val + checkpoint, then for the WATCHDOG to exit (its own
#    probe battery may still be running; racing it on the same GPUs is how an
#    earlier A2 job hit OOM), then for the GPUs to drain.
while [ ! -f "$CKPT" ] || ! grep -q 'Epoch(val) \[6\]' "$LOG" 2>/dev/null; do
  sleep 30
done
echo "[finish-$ARM] $(date -u '+%F %T') ep6 present; waiting for the watchdog to exit" >> "$WATCH_LOG"
for _ in $(seq 1 480); do
  pgrep -f "$WATCHDOG_SCRIPT_NAME" >/dev/null 2>&1 || break
  sleep 15
done
if pgrep -f "$WATCHDOG_SCRIPT_NAME" >/dev/null 2>&1; then
  echo "[finish-$ARM] $(date -u '+%F %T') WARNING: watchdog still alive; continuing anyway (its outputs are existence-checked below)" >> "$WATCH_LOG"
fi
for _ in $(seq 1 240); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    -i 5,6,7 2>/dev/null | awk -F', ' '$2 > 2000 {n++} END {print n+0}')
  [ "$busy" = "0" ] && break
  sleep 15
done
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  -i 5,6,7 2>/dev/null | awk -F', ' '$2 > 2000 {n++} END {print n+0}')
if [ "$busy" != "0" ]; then
  echo "[finish-$ARM] $(date -u '+%F %T') ABORT: $busy/3 GPUs still busy" >> "$WATCH_LOG"
  exit 1
fi

# shellcheck disable=SC1091
source .envrc
if ! command -v python >/dev/null 2>&1 || ! python -c 'import torch, mmcv, mmdet3d' >/dev/null 2>&1; then
  echo "[finish-$ARM] $(date -u '+%F %T') ABORT: python/torch/mmcv/mmdet3d unavailable" >> "$WATCH_LOG"
  exit 1
fi

# 2) run only the missing family-standard artifacts. Same windows, offsets and
#    scoring as watch_semA_ep6.sh so the readings stay cross-generation
#    comparable; existence check makes a re-run a no-op.
if [ ! -f "$WORK_DIR/z_util_std_ep6_w3.json" ]; then
  CUDA_VISIBLE_DEVICES=6 python me_rssm/sanity/probe_z_utilization_standard.py \
    --config "$CFG" --checkpoint "$CKPT" \
    --start-index 3 --limit 4 --batch-size 4 --shuffle-offset 2 \
    --output "$WORK_DIR/z_util_std_ep6_w3.json" >> "$WATCH_LOG" 2>&1
fi
if [ ! -f "$WORK_DIR/z_util_std_ep6_w100.json" ]; then
  CUDA_VISIBLE_DEVICES=7 python me_rssm/sanity/probe_z_utilization_standard.py \
    --config "$CFG" --checkpoint "$CKPT" \
    --start-index 100 --limit 4 --batch-size 4 --shuffle-offset 2 \
    --output "$WORK_DIR/z_util_std_ep6_w100.json" >> "$WATCH_LOG" 2>&1
fi
if [ ! -f "$WORK_DIR/z_util_std_ep6_w1800.json" ]; then
  CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_z_utilization_standard.py \
    --config "$CFG" --checkpoint "$CKPT" \
    --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset -1 \
    --output "$WORK_DIR/z_util_std_ep6_w1800.json" >> "$WATCH_LOG" 2>&1
fi
if [ ! -f "$WORK_DIR/post_source_std_ep6_stride700.json" ]; then
  CUDA_VISIBLE_DEVICES=5 python tools/probe_posterior_source.py \
    --config "$CFG" --checkpoint "$CKPT" \
    --start-index 3 --limit 4 --stride 700 \
    --output "$WORK_DIR/post_source_std_ep6_stride700.json" >> "$WATCH_LOG" 2>&1
fi

# 3) freeze the verdict + the section-72 draft (always regenerate: they are
#    pure functions of the bundle and the later the better, once it is whole).
python me_rssm/sanity/semA_ep6_verdict.py --run "$WORK_DIR" >> "$WATCH_LOG" 2>&1
python me_rssm/sanity/semA_ep6_record.py --run "$WORK_DIR" \
  --output "$WORK_DIR/section72_draft.md" >> "$WATCH_LOG" 2>&1

echo "[finish-$ARM] $(date -u '+%F %T') done; decision remains manual" >> "$WATCH_LOG"
