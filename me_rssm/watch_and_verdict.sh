#!/usr/bin/env bash
# Watcher: capture the CycCls three-seed verdict numbers automatically
# once seed2's training exits (research-package output; touches no
# baseline file). The JUDGMENT (writing conclusions into
# docs/training_runs_full.md) stays a human/agent decision; this script
# only freezes the numbers so nothing is lost between sessions.
#
# Log: /data/lurui/work_dirs/cyccls_verdict_watch.log

set -u
cd /home/lurui/workspace/R4Det

LOG=/data/lurui/work_dirs/cyccls_verdict_watch.log
OUT=/data/lurui/work_dirs/cyccls_verdict_ep12_16.txt
PY=/home/lurui/envs/miniforge3/envs/r4det/bin/python

echo "[watch] $(date '+%F %T') waiting for seed2 training to exit ..." >> "$LOG"
while pgrep -f "train_vod.py --config" > /dev/null 2>&1; do
  sleep 300
done
echo "[watch] $(date '+%F %T') training gone; computing verdict numbers" >> "$LOG"

{
  echo "=== captured $(date -u '+%F %T') UTC (after training exit) ==="
  "$PY" tools/verdict_window.py \
    --base work_dirs/run10_headv2_multiseed \
    --cand work_dirs/cyccls_branch_N4_2x4_24e_multiseed \
    --window 12 16
  echo
  echo "=== per-seed BEST (max Overall_3D_moderate, all available epochs) ==="
  for d in /data/lurui/work_dirs/cyccls_branch_N4_2x4_24e_multiseed/seed_*; do
    s=$(basename "$d")
    "$PY" tools/summarize_run.py "$d" --tail 5 | sed "s/^/[$s] /"
  done
} >> "$OUT" 2>&1

echo "[watch] $(date '+%F %T') verdict numbers saved -> $OUT" >> "$LOG"
