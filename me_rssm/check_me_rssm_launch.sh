#!/usr/bin/env bash
# Health check for the auto-started ME-RSSM seed0 training.
# Run a few minutes after me_rssm_queue reports the launch:
#   bash me_rssm/check_me_rssm_launch.sh
# Verifies (research-package output; touches no baseline file):
#   H1 work_dir exists and a fresh log file appeared
#   H2 log shows Iter/Epoch progress (training loop actually running)
#   H3 no NaN/Inf in the log so far
#   H4 ME-specific stat_* channels are being logged (producers alive)
#   H5 all three ranks show progress lines (DDP healthy)

set -u
WD=/data/lurui/work_dirs/me_rssm_N4_2x4_24e_seed0
LOG=/data/lurui/work_dirs/me_rssm_queue.log

fail=0
say() { printf '%s\n' "$*"; }

if [ ! -d "$WD" ]; then
  say "[FAIL] H1 work_dir missing: $WD"
  tail -5 "$LOG" 2>/dev/null
  exit 1
fi
LATEST=$(ls -t "$WD"/*.log 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
  say "[FAIL] H1 no .log under $WD yet"
  tail -20 "$LOG" 2>/dev/null
  exit 1
fi
say "[PASS] H1 fresh log: $LATEST"

if grep -qE "Epoch \[[0-9]+\]\[[0-9]+/" "$LATEST"; then
  say "[PASS] H2 training loop running: $(grep -oE 'Epoch \[[0-9]+\]\[[0-9]+/1902\]' "$LATEST" | tail -1)"
else
  say "[WARN] H2 no Epoch progress line yet (may still be in init/iter warmup)"
fi

if grep -qiE "nan|inf" "$LATEST"; then
  say "[FAIL] H3 NaN/Inf found in log"
  grep -iE "nan|inf" "$LATEST" | head -3
  fail=1
else
  say "[PASS] H3 no NaN/Inf so far"
fi

if grep -q "stat_gain_mean" "$LATEST"; then
  say "[PASS] H4 ME stat channels present (modality bus consumed)"
else
  say "[FAIL] H4 stat_gain_mean not in log -- bus/producer chain broken?"
  fail=1
fi

NPROC=$(grep -c "Epoch \[[0-9]" "$LATEST")
say "[INFO] H5 progress lines seen: $NPROC (3-rank DDP merges into one log; check nvidia-smi for 3 busy GPUs)"

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | \
  awk -F, '$2+0 > 10000 {busy++} END {print "[INFO] GPUs busy >10GB: " busy+0 " (expect >=3 on 5,6,7)"}'

exit $fail
