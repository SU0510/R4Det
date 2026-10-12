#!/usr/bin/env bash
# Queue for F1 (reliability-gated front-end fusion), serial behind C1 -> A2-R.
# Pre-registration: docs/rssm_reliability_fusion_prereg_20261012.md
#
#   wait for the A2-R queue to finish -> drain GPUs 5-7 -> 450-iter smoke
#   -> S0 verdict (artifact-only) -> 24e run -> ep6 task-harm check
#   -> ep12-16 mature window report
#
# Stops (does NOT continue to the 24e run) if the smoke throws, if S0 fails, or
# if the ep6 early check is more than 2.0 points below B0's same window.
set -u
cd /home/lurui/workspace/R4Det
source .envrc

WD=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_relfusion_seed0
SMOKE_WD=$WD/smoke_450iter
CFG=me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_relfusion.py
SMOKE_CFG=me_rssm/sanity/relfus_smoke_450iter.py
B0=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed0
A2R_QLOG=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_semA2R_seed0/queue_semA2R_resume.log
LOG=/tmp/relfus_24e.log
SMOKE_LOG=/tmp/relfus_smoke.log
QLOG=$WD/queue_relfus.log
mkdir -p "$WD" "$SMOKE_WD"

say(){ echo "[relfus] $(date -u '+%F %T') $*" >> "$QLOG"; }
say "armed; waiting for the A2-R resume queue to finish"

# ---- 1) wait for the A2-R queue (log DONE + no job process) -----------------
while :; do
  if grep -q 'DONE' "$A2R_QLOG" 2>/dev/null && ! pgrep -f '[s]emA2R_24e' >/dev/null 2>&1; then
    break
  fi
  sleep 60
done
say "A2-R queue finished"

# ---- 2) drain GPUs 5-7 ------------------------------------------------------
used=99999
for _ in $(seq 1 240); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5,6,7 \
         | tr -d ' ' | paste -sd+ | bc)
  [ "${used:-99999}" -lt 3000 ] && break
  sleep 60
done
say "gpu used=${used} MiB"

# ---- 3) 450-iter wiring smoke ----------------------------------------------
if [ ! -f "$SMOKE_WD/iter_450.pth" ]; then
  setsid bash -c "source .envrc; CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh $SMOKE_CFG 3 --seed 0 --deterministic --work-dir $SMOKE_WD" < /dev/null > "$SMOKE_LOG" 2>&1 &
  say "smoke launched pid $!"
  for _ in $(seq 1 120); do
    [ -f "$SMOKE_WD/iter_450.pth" ] && break
    grep -qi 'CAUGHT AN EXCEPTION' "$SMOKE_LOG" 2>/dev/null && { say "SMOKE EXCEPTION"; break; }
    sleep 60
  done
  sleep 60
else
  say "smoke checkpoint already present"
fi

# ---- 4) S0 verdict ----------------------------------------------------------
if python me_rssm/sanity/relfus_s0_verdict.py --work-dir "$SMOKE_WD" \
     --out "$SMOKE_WD/s0_verdict.json" >> "$QLOG" 2>&1; then
  say "S0_ALL_PASS -> launching the 24e run"
else
  say "S0 FAILED -> NOT launching the 24e run (see $SMOKE_WD/s0_verdict.json)"
  exit 1
fi

# ---- 5) 24e run -------------------------------------------------------------
setsid bash -c "source .envrc; CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh $CFG 3 --seed 0 --deterministic --work-dir $WD" < /dev/null > "$LOG" 2>&1 &
say "24e launched pid $! at $(date -u '+%F %T')"

# ---- 6) ep6 early task-harm check ------------------------------------------
for _ in $(seq 1 720); do
  grep -q 'Epoch(val) \[6\]' "$LOG" 2>/dev/null && break
  grep -qi 'CAUGHT AN EXCEPTION' "$LOG" 2>/dev/null && { say "24e EXCEPTION"; break; }
  sleep 60
done
python me_rssm/sanity/c1valid_curve_check.py --log "$WD" --work-dir "$WD" \
  --b0-log "$B0" --window 1-6 --out "$WD/relfus_ep6_early_check.json" >> "$QLOG" 2>&1
if python - <<'PY' >> "$QLOG" 2>&1
import json, sys
d = json.load(open('/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_relfusion_seed0/relfus_ep6_early_check.json'))
sys.exit(0 if d.get('early_check', {}).get('pass') else 1)
PY
then say "ep6 early check PASS -> continuing"; else say "ep6 early check FAIL -> flagged for review"; fi

# ---- 7) ep12-16 mature window ----------------------------------------------
for _ in $(seq 1 2880); do
  grep -q 'Epoch(val) \[16\]' "$LOG" 2>/dev/null && break
  sleep 60
done
python me_rssm/sanity/c1valid_curve_check.py --log "$WD" --work-dir "$WD" \
  --b0-log "$B0" --window 12-16 --out "$WD/relfus_ep16_window_check.json" >> "$QLOG" 2>&1
say "report -> $WD/relfus_ep16_window_check.json ; DONE (decisions remain manual)"
