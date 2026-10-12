#!/usr/bin/env bash
# Queue (user-approved 2026-10-12): after the C1 run exits, resume the A2-R arm
# from its own epoch_6.pth so the frozen ep12-16 window (n=5) can be evaluated.
#
# Why resume instead of rebuild: epoch_6.pth already exists (docs section 81.5),
# so ep6->ep16 costs 10 epochs (~10 h) instead of 16 (~16 h). The 24e config is
# kept so the CosineAnnealing LR schedule and the epoch counter stay continuous
# with the arm's own ep1-6 run; the job is stopped right after ep16's val.
#
# GPU policy: serially after C1 (only GPUs 5,6,7 are ours and C1 holds all three).
set -u
cd /home/lurui/workspace/R4Det
source .envrc

WD=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_semA2R_seed0
CFG=me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semA2R_24e.py
RESUME=$WD/epoch_6.pth
LOG=/tmp/semA2R_resume_16e.log
QLOG=$WD/queue_semA2R_resume.log
B0=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed0
C1WD=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_c1valid_seed0

say(){ echo "[queue] $(date -u '+%F %T') $*" >> "$QLOG"; }
say "armed; waiting for the C1 run to exit"

# ---- 1) wait for C1 training ------------------------------------------------
while pgrep -f '[n]o2d_igdr_c1valid' >/dev/null 2>&1; do sleep 60; done
say "C1 gone"

# ---- 2) let GPUs 5-7 drain --------------------------------------------------
used=99999
for _ in $(seq 1 240); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5,6,7 \
         | tr -d ' ' | paste -sd+ | bc)
  [ "${used:-99999}" -lt 3000 ] && break
  sleep 60
done
say "gpu used=${used} MiB"

# ---- 3) safety net: finish C1's ep14 products if the in-training probe OOMed -
if [ ! -f "$C1WD/c1valid_ep14_mechanism_verdict.json" ] && \
   [ -f "$C1WD/epoch_14.pth" ] && \
   ! pgrep -f '[w]atch_c1valid_ep6' >/dev/null 2>&1; then
  say "C1 ep14 products missing and no watchdog alive -> re-running watchdog"
  timeout 7200 bash me_rssm/sanity/watch_c1valid_ep6.sh >> "$QLOG" 2>&1
  say "watchdog re-run finished"
else
  say "C1 ep14 products present or watchdog still alive -> skip safety net"
fi

# ---- 4) pre-flight (hard-problem class: interface / wiring drift) -----------
[ -f "$RESUME" ] || { say "STOP: $RESUME missing"; exit 1; }
if [ -f "$WD/epoch_16.pth" ]; then say "STOP: epoch_16.pth exists (idempotent)"; exit 0; fi
ftype=$(python -c "from mmcv import Config; print(Config.fromfile('$CFG').model.temporal_fusion.type)" 2>>"$QLOG")
if [ "$ftype" != "SemanticReadoutRSSMFusion" ]; then
  say "STOP: fusion type is '$ftype' (expected SemanticReadoutRSSMFusion)"; exit 1
fi
say "pre-flight ok (fusion=$ftype, resume=$RESUME)"

# ---- 5) launch the resume ---------------------------------------------------
setsid bash -c "source .envrc; CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh $CFG 3 --seed 0 --deterministic --resume-from $RESUME --work-dir $WD" < /dev/null > "$LOG" 2>&1 &
say "resume launched pid $! at $(date -u '+%F %T')"

# ---- 6) wait for ep16 val, then stop the job --------------------------------
for _ in $(seq 1 2880); do
  if grep -q 'Epoch(val) \[16\]' "$LOG" 2>/dev/null; then break; fi
  if grep -qi 'CAUGHT AN EXCEPTION' "$LOG" 2>/dev/null; then say "exception detected"; break; fi
  sleep 60
done
sleep 180                      # let the ep16 checkpoint + log flush
pkill -f '[s]emA2R_24e' || true
sleep 60
say "job stopped after ep16 (config stays 24e for schedule continuity)"

# ---- 7) three-caliber report -------------------------------------------------
python me_rssm/sanity/merge_val_curves.py --work-dir "$WD" --b0-log "$B0" --window 12-16 \
  --out "$WD/merged_val_curve.log.json" --json-out "$WD/merged_val_curve_report.json" \
  >> "$QLOG" 2>&1
say "report -> $WD/merged_val_curve_report.json ; DONE"
