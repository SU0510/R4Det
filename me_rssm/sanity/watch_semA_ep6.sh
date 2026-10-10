#!/usr/bin/env bash
# Candidate A ep6 gate watchdog (pre-registered S1 stop point).
#
# Waits for epoch 6 val + a FRESH epoch_6.pth (mtime stale-guard so a leftover
# checkpoint from an earlier attempt cannot trigger it), interrupts the tmux
# training session, waits for the DDP workers to release GPUs 5/6/7, then runs
# the pre-registered M1-M4 mechanism battery and the family-standard
# counterfactual readout for cross-generation comparability.
#
#   M1-M4  me_rssm/sanity/probe_semreadout_ep6.py  -- the S1 gate itself
#   P1     me_rssm/sanity/probe_z_utilization_standard.py (w3/w100/w1800) --
#          shuffle_z / zero_z / prior_only readout, comparable with every
#          previous generation (MAI / MCI / DC-RSSM). Candidate A runs the
#          STANDARD family (MotionAlignedRSSMFusion), which has no
#          `_z_intervention` hook point, so the gated probe aborts on it with
#          "target model is not a gated variant" (verified 2026-10-10). The
#          standard probe runs the same posterior-head intervention and the
#          same scoring, and forwards the deployed class residual to the head.
#   P2     tools/probe_posterior_source.py -- source attribution. The standard
#          tool (transition hook), not the MCI variant that reads the eval-mode
#          `_last_h_t` stash.
#
# The DECISION stays manual: this script only stops training and freezes
# evidence. It never launches a follow-up run (a GO requires user-approved
# §5 S2 numbers first).
#
# Usage:
#   SESSION=semA_ep6 tmux new-session -d -s semA_watch \
#     'bash me_rssm/sanity/watch_semA_ep6.sh'
set -u

cd /home/lurui/workspace/R4Det

SESSION="${SESSION:-semA_ep6}"
WORK_DIR="${WORK_DIR:-/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_semA_seed0}"
LOG="${LOG:-$WORK_DIR/stdout.log}"
CFG="${CFG:-me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semreadout.py}"
CKPT="$WORK_DIR/epoch_6.pth"
WATCH_LOG="$WORK_DIR/watch_semA_ep6.log"
STOP_TIMEOUT_SEC="${STOP_TIMEOUT_SEC:-1800}"

START_TS=$(date +%s)
echo "[watch] $(date -u '+%F %T') waiting for ep6 val + fresh checkpoint (start ts $START_TS)" >> "$WATCH_LOG"
while true; do
  if [ -f "$CKPT" ] && grep -q 'Epoch(val) \[6\]' "$LOG" 2>/dev/null; then
    CKPT_TS=$(stat -c %Y "$CKPT")
    if [ "$CKPT_TS" -gt "$START_TS" ]; then
      echo "[watch] $(date -u '+%F %T') gate hit (fresh ckpt); stopping $SESSION" >> "$WATCH_LOG"
      tmux send-keys -t "$SESSION" C-c 2>/dev/null
      break
    fi
    echo "[watch] $(date -u '+%F %T') stale epoch_6.pth ignored" >> "$WATCH_LOG"
  fi
  sleep 30
done

# GPU-idle is the stop criterion, not process exit: torch distributed teardown
# has exceeded 10 min before (2026-10-01 seed1 incident).
waited=0
while [ "$waited" -lt "$STOP_TIMEOUT_SEC" ]; do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    -i 5,6,7 2>/dev/null | awk -F', ' '$2 > 2000 {n++} END {print n+0}')
  [ "$busy" = "0" ] && break
  sleep 15
  waited=$((waited + 15))
done
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  -i 5,6,7 2>/dev/null | awk -F', ' '$2 > 2000 {n++} END {print n+0}')
if [ "$busy" != "0" ]; then
  # Running the battery on busy GPUs only yields OOM garbage. Abort loudly.
  echo "[watch] $(date -u '+%F %T') ABORT: $busy/3 GPUs busy after ${waited}s (stop signal did not land) -- rerun this script" >> "$WATCH_LOG"
  exit 1
fi
echo "[watch] $(date -u '+%F %T') GPUs free after ${waited}s; running probes" >> "$WATCH_LOG"

# shellcheck disable=SC1091
source .envrc

# Preflight: every probe below runs through `python`, and a probe that cannot
# start writes NO JSON while this script still reaches "probes done". Fail
# here, loudly, instead of leaving a silently empty evidence bundle. The
# verdict assembler also checks the artifacts, but it cannot tell the
# difference between "the environment was broken" and "the probe crashed".
if ! command -v python >/dev/null 2>&1; then
  echo "[watch] $(date -u '+%F %T') ABORT: 'python' not on PATH after sourcing .envrc -- rerun this script in a shell where the r4det env activates" >> "$WATCH_LOG"
  exit 1
fi
if ! python -c 'import torch, mmcv, mmdet3d' >/dev/null 2>&1; then
  echo "[watch] $(date -u '+%F %T') ABORT: python cannot import torch/mmcv/mmdet3d -- environment is not the r4det env; rerun this script" >> "$WATCH_LOG"
  exit 1
fi

# --- the pre-registered S1 gate -----------------------------------------
CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_semreadout_ep6.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --limit 10 --stride 200 --shuffle-offset 1 \
  --output "$WORK_DIR/semreadout_ep6_gate.json" >> "$WATCH_LOG" 2>&1

# --- family-standard counterfactual readout (three windows) -------------
# Candidate A runs the STANDARD family (MotionAlignedRSSMFusion): it has no
# `_z_intervention` hook point, so probe_z_utilization_gated.py aborts with
# "target model is not a gated variant" (verified 2026-10-10 on the 450-iter
# smoke). Use the standard-family probe instead; it performs the same
# posterior-head intervention and the same scoring (dz's own code), and
# additionally forwards the deployed class residual to the head.
CUDA_VISIBLE_DEVICES=6 python me_rssm/sanity/probe_z_utilization_standard.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --limit 4 --batch-size 4 --shuffle-offset 2 \
  --output "$WORK_DIR/z_util_std_ep6_w3.json" >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=7 python me_rssm/sanity/probe_z_utilization_standard.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 100 --limit 4 --batch-size 4 --shuffle-offset 2 \
  --output "$WORK_DIR/z_util_std_ep6_w100.json" >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_z_utilization_standard.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset -1 \
  --output "$WORK_DIR/z_util_std_ep6_w1800.json" >> "$WATCH_LOG" 2>&1

# Source attribution: the standard-family tool (transition hook), NOT the
# MCI variant that reads the eval-mode `_last_h_t` stash.
CUDA_VISIBLE_DEVICES=5 python tools/probe_posterior_source.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --limit 4 --stride 700 \
  --output "$WORK_DIR/post_source_std_ep6_stride700.json" >> "$WATCH_LOG" 2>&1

# Assemble the readings + print the pre-registered fork. Read-only, decides
# nothing, launches nothing: the verdict stays a manual step.
python me_rssm/sanity/semA_ep6_verdict.py --run "$WORK_DIR" >> "$WATCH_LOG" 2>&1

# Render the rule-5/rule-6 section-72 draft from the evidence bundle so the
# training record is never hand-transcribed. Also read-only; the generated file
# is a DRAFT -- a human pastes it into docs/training_runs_full.md after filling
# the FILL markers. A failure here must not be silent, so it is reported.
python me_rssm/sanity/semA_ep6_record.py --run "$WORK_DIR" \
  --output "$WORK_DIR/section72_draft.md" >> "$WATCH_LOG" 2>&1
RECORD_RC=$?
if [ "$RECORD_RC" -eq 0 ]; then
  echo "[watch] $(date -u '+%F %T') section-72 draft written to $WORK_DIR/section72_draft.md" >> "$WATCH_LOG"
else
  echo "[watch] $(date -u '+%F %T') WARNING: section-72 draft generation failed (rc=$RECORD_RC); the evidence bundle is unaffected" >> "$WATCH_LOG"
fi

echo "[watch] $(date -u '+%F %T') probes done; gate decision is a manual step" >> "$WATCH_LOG"
