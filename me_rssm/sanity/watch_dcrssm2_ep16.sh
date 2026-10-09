#!/usr/bin/env bash
# DC-RSSM-2 (L1 analytic ego-warp) ep16 gate watchdog.
#
# Waits for epoch_16 val + checkpoint of the DC-RSSM-2 seed0 run, stops the
# training session, then runs the pre-registered probe battery:
#
#   P1  z_utilization_gated (w3/w100/w1800)  -- family-standard readout
#       (shuffle_z / zero_z head, h/e dependency) for cross-generation
#       comparability with MCI / MCI-v2 / DC-RSSM-1.
#   P2  posterior_source_gated               -- source attribution.
#   P3  probe_ego_warp_sanity                -- v_ego distribution, mirror
#       fidelity (asserted equal to the module's stashed solve), static-set
#       health (sel/measured, trim_frac, tau_r). Geometry teeth: |v_ego| in
#       [3, 20] m/s on ego-moving val frames is NOT expected on all frames
#       (TJ4D city crawl measured 1.4-3.0 m/s on the probed frames); the
#       gate is the LMedS agreement in P4 plus non-inertness.
#   P4  probe_ego_geometry_test dump + robust + robustfix -- LMedS truth
#       (rigid field, 0.013 m/s median residual on the smoke checkpoint)
#       vs the module's robust solve; the sign/shape agreement IS the
#       geometry-wiring gate.
#   P5  probe_ego_alignment_toggle           -- PRE-REGISTERED mechanism
#       gate: paired toggle of warp_scale (ON vs forced 0) on the same
#       clips, static-region alignment residual sign test, plus the
#       inertness measurement (mean |warp_scale|*dt*|v_ego| against the
#       needed per-frame displacement).
#
# Usage (user-approved launch only):
#   SESSION=dcrssm2 tmux new-session -d -s dcrssm2_watch \
#     'bash me_rssm/sanity/watch_dcrssm2_ep16.sh'
set -u

cd /home/lurui/workspace/R4Det

SESSION="${SESSION:-dcrssm2}"
WORK_DIR="${WORK_DIR:-/data/lurui/work_dirs/fgfull_N4_no2d_igdr_dcrssm2_2x4_24e_seed0}"
LOG="${LOG:-$WORK_DIR/stdout.log}"
CFG="${CFG:-me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_dcrssm2.py}"
CKPT="$WORK_DIR/epoch_16.pth"
WATCH_LOG="$WORK_DIR/watch_seed0_ep16.log"
STOP_TIMEOUT_SEC="${STOP_TIMEOUT_SEC:-1800}"

START_TS=$(date +%s)
echo "[watch] $(date -u '+%F %T') waiting for ep16 val + fresh checkpoint (start ts $START_TS) ..." >> "$WATCH_LOG"
while true; do
  if [ -f "$CKPT" ] && grep -q 'Epoch(val) \[16\]' "$LOG" 2>/dev/null; then
    CKPT_TS=$(stat -c %Y "$CKPT")
    if [ "$CKPT_TS" -gt "$START_TS" ]; then
      echo "[watch] $(date -u '+%F %T') gate hit (fresh ckpt); stopping $SESSION" >> "$WATCH_LOG"
      tmux send-keys -t "$SESSION" C-c 2>/dev/null
      break
    fi
    echo "[watch] $(date -u '+%F %T') stale epoch_16.pth ignored" >> "$WATCH_LOG"
  fi
  sleep 30
done

waited=0
while [ "$waited" -lt "$STOP_TIMEOUT_SEC" ]; do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    -i 5,6,7 2>/dev/null | awk -F', ' '$2 > 2000 {n++} END {print n+0}')
  if [ "$busy" = "0" ]; then
    break
  fi
  sleep 15
  waited=$((waited + 15))
done
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  -i 5,6,7 2>/dev/null | awk -F', ' '$2 > 2000 {n++} END {print n+0}')
if [ "$busy" != "0" ]; then
  # Same failure mode that cost the DC-RSSM-1 generation its probe pass:
  # if the stop signal did not land, running the battery anyway only
  # yields OOM garbage. Abort loudly; the operator reruns this script.
  echo "[watch] $(date -u '+%F %T') ABORT: $busy of 3 GPUs still busy after ${waited}s (stop signal did not land?) -- no probes run, rerun this script" >> "$WATCH_LOG"
  exit 1
fi
echo "[watch] $(date -u '+%F %T') GPUs 5/6/7 free after ${waited}s; running probes" >> "$WATCH_LOG"

# shellcheck disable=SC1091
source .envrc

# --- P1/P2 family-standard readouts -------------------------------------
CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_z_utilization_gated.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --limit 4 --batch-size 4 --shuffle-offset 2 \
  --output "$WORK_DIR/z_util_gated_ep16_w3.json" >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=6 python me_rssm/sanity/probe_z_utilization_gated.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 100 --limit 4 --batch-size 4 --shuffle-offset 2 \
  --output "$WORK_DIR/z_util_gated_ep16_w100.json" >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=7 python me_rssm/sanity/probe_z_utilization_gated.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset 2 \
  --output "$WORK_DIR/z_util_gated_ep16_w1800.json" >> "$WATCH_LOG" 2>&1

CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_posterior_source_gated.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --limit 7 --stride 700 \
  --output "$WORK_DIR/post_source_ep16_stride700.json" >> "$WATCH_LOG" 2>&1

# --- P3 ego solve readout ----------------------------------------------
CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_ego_warp_sanity.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --limit 4 --stride 700 \
  --output "$WORK_DIR/ego_warp_sanity_ep16.json" >> "$WATCH_LOG" 2>&1

# --- P4 geometry truth (dump, then offline LMedS vs module solve) -------
CUDA_VISIBLE_DEVICES=6 python me_rssm/sanity/probe_ego_geometry_test.py dump \
  --config "$CFG" --checkpoint "$CKPT" --indices 3 703 1403 \
  --output "$WORK_DIR/canvases_ep16.pt" >> "$WATCH_LOG" 2>&1

python me_rssm/sanity/probe_ego_geometry_test.py robust \
  --dump "$WORK_DIR/canvases_ep16.pt" \
  --output "$WORK_DIR/robust_solve_ep16.json" >> "$WATCH_LOG" 2>&1
python me_rssm/sanity/probe_ego_geometry_test.py robustfix \
  --dump "$WORK_DIR/canvases_ep16.pt" \
  --output "$WORK_DIR/robustfix_ep16.json" >> "$WATCH_LOG" 2>&1

# --- P5 pre-registered mechanism gate (paired toggle + inertness) -------
CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_ego_alignment_toggle.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --start-index 3 --limit 10 --stride 300 \
  --output "$WORK_DIR/ego_alignment_toggle_ep16.json" >> "$WATCH_LOG" 2>&1

echo "[watch] $(date -u '+%F %T') probes done; gate decision is a manual step" >> "$WATCH_LOG"
