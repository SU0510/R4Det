#!/usr/bin/env bash
# Overnight pipeline after MCI seed2's ep16 gate (prepared 2026-10-05).
#
# Runs in tmux, fully autonomous, bounded to CHEAP operations:
#   1. wait for the seed2 watchdog to finish its ep16 probes
#   2. three-seed paired verdict (CPU) -> verdict file
#   3. Ped spatial gate probe on the seed0 ep16 checkpoint (GPU, ~15 min;
#      H2/H3 adjudication for the Ped red flag)
#   4. the pre-registered 7-ablation 450-iter mechanism smokes, sequential
#      (each ~25 min on GPU 5/6/7; gate reports appended)
# Nothing here launches any 24e run. Disk guard: skip a smoke if /data
# free < 20G. Idempotent: skips a smoke whose iter_450.pth already exists.
#
# Launch:
#   tmux new-session -d -s mci_overnight 'bash me_rssm/sanity/overnight_after_seed2.sh'
set -u
cd /home/lurui/workspace/R4Det

LOG=/data/lurui/work_dirs/mci_overnight.log
SEED2_WD=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mci_2x4_24e_seed2
SEED0_CKPT=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mci_2x4_24e_seed0/epoch_16.pth
CFG=me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py
SMOKE_CFG=me_rssm/sanity/mci_smoke_450iter.py

log() { echo "[overnight] $(date -u '+%F %T') $*" >> "$LOG"; }

log "pipeline started; waiting for seed2 ep16 probes ..."

# ---- 1. wait for the seed2 watchdog probes --------------------------------
while [ ! -f "$SEED2_WD/post_source_gated_ep16_stride700.json" ]; do
  sleep 300
done
log "seed2 probes detected"

# wait for the watchdog's GPU window to fully drain (probe PIDs exit)
for _ in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    -i 5,6,7 2>/dev/null | awk -F', ' '$2 > 2000 {n++} END {print n+0}')
  [ "$busy" = "0" ] && break
  sleep 20
done

# shellcheck disable=SC1091
source .envrc || true

# ---- 2. three-seed verdict -------------------------------------------------
python3 me_rssm/sanity/mci_threeseed_verdict.py \
  > /data/lurui/work_dirs/mci_threeseed_verdict.txt 2>&1
log "three-seed verdict written (mci_threeseed_verdict.txt):"
cat /data/lurui/work_dirs/mci_threeseed_verdict.txt >> "$LOG"

# ---- 3. Ped spatial gate probe (H2/H3 adjudication) ------------------------
if [ ! -f "$SEED2_WD/gate_spatial_ep16.json" ]; then
  CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_gate_spatial.py \
    --config "$CFG" --checkpoint "$SEED0_CKPT" \
    --start-index 3 --limit 6 --stride 700 \
    --output "$SEED2_WD/gate_spatial_ep16.json" >> "$LOG" 2>&1 \
    && log "spatial gate probe done" || log "spatial gate probe FAILED (see log)"
fi

# ---- 4. ablation 450-iter mechanism smokes (pre-registered grid) -----------
run_smoke() {
  local name="$1"; shift
  local wd="/data/lurui/work_dirs/mci_abl_${name}_smoke"
  local free_gb
  free_gb=$(df -BG --output=avail /data | tail -1 | tr -dc '0-9')
  if [ "${free_gb:-0}" -lt 20 ]; then
    log "SKIP $name: only ${free_gb}G free on /data"
    return
  fi
  if [ -f "$wd/iter_450.pth" ]; then
    log "SKIP $name: iter_450.pth already exists"
    return
  fi
  mkdir -p "$wd"
  log "smoke $name starting"
  CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh \
    "$SMOKE_CFG" 3 --seed 0 --deterministic --work-dir "$wd" \
    "$@" > "$wd/stdout.log" 2>&1
  if [ -f "$wd/iter_450.pth" ]; then
    log "smoke $name finished"
    python3 me_rssm/sanity/mci_gate_report.py "$wd" >> "$LOG" 2>&1 \
      && log "gate report for $name appended"
  else
    log "smoke $name FAILED (no checkpoint; see $wd/stdout.log)"
  fi
}

run_smoke no_motion_offsets  --cfg-options model.temporal_fusion.use_motion_offsets=False
run_smoke no_dynamic_gate    --cfg-options model.temporal_fusion.use_dynamic_gate=False
run_smoke no_reliability_gain --cfg-options model.temporal_fusion.use_reliability_gain=False
run_smoke no_gated_readout   --cfg-options model.temporal_fusion.use_gated_readout=False
run_smoke no_innov_in_gain   --cfg-options model.temporal_fusion.innov_in_gain=False
run_smoke no_modality_obs    --cfg-options model.temporal_fusion.use_modality_obs=False model.temporal_fusion.use_motion_offsets=False model.temporal_fusion.use_dynamic_gate=False
run_smoke no_innovation      --cfg-options model.temporal_fusion.posterior_mode=standard model.temporal_fusion.innov_in_gain=False

log "pipeline complete"
