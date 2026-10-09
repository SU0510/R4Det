#!/usr/bin/env bash
# GPU-4 read-only probe queue: measurement (1) region-split z_t dependency and
# measurement (2) reconstruction-target variance decomposition.
#
# The user explicitly authorised GPU 4 for exactly these two measurements
# ("可以现在gpu4上做1和2"); GPUs 5/6/7 stay reserved for DC-RSSM-2 training.
# Read-only: no training, no checkpoint writes; every artifact lands in the
# run's work_dir (or under /data/lurui/work_dirs/gpu4_probes for the logs).
set -u

PY=/home/lurui/envs/miniforge3/envs/r4det/bin/python
REPO=/home/lurui/workspace/R4Det
LOGDIR=/data/lurui/work_dirs/gpu4_probes
WD=/data/lurui/work_dirs
cd "$REPO"
export CUDA_VISIBLE_DEVICES=4
mkdir -p "$LOGDIR"

run() {
  local name="$1"; shift
  echo "=== $name start $(date -u +%FT%TZ)" | tee -a "$LOGDIR/queue.log"
  "$@" > "$LOGDIR/$name.log" 2>&1
  local rc=$?
  echo "=== $name exit=$rc $(date -u +%FT%TZ)" | tee -a "$LOGDIR/queue.log"
  return $rc
}

# (2a) tool validation anchor: the extended probe must reproduce the §51/52
# table on the same checkpoint and sample count (pool16_globalstd 92.1/7.9,
# pool32 88.8/11.2, fullres 81.4/18.6, pool16-chstd 89.1/10.9).
run m2_anchor_lowdim_ep12 "$PY" tools/probe_recon_target.py \
  --config configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z.py \
  --checkpoint "$WD/lowdim_z_future_consistent_3x2x2_24e_seed0/epoch_12.pth" \
  --start-index 3 --limit 16 \
  --output "$WD/lowdim_z_future_consistent_3x2x2_24e_seed0/recon_target_anchor_ep12_w3_16.json"

# (2b) current-generation headroom on the DC-RSSM-1 ep16 checkpoint.
run m2_dcrssm1_ep16 "$PY" tools/probe_recon_target.py \
  --config me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_dcrssm1.py \
  --checkpoint "$WD/fgfull_N4_no2d_igdr_dcrssm1_2x4_24e_seed0/epoch_16.pth" \
  --start-index 3 --limit 16 \
  --output "$WD/fgfull_N4_no2d_igdr_dcrssm1_2x4_24e_seed0/recon_target_ep16_w3_16.json"

# (1b) baseline contrast for the region split. The baseline/MAI config has no
# motion chain, so the probe reports fg/bg/bg_hi/bg_top10 only and says so.
run m1_baseline_ep14 "$PY" me_rssm/sanity/probe_z_util_region_split.py \
  --config configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py \
  --checkpoint "$WD/fgfull_N4_no2d_igdr_2x4_24e_seed0/epoch_14.pth" \
  --start-index 3 --limit 12 \
  --output "$WD/fgfull_N4_no2d_igdr_2x4_24e_seed0/zutil_region_ep14_w3_12.json"

# (2c) the L1 ego-warp generation: dcrssm2 epoch_4 (early checkpoint; the same
# probe runs again at ep16 under the gate). Analytic v_ego is measured from the
# Doppler canvas, so diff_warped_a1 does not depend on the warp having been
# learned yet.
run m2_dcrssm2_ep4 "$PY" tools/probe_recon_target.py \
  --config me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_dcrssm2.py \
  --checkpoint "$WD/fgfull_N4_no2d_igdr_dcrssm2_2x4_24e_seed0/epoch_4.pth" \
  --start-index 3 --limit 16 \
  --output "$WD/fgfull_N4_no2d_igdr_dcrssm2_2x4_24e_seed0/recon_target_ep4_w3_16.json"

echo "=== queue done $(date -u +%FT%TZ)" | tee -a "$LOGDIR/queue.log"
