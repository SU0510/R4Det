#!/usr/bin/env bash
# Queue the MCI-RSSM seed0 24e mainline run behind whatever is currently
# training on the 5/6/7 slot (research-package output; touches no baseline
# file). Prepared 2026-10-02; launch is a user decision.
#
# Trigger condition: no `train_vod.py --config` process left running. This
# waits for the WHOLE current queue — MAI seed2 (auto-stops at ep16) AND the
# fgfull18 seed0 relay that the chain watcher starts afterwards. To run MCI
# before fgfull18 instead, cancel the fgfull18 watcher first
# (tmux kill-session -t fgfull18_seed0_watch) and re-run this script; the
# choice is recorded in docs/12 §runbook as a user decision.
#
# Log: /data/lurui/work_dirs/mci_queue.log

set -u
cd /home/lurui/workspace/R4Det

LOG=/data/lurui/work_dirs/mci_queue.log
echo "[queue] $(date '+%F %T') waiting for current training to finish ..." >> "$LOG"

while pgrep -f "train_vod.py --config" > /dev/null 2>&1; do
  sleep 300
done

echo "[queue] $(date '+%F %T') GPU slot free; starting MCI-RSSM seed0" >> "$LOG"

source /home/lurui/envs/miniforge3/etc/profile.d/conda.sh
conda activate r4det
export CUDA_VISIBLE_DEVICES=5,6,7

bash tools/dist_train.sh \
  me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py 3 \
  --seed 0 --deterministic \
  --work-dir /data/lurui/work_dirs/fgfull_N4_no2d_igdr_mci_2x4_24e_seed0 \
  >> "$LOG" 2>&1

echo "[queue] $(date '+%F %T') MCI-RSSM seed0 exited with status $?" >> "$LOG"
