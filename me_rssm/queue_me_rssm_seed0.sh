#!/usr/bin/env bash
# Queue the ME-RSSM seed0 mainline training behind the currently running
# CycCls seed2 run (research-package output; touches no baseline file).
#
# Trigger condition: no `train_vod.py --config` process left running
# (robust against tmux sessions lingering after the training exits).
# Launch target: GPUs 5,6,7 -- the same 3-GPU slot the multiseed queue used,
# same effective batch (3 x samples_per_gpu=4), same protocol as Run 10
# multi-seed (seed 0, --deterministic, 24 epochs).
#
# Log: /data/lurui/work_dirs/me_rssm_queue.log

set -u
cd /home/lurui/workspace/R4Det

LOG=/data/lurui/work_dirs/me_rssm_queue.log
echo "[queue] $(date '+%F %T') waiting for current training to finish ..." >> "$LOG"

while pgrep -f "train_vod.py --config" > /dev/null 2>&1; do
  sleep 300
done

echo "[queue] $(date '+%F %T') GPU slot free; starting ME-RSSM seed0" >> "$LOG"

source /home/lurui/envs/miniforge3/etc/profile.d/conda.sh
conda activate r4det
export CUDA_VISIBLE_DEVICES=5,6,7

bash tools/dist_train.sh \
  me_rssm/configs/TJ4D-R4Det_me_rssm_N4_24e_pretrained_v2_head.py 3 \
  --seed 0 --deterministic \
  --work-dir /data/lurui/work_dirs/me_rssm_N4_2x4_24e_seed0 \
  >> "$LOG" 2>&1

echo "[queue] $(date '+%F %T') ME-RSSM seed0 exited with status $?" >> "$LOG"
