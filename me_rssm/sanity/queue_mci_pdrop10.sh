#!/usr/bin/env bash
# Follow-up queue: after the overnight pipeline completes (three-seed
# verdict + Ped spatial probe + 7 ablation smokes), launch the Ped-fix
# run (MCI + modality_dropout=0.1, seed0, 24e) with an ep16 watchdog.
# Prepared 2026-10-05 per user approval ("然后进行ped修复").
set -u
cd /home/lurui/workspace/R4Det

LOG=/data/lurui/work_dirs/mci_pdrop10_queue.log
log() { echo "[pdrop10] $(date -u '+%F %T') $*" >> "$LOG"; }

log "waiting for the overnight pipeline to complete ..."
while ! grep -q 'pipeline complete' /data/lurui/work_dirs/mci_overnight.log 2>/dev/null; do
  sleep 300
done
log "pipeline complete; checking GPUs"

for _ in $(seq 1 120); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    -i 5,6,7 2>/dev/null | awk -F', ' '$2 > 2000 {n++} END {print n+0}')
  [ "$busy" = "0" ] && break
  sleep 20
done

log "launching MCI+modality_dropout=0.1 seed0 24e"
source /home/lurui/envs/miniforge3/etc/profile.d/conda.sh
conda activate r4det
export CUDA_VISIBLE_DEVICES=5,6,7

WD=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mci_pdrop10_2x4_24e_seed0
mkdir -p "$WD"
tmux kill-session -t mci_pdrop10 2>/dev/null || true
tmux new-session -d -s mci_pdrop10 \
  "cd /home/lurui/workspace/R4Det && source .envrc && CUDA_VISIBLE_DEVICES=5,6,7 \
   bash tools/dist_train.sh \
   me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci_pdrop10.py 3 \
   --seed 0 --deterministic --work-dir $WD > $WD/stdout.log 2>&1"
log "training launched in tmux mci_pdrop10"

sleep 60
tmux kill-session -t mci_pdrop10_watch 2>/dev/null || true
tmux new-session -d -s mci_pdrop10_watch \
  "SEED=0 SESSION=mci_pdrop10 WORK_DIR=$WD \
   LOG=$WD/stdout.log \
   bash /home/lurui/workspace/R4Det/me_rssm/sanity/watch_mci_seed_ep16.sh"
log "ep16 watchdog armed (mci_pdrop10_watch)"
