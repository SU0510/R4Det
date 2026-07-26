#!/bin/bash
# Single-GPU test script for TJ4D baseline temporal model
# Usage: bash test_TJ4D.sh [checkpoint_path]
# Example: bash test_TJ4D.sh work_dirs/r4det_baseline_temporal/epoch_18.pth

CONFIG_PATH=./configs/r4det/TJ4D-R4Det_baseline_temporal_det3d_2x4_12e.py
CHECKPOINT_PATH=${1:-./work_dirs/r4det_baseline_temporal/latest.pth}

python tools/test_vod.py \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --eval bbox