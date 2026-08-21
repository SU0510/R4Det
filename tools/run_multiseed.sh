#!/usr/bin/env bash
# Run one config with multiple seeds for paper-grade reproducibility.
#
# Usage:
#   SEEDS="0 1 2" BASE_DIR=work_dirs/run10_multiseed \
#     bash tools/run_multiseed.sh configs/r4det/....py 3
#
# Each seed writes to BASE_DIR/seed_<seed> so checkpoints/logs never overwrite.

set -euo pipefail

CONFIG=$1
GPUS=$2
DETERMINISTIC=${DETERMINISTIC:-1}
SEEDS=${SEEDS:-"0 1 2"}
BASE_DIR=${BASE_DIR:?Set BASE_DIR to the multiseed output dir}
CFG_OPTIONS=${CFG_OPTIONS:-""}

extra_args=()
[[ "$DETERMINISTIC" == "1" ]] && extra_args+=(--deterministic)
[[ -n "$CFG_OPTIONS" ]] && extra_args+=(--cfg-options $CFG_OPTIONS)

for seed in $SEEDS; do
  work_dir="${BASE_DIR}/seed_${seed}"
  echo "=== seed=$seed work_dir=$work_dir ==="
  bash tools/dist_train.sh "$CONFIG" "$GPUS" \
    --seed "$seed" --work-dir "$work_dir" "${extra_args[@]}"
done
