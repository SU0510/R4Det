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

if [[ "$DETERMINISTIC" == "1" ]]; then
  EXTRA="--deterministic"
else
  EXTRA=""
fi

for seed in $SEEDS; do
  work_dir="${BASE_DIR}/seed_${seed}"
  echo "=== seed=$seed work_dir=$work_dir ==="
  bash tools/dist_train.sh "$CONFIG" "$GPUS" \
    --seed "$seed" \
    --work-dir "$work_dir" \
    $EXTRA
done
