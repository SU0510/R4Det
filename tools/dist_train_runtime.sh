#!/usr/bin/env bash
# Same single-node environment as dist_train.sh; explicit runtime mode required.
set -euo pipefail
if (( $# < 2 )); then
    echo "Usage: $0 CONFIG GPUS --detect-anomaly on|off --work-dir NEW_DIR [--resume-from CKPT ...]" >&2
    exit 2
fi
CONFIG=$1
GPUS=$2
shift 2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,6,7}
IFS=, read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
declare -A SEEN=()
for device in "${DEVICES[@]}"; do
    [[ $device =~ ^[567]$ ]] || { echo "Only GPUs 5/6/7 allowed" >&2; exit 2; }
    [[ -z ${SEEN[$device]+set} ]] || { echo "Duplicate GPU IDs" >&2; exit 2; }
    SEEN[$device]=1
done
[[ $GPUS =~ ^[1-3]$ && $GPUS -le ${#DEVICES[@]} ]] || { echo "Invalid GPUS" >&2; exit 2; }
export NUMBA_CUDA_FORCE_PTX_VERSION=82 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONFAULTHANDLER=1
export NATTEN_LOG_LEVEL=critical
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
exec python -m torch.distributed.launch \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
    --nproc_per_node="$GPUS" --master_port="${PORT:-$((RANDOM + 10000))}" \
    "$ROOT/tools/train_vod_runtime.py" --config "$CONFIG" \
    --seed "${SEED:-0}" --launcher pytorch "$@"
