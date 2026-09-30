#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
config=${1:?Usage: bash scripts/train_fsis.sh FSIS_CONFIG MERGED_CHECKPOINT [Detectron2 options]}
weights=${2:?Provide a merged EADA detector + base mask checkpoint}
shift 2
exec python train_net.py --config-file "$config" --num-gpus "${NUM_GPUS:-1}" \
  MODEL.WEIGHTS "$weights" "$@"
