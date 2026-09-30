#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
config=${1:?Usage: bash scripts/eval.sh CONFIG CHECKPOINT [config overrides]}
weights=${2:?Provide a detector checkpoint}
shift 2
exec python train_net.py --config-file "$config" --num-gpus "${NUM_GPUS:-1}" --eval-only MODEL.WEIGHTS "$weights" "$@"
