#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
config=${1:?Usage: bash scripts/train.sh CONFIG [Detectron2 options]}
shift
exec python train_net.py --config-file "$config" --num-gpus "${NUM_GPUS:-1}" "$@"
