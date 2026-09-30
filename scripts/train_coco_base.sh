#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

init=${EADA_R101_INIT:-weights/R-101.pkl}
[[ -f "${init}" ]] || { echo "Missing ImageNet R101 init: ${init}" >&2; exit 3; }

output="outputs/base/coco"
expanded="weights/coco_all.pth"
mkdir -p "${output}" weights

bash scripts/train.sh configs/coco/base.yaml \
  MODEL.WEIGHTS "${init}" \
  OUTPUT_DIR "${output}"

python tools/expand_checkpoint.py \
  "${output}/model_final.pth" "${expanded}" \
  --dataset coco --seed 0

echo "expanded_checkpoint=${expanded}"

