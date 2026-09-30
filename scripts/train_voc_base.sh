#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

split=${1:?Usage: bash scripts/train_voc_base.sh SPLIT_ID}
[[ "${split}" =~ ^[123]$ ]] || { echo "SPLIT_ID must be 1, 2, or 3" >&2; exit 2; }

init=${EADA_R101_INIT:-weights/R-101.pkl}
[[ -f "${init}" ]] || { echo "Missing ImageNet R101 init: ${init}" >&2; exit 3; }

output="outputs/base/voc_split${split}"
expanded="weights/voc_split${split}_all.pth"
mkdir -p "${output}" weights

bash scripts/train.sh "configs/voc/base${split}.yaml" \
  MODEL.WEIGHTS "${init}" \
  OUTPUT_DIR "${output}"

python tools/expand_checkpoint.py \
  "${output}/model_final.pth" "${expanded}" \
  --dataset voc --split "${split}" --seed 0

echo "expanded_checkpoint=${expanded}"

