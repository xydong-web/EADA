#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

detector=${EADA_COCO_BASE_DETECTOR:-outputs/base/coco/model_final.pth}
[[ -f "${detector}" ]] || {
  echo "Missing ${detector}. Run: NUM_GPUS=4 bash scripts/train_coco_base.sh" >&2
  exit 3
}

out=outputs/fsis/base_mask
mkdir -p "${out}"
bash scripts/train.sh configs/coco/fsis/base_mask.yaml \
  MODEL.WEIGHTS "${detector}" \
  OUTPUT_DIR "${out}"

echo "base_mask_checkpoint=${out}/model_final.pth"

