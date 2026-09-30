#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

: "${DETECTRON2_DATASETS:?Set DETECTRON2_DATASETS to the prepared VOC/COCO root}"

python tools/check_environment.py
python tools/check_data.py --dataset voc
python tools/check_weights.py --detection-only

NUM_GPUS=${NUM_GPUS:-4} bash scripts/run_voc_cell.sh 1 1 0

python tools/aggregate_runs.py \
  --dataset voc --root outputs/voc --output outputs/voc/quick_summary.json
python tools/compare_expected.py \
  --results outputs/voc/quick_summary.json \
  --table table1_pascal_voc \
  --allow-missing

echo "Reviewer quick reproduction: PASS"

