#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

for shot in ${EADA_COCO_SHOTS:-1 2 3 5 10 30}; do
  for seed in ${EADA_SEEDS:-0 1 2}; do
    bash scripts/run_coco_cell.sh "${shot}" "${seed}"
  done
done

python tools/aggregate_runs.py --dataset coco --root outputs/coco --output outputs/coco/summary.json
python tools/compare_expected.py --results outputs/coco/summary.json --table table2_ms_coco

