#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

# Reproduce exactly the seed selected for each MS COCO table cell. Use
# reproduce_table2_coco.sh to rerun all three seeds and repeat selection.
for entry in "1 0" "2 2" "3 2" "5 2" "10 2" "30 1"; do
  read -r shot seed <<<"${entry}"
  bash scripts/run_coco_cell.sh "${shot}" "${seed}"
done

python tools/aggregate_runs.py \
  --dataset coco --root outputs/coco --output outputs/coco/summary.json
python tools/compare_expected.py \
  --results outputs/coco/summary.json --table table2_ms_coco

