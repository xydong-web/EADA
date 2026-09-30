#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

# Reproduce exactly the seed selected for each PASCAL VOC table cell in the
# manuscript. Use reproduce_table1_voc.sh instead to rerun all three seeds and
# independently repeat the selection procedure.
for entry in \
  "1 1 0" "1 2 0" "1 3 0" "1 5 0" "1 10 0" \
  "2 1 0" "2 2 1" "2 3 0" "2 5 0" "2 10 2" \
  "3 1 0" "3 2 2" "3 3 2" "3 5 1" "3 10 2"; do
  read -r split shot seed <<<"${entry}"
  bash scripts/run_voc_cell.sh "${split}" "${shot}" "${seed}"
done

python tools/aggregate_runs.py \
  --dataset voc --root outputs/voc --output outputs/voc/summary.json
python tools/compare_expected.py \
  --results outputs/voc/summary.json --table table1_pascal_voc
python tools/compare_expected.py \
  --results outputs/voc/summary.json --table table4_generalized_voc

