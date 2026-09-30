#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

for split in ${EADA_VOC_SPLITS:-1 2 3}; do
  for shot in ${EADA_VOC_SHOTS:-1 2 3 5 10}; do
    for seed in ${EADA_SEEDS:-0 1 2}; do
      bash scripts/run_voc_cell.sh "${split}" "${shot}" "${seed}"
    done
  done
done

python tools/aggregate_runs.py --dataset voc --root outputs/voc --output outputs/voc/summary.json
python tools/compare_expected.py --results outputs/voc/summary.json --table table1_pascal_voc
python tools/compare_expected.py --results outputs/voc/summary.json --table table4_generalized_voc

