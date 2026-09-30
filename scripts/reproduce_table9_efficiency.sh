#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

clip=${EADA_CLIP_CHECKPOINT:-}
mkdir -p outputs/efficiency

baseline=outputs/ablation/table5/10shot/baseline/train/model_final.pth
adapted=outputs/ablation/table6/10shot/dc_saia/train/model_final.pth

if [[ ! -f "${baseline}" ]]; then
  EADA_ABLATION_SHOTS=10 bash scripts/reproduce_table5_ablation.sh
fi
if [[ ! -f "${adapted}" ]]; then
  bash scripts/reproduce_table6_spdi.sh
fi

python tools/profile_efficiency.py \
  --config configs/voc/split1_10shot.yaml \
  --checkpoint "${baseline}" \
  --disable-saia \
  --training-log outputs/ablation/table5/10shot/baseline/train/log.txt \
  --dataset voc_2007_test_all1 \
  --output outputs/efficiency/baseline.json

python tools/profile_efficiency.py \
  --config configs/voc/split1_10shot.yaml \
  --checkpoint "${adapted}" \
  --training-log outputs/ablation/table6/10shot/dc_saia/train/log.txt \
  --dataset voc_2007_test_all1 \
  --output outputs/efficiency/dc_saia.json

clip_args=()
if [[ -n "${clip}" ]]; then clip_args=(--clip-checkpoint "${clip}"); fi
python tools/profile_efficiency.py \
  --config configs/voc/split1_10shot.yaml \
  --checkpoint "${adapted}" \
  --training-log outputs/ablation/table6/10shot/dc_saia/train/log.txt \
  --dataset voc_2007_test_all1 \
  --spdi --top-k 100 "${clip_args[@]}" \
  --output outputs/efficiency/full_eada.json

echo "Reference paper hardware: one RTX 4070 Ti."
echo "Compare with expected/paper_results.json -> table9_efficiency_rtx4070ti."
python tools/verify_auxiliary.py --table table9

