#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

base=${EADA_VOC_BASE_EXPANDED:-weights/voc_split1_all.pth}
[[ -f "${base}" ]] || { echo "Missing ${base}" >&2; exit 3; }

mkdir -p outputs/diagnostics/table7

baseline=outputs/ablation/table5/1shot/baseline/train/model_final.pth
saia=outputs/ablation/table5/1shot/saia/train/model_final.pth
if [[ ! -f "${baseline}" || ! -f "${saia}" ]]; then
  echo "Table-5 checkpoints are missing; generating the required 1-shot controls"
  EADA_ABLATION_SHOTS=1 bash scripts/reproduce_table5_ablation.sh
fi

python tools/saia_proposal_recall.py \
  --config configs/voc/split1_1shot.yaml \
  --checkpoint "${baseline}" \
  --output outputs/diagnostics/table7/saia_control.json \
  --top-k 100

python tools/saia_proposal_recall.py \
  --config configs/voc/split1_1shot.yaml \
  --checkpoint "${saia}" \
  --output outputs/diagnostics/table7/saia_proposed.json \
  --saia --alpha 0.2 --top-k 100

python tools/dc_gradient_diagnostic.py \
  --config configs/voc/split1_1shot.yaml \
  --checkpoint "${baseline}" \
  --output outputs/diagnostics/table7/dc_gradient.json \
  --max-images 64

python tools/verify_auxiliary.py --table table7

