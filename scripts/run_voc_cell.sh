#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

split=${1:?Usage: bash scripts/run_voc_cell.sh SPLIT SHOT SEED}
shot=${2:?Usage: bash scripts/run_voc_cell.sh SPLIT SHOT SEED}
seed=${3:?Usage: bash scripts/run_voc_cell.sh SPLIT SHOT SEED}
[[ "${split}" =~ ^[123]$ ]] || { echo "split must be 1/2/3" >&2; exit 2; }
[[ "${shot}" =~ ^(1|2|3|5|10)$ ]] || { echo "shot must be 1/2/3/5/10" >&2; exit 2; }
[[ "${seed}" =~ ^[012]$ ]] || { echo "paper reproduction uses seed 0/1/2" >&2; exit 2; }

config="configs/voc/split${split}_${shot}shot.yaml"
base=${EADA_VOC_BASE_EXPANDED:-weights/voc_split${split}_all.pth}
clip=${EADA_CLIP_CHECKPOINT:-}
run="outputs/voc/split${split}/${shot}shot/seed${seed}"
train_out="${run}/train"
eval_out="${run}/eval"
train_dataset="removevoc_2007_trainval_all${split}_${shot}shot_seed${seed}"
test_dataset="voc_2007_test_all${split}"

[[ -f "${base}" ]] || {
  echo "Missing ${base}. Run: NUM_GPUS=4 bash scripts/train_voc_base.sh ${split}" >&2
  exit 3
}

mkdir -p "${train_out}" "${eval_out}"
if [[ ! -f "${train_out}/model_final.pth" ]]; then
  bash scripts/train.sh "${config}" \
    MODEL.WEIGHTS "${base}" \
    DATASETS.TRAIN "('${train_dataset}',)" \
    DATASETS.TEST "('${test_dataset}',)" \
    SEED "${seed}" \
    TEST.SPDI_ENABLED False \
    OUTPUT_DIR "${train_out}"
else
  echo "reuse ${train_out}/model_final.pth"
fi

clip_args=()
if [[ -n "${clip}" ]]; then
  clip_args=(TEST.SPDI_CHECKPOINT "${clip}")
fi
NUM_GPUS=1 bash scripts/eval.sh "${config}" "${train_out}/model_final.pth" \
  DATASETS.TRAIN "('${train_dataset}',)" \
  DATASETS.TEST "('${test_dataset}',)" \
  SEED "${seed}" \
  TEST.SPDI_ENABLED True \
  TEST.SPDI_PLACEMENT pre_nms \
  TEST.SPDI_FUSION_MODE absolute \
  "${clip_args[@]}" \
  OUTPUT_DIR "${eval_out}"

cat > "${run}/run_protocol.json" <<EOF
{
  "dataset": "PASCAL VOC",
  "split": ${split},
  "shot": ${shot},
  "seed": ${seed},
  "train_dataset": "${train_dataset}",
  "test_dataset": "${test_dataset}",
  "saia": true,
  "dc": true,
  "spdi": true,
  "spdi_placement": "pre_nms",
  "spdi_fusion": "absolute"
}
EOF

echo "results=${eval_out}/evaluation_results.json"

