#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

shot=${1:?Usage: bash scripts/run_coco_cell.sh SHOT SEED}
seed=${2:?Usage: bash scripts/run_coco_cell.sh SHOT SEED}
[[ "${shot}" =~ ^(1|2|3|5|10|30)$ ]] || { echo "shot must be 1/2/3/5/10/30" >&2; exit 2; }
[[ "${seed}" =~ ^[012]$ ]] || { echo "paper reproduction uses seed 0/1/2" >&2; exit 2; }

config="configs/coco/${shot}shot.yaml"
base=${EADA_COCO_BASE_EXPANDED:-weights/coco_all.pth}
clip=${EADA_CLIP_CHECKPOINT:-}
run="outputs/coco/${shot}shot/seed${seed}"
train_out="${run}/train"
eval_out="${run}/eval"
train_dataset="removecoco14_trainval_all_${shot}shot_seed${seed}"
test_dataset="coco14_test_all"

[[ -f "${base}" ]] || {
  echo "Missing ${base}. Run: NUM_GPUS=4 bash scripts/train_coco_base.sh" >&2
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
  "dataset": "MS COCO",
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

