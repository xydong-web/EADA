#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

if [[ -f weights/coco_base_mask_init.pth ]]; then
  default_base_mask=weights/coco_base_mask_init.pth
else
  default_base_mask=outputs/fsis/base_mask/model_final.pth
fi
base_mask=${EADA_BASE_MASK_CHECKPOINT:-${default_base_mask}}
[[ -f "${base_mask}" ]] || {
  echo "Missing ${base_mask}. Extract the reviewer initialization bundle or run: NUM_GPUS=4 bash scripts/train_coco_base_mask.sh" >&2
  exit 3
}

declare -A selected=( [1]=0 [2]=2 [3]=2 [5]=2 [10]=2 [30]=1 )
clip=${EADA_CLIP_CHECKPOINT:-}

for shot in ${EADA_COCO_SHOTS:-1 2 3 5 10 30}; do
  seed=${selected[${shot}]}
  detector="outputs/coco/${shot}shot/seed${seed}/train/model_final.pth"
  [[ -f "${detector}" ]] || {
    echo "Missing selected detector ${detector}; reproducing it first"
    bash scripts/run_coco_cell.sh "${shot}" "${seed}"
  }
  run="outputs/fsis/${shot}shot/seed${seed}"
  merged="${run}/merged_detector_mask.pth"
  train_out="${run}/train"
  eval_out="${run}/eval"
  mkdir -p "${run}" "${train_out}" "${eval_out}"

  if [[ ! -f "${merged}" ]]; then
    python tools/merge_fsis_checkpoint.py "${detector}" "${base_mask}" "${merged}"
  fi
  if [[ ! -f "${train_out}/model_final.pth" ]]; then
    bash scripts/train_fsis.sh "configs/coco/fsis/${shot}shot.yaml" "${merged}" \
      DATASETS.TRAIN "('removecoco14_trainval_all_${shot}shot_seed${seed}',)" \
      SEED "${seed}" \
      TEST.SPDI_ENABLED False \
      OUTPUT_DIR "${train_out}"
  fi
  python tools/verify_fsis_freeze.py "${merged}" "${train_out}/model_final.pth"

  clip_args=()
  if [[ -n "${clip}" ]]; then clip_args=(TEST.SPDI_CHECKPOINT "${clip}"); fi
  NUM_GPUS=1 bash scripts/eval.sh "configs/coco/fsis/${shot}shot.yaml" "${train_out}/model_final.pth" \
    DATASETS.TRAIN "('removecoco14_trainval_all_${shot}shot_seed${seed}',)" \
    SEED "${seed}" \
    TEST.SPDI_ENABLED True \
    TEST.SPDI_PLACEMENT pre_nms \
    TEST.SPDI_FUSION_MODE absolute \
    "${clip_args[@]}" \
    OUTPUT_DIR "${eval_out}"
done

python tools/verify_auxiliary.py --table table3

