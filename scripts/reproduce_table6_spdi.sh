#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

base=${EADA_VOC_BASE_EXPANDED:-weights/voc_split1_all.pth}
clip=${EADA_CLIP_CHECKPOINT:-}
[[ -f "${base}" ]] || { echo "Missing ${base}" >&2; exit 3; }

prepare_checkpoint() {
  local shot=$1
  local config="configs/voc/split1_${shot}shot.yaml"
  local out="outputs/ablation/table6/${shot}shot/dc_saia/train"
  mkdir -p "${out}"
  if [[ ! -f "${out}/model_final.pth" ]]; then
    bash scripts/train.sh "${config}" \
      MODEL.WEIGHTS "${base}" \
      MODEL.EADA.SAIA_ENABLED True MODEL.EADA.DC_ENABLED True \
      MODEL.EADA.FREEZE_ROI_FEATURE True \
      DATASETS.TRAIN "('removevoc_2007_trainval_all1_${shot}shot_seed0',)" \
      SEED 0 TEST.SPDI_ENABLED False OUTPUT_DIR "${out}"
  fi
  echo "${out}/model_final.pth"
}

evaluate_mode() {
  local shot=$1 checkpoint=$2 name=$3 enabled=$4 placement=$5 fusion=$6
  local out="outputs/ablation/table6/${shot}shot/${name}"
  local config="configs/voc/split1_${shot}shot.yaml"
  mkdir -p "${out}"
  local clip_args=()
  if [[ "${enabled}" == True && -n "${clip}" ]]; then clip_args=(TEST.SPDI_CHECKPOINT "${clip}"); fi
  NUM_GPUS=1 bash scripts/eval.sh "${config}" "${checkpoint}" \
    DATASETS.TRAIN "('removevoc_2007_trainval_all1_${shot}shot_seed0',)" \
    SEED 0 TEST.SPDI_ENABLED "${enabled}" TEST.SPDI_PLACEMENT "${placement}" \
    TEST.SPDI_FUSION_MODE "${fusion}" "${clip_args[@]}" OUTPUT_DIR "${out}"
}

prepare_checkpoint 1
ckpt1=outputs/ablation/table6/1shot/dc_saia/train/model_final.pth
evaluate_mode 1 "${ckpt1}" detector_only False pre_nms absolute
evaluate_mode 1 "${ckpt1}" post_nms_spdi True post_nms absolute
evaluate_mode 1 "${ckpt1}" pre_nms_spdi True pre_nms absolute

prepare_checkpoint 10
ckpt10=outputs/ablation/table6/10shot/dc_saia/train/model_final.pth
evaluate_mode 10 "${ckpt10}" detector_only False pre_nms absolute
evaluate_mode 10 "${ckpt10}" normalized_pre_nms True pre_nms normalized
evaluate_mode 10 "${ckpt10}" absolute_pre_nms True pre_nms absolute

python tools/verify_auxiliary.py --table table6

