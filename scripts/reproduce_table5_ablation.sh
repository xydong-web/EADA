#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

base=${EADA_VOC_BASE_EXPANDED:-weights/voc_split1_all.pth}
clip=${EADA_CLIP_CHECKPOINT:-}
[[ -f "${base}" ]] || { echo "Missing ${base}" >&2; exit 3; }

train_variant() {
  local shot=$1 name=$2 saia=$3 dc=$4
  local config="configs/voc/split1_${shot}shot.yaml"
  local out="outputs/ablation/table5/${shot}shot/${name}/train"
  mkdir -p "${out}"
  if [[ ! -f "${out}/model_final.pth" ]]; then
    bash scripts/train.sh "${config}" \
      MODEL.WEIGHTS "${base}" \
      MODEL.EADA.SAIA_ENABLED "${saia}" \
      MODEL.EADA.DC_ENABLED "${dc}" \
      MODEL.EADA.FREEZE_ROI_FEATURE True \
      DATASETS.TRAIN "('removevoc_2007_trainval_all1_${shot}shot_seed0',)" \
      SEED 0 TEST.SPDI_ENABLED False OUTPUT_DIR "${out}"
  fi
}

eval_variant() {
  local shot=$1 row=$2 checkpoint=$3 spdi=$4 saia=$5
  local config="configs/voc/split1_${shot}shot.yaml"
  local out="outputs/ablation/table5/${shot}shot/${row}/eval"
  mkdir -p "${out}"
  local clip_args=()
  if [[ "${spdi}" == True && -n "${clip}" ]]; then clip_args=(TEST.SPDI_CHECKPOINT "${clip}"); fi
  NUM_GPUS=1 bash scripts/eval.sh "${config}" "${checkpoint}" \
    MODEL.EADA.SAIA_ENABLED "${saia}" \
    DATASETS.TRAIN "('removevoc_2007_trainval_all1_${shot}shot_seed0',)" \
    SEED 0 TEST.SPDI_ENABLED "${spdi}" TEST.SPDI_PLACEMENT pre_nms \
    TEST.SPDI_FUSION_MODE absolute "${clip_args[@]}" OUTPUT_DIR "${out}"
}

for shot in ${EADA_ABLATION_SHOTS:-1 3 5}; do
  train_variant "${shot}" baseline False False
  train_variant "${shot}" dc False True
  train_variant "${shot}" saia True False
  train_variant "${shot}" dc_saia True True

  base_ckpt="outputs/ablation/table5/${shot}shot/baseline/train/model_final.pth"
  dc_ckpt="outputs/ablation/table5/${shot}shot/dc/train/model_final.pth"
  saia_ckpt="outputs/ablation/table5/${shot}shot/saia/train/model_final.pth"
  full_ckpt="outputs/ablation/table5/${shot}shot/dc_saia/train/model_final.pth"

  eval_variant "${shot}" baseline "${base_ckpt}" False False
  eval_variant "${shot}" dc "${dc_ckpt}" False False
  eval_variant "${shot}" saia "${saia_ckpt}" False True
  eval_variant "${shot}" dc_saia "${full_ckpt}" False True
  eval_variant "${shot}" spdi "${base_ckpt}" True False
  eval_variant "${shot}" full_eada "${full_ckpt}" True True
done

python tools/verify_auxiliary.py --table table5

