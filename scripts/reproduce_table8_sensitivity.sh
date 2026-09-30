#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

base=${EADA_VOC_BASE_EXPANDED:-weights/voc_split1_all.pth}
clip=${EADA_CLIP_CHECKPOINT:-}
[[ -f "${base}" ]] || { echo "Missing ${base}" >&2; exit 3; }

for alpha in 0.1 0.2 0.3 0.4 0.5; do
  out="outputs/sensitivity/saia_alpha/${alpha}"
  mkdir -p "${out}/train" "${out}/eval"
  if [[ ! -f "${out}/train/model_final.pth" ]]; then
    bash scripts/train.sh configs/voc/split1_1shot.yaml \
      MODEL.WEIGHTS "${base}" \
      MODEL.EADA.SAIA_ENABLED True MODEL.EADA.DC_ENABLED True \
      MODEL.EADA.SAIA_ALPHA "${alpha}" MODEL.EADA.FREEZE_ROI_FEATURE True \
      DATASETS.TRAIN "('removevoc_2007_trainval_all1_1shot_seed0',)" \
      SEED 0 TEST.SPDI_ENABLED False OUTPUT_DIR "${out}/train"
  fi
  NUM_GPUS=1 bash scripts/eval.sh configs/voc/split1_1shot.yaml "${out}/train/model_final.pth" \
    DATASETS.TRAIN "('removevoc_2007_trainval_all1_1shot_seed0',)" \
    SEED 0 TEST.SPDI_ENABLED False OUTPUT_DIR "${out}/eval"
done

ckpt=outputs/ablation/table6/10shot/dc_saia/train/model_final.pth
if [[ ! -f "${ckpt}" ]]; then
  bash scripts/reproduce_table6_spdi.sh
fi

eval_spdi() {
  local group=$1 value=$2
  shift 2
  local out="outputs/sensitivity/${group}/${value}"
  mkdir -p "${out}"
  local clip_args=()
  if [[ -n "${clip}" ]]; then clip_args=(TEST.SPDI_CHECKPOINT "${clip}"); fi
  NUM_GPUS=1 bash scripts/eval.sh configs/voc/split1_10shot.yaml "${ckpt}" \
    DATASETS.TRAIN "('removevoc_2007_trainval_all1_10shot_seed0',)" \
    SEED 0 TEST.SPDI_ENABLED True TEST.SPDI_PLACEMENT pre_nms \
    TEST.SPDI_FUSION_MODE absolute "${clip_args[@]}" "$@" OUTPUT_DIR "${out}"
}

for value in 0.5 0.6 0.7 0.8 0.9 1.0; do
  eval_spdi lambda "${value}" TEST.SPDI_LAMBDA_NOVEL "${value}"
done
for value in 0.005 0.01 0.02 0.05; do
  eval_spdi temperature "${value}" TEST.SPDI_TEMPERATURE "${value}"
done
for value in 25 50 100 200; do
  eval_spdi topk "${value}" TEST.SPDI_TOPK "${value}"
done

python tools/verify_auxiliary.py --table table8

