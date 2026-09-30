#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA_ROOT=${DETECTRON2_DATASETS:-${ROOT}/datasets}
COPY_SPLITS=${COPY_SPLITS:-1}

mkdir -p "${DATA_ROOT}/vocsplit" "${DATA_ROOT}/cocosplit"

install_seed() {
  local source=$1
  local target=$2
  if [[ -e "${target}" || -L "${target}" ]]; then
    echo "keep existing ${target}"
    return
  fi
  if [[ "${COPY_SPLITS}" == 1 ]]; then
    cp -a "${source}" "${target}"
  else
    ln -s "${source}" "${target}"
  fi
  echo "installed ${target}"
}

for seed in 0 1 2; do
  install_seed "${ROOT}/splits/voc/seed${seed}" "${DATA_ROOT}/vocsplit/seed${seed}"
  install_seed "${ROOT}/splits/coco/seed${seed}" "${DATA_ROOT}/cocosplit/seed${seed}"
done

echo "DETECTRON2_DATASETS=${DATA_ROOT}"

