#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DETECTRON2_DATASETS:?Set DETECTRON2_DATASETS first}
OUT=${DATA_ROOT}/cocosplit/datasplit
mkdir -p "${OUT}"

BASE_URL=${EADA_FSDET_SPLIT_URL:-http://dl.yf.io/fs-det/datasets/cocosplit/datasplit}

download() {
  local name=$1
  local url=${BASE_URL}/${name}
  local dst=${OUT}/${name}
  if [[ -f "${dst}" ]]; then
    echo "keep existing ${dst}"
    return
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "${dst}.part" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${dst}.part" "${url}"
  else
    echo "curl or wget is required" >&2
    exit 2
  fi
  mv "${dst}.part" "${dst}"
}

download trainvalno5k.json
download 5k.json

python "$(dirname "$0")/../tools/check_data.py" --dataset coco

