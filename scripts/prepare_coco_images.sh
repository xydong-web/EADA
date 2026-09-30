#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA_ROOT=${DETECTRON2_DATASETS:?Set DETECTRON2_DATASETS first}
COCO_ROOT=${DATA_ROOT}/coco
TRAIN=${COCO_ROOT}/train2014
VAL=${COCO_ROOT}/val2014
TARGET=${COCO_ROOT}/trainval2014

[[ -d "${TRAIN}" ]] || { echo "Missing ${TRAIN}" >&2; exit 2; }
[[ -d "${VAL}" ]] || { echo "Missing ${VAL}" >&2; exit 2; }
mkdir -p "${TARGET}"

python - "${TRAIN}" "${VAL}" "${TARGET}" <<'PY'
import os
import sys
from pathlib import Path

sources = [Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()]
target = Path(sys.argv[3]).resolve()
created = existing = 0
for source in sources:
    for image in source.glob("*.jpg"):
        destination = target / image.name
        if destination.exists() or destination.is_symlink():
            existing += 1
            continue
        os.symlink(image, destination)
        created += 1
print(f"created={created} existing={existing} target={target}")
PY

echo "COCO trainval2014 image view prepared."

