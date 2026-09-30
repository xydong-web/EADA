#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python tools/audit_release.py
bash -n scripts/*.sh
echo "release verification: PASS"

