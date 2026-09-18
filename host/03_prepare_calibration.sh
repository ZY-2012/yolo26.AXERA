#!/usr/bin/env bash
# 主机执行 3/8：生成 dataset/images100.tar（100 张真实图，首次自动下载 coco128）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/env.sh"
exec "${PYTHON_BIN}" "${ROOT}/scripts/prepare_calibration.py" "$@"
