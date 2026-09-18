#!/usr/bin/env bash
# 主机执行 1/8：下载 yolo26n.pt 并导出固定 640 ONNX。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/env.sh"
exec "${PYTHON_BIN}" "${ROOT}/scripts/export_yolo26n.py" "$@"
