#!/usr/bin/env bash
# 主机执行 6/8：生成板端用 640×640 U8 NHWC 输入 artifacts/input_640.npy。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/env.sh"
exec "${PYTHON_BIN}" "${ROOT}/scripts/prepare_input.py" "$@"
