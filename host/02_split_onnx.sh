#!/usr/bin/env bash
# 主机执行 2/8：one2one 头按 P3/P4/P5 切成 output_split_0/1/2，对齐 exp_e3_mix.json 的 layer 名。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/env.sh"
exec "${PYTHON_BIN}" "${ROOT}/scripts/build_split_onnx.py" "$@"
