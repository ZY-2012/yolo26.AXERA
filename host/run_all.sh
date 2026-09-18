#!/usr/bin/env bash
# 主机一键执行：导出 ONNX → 拆图 → 校准集 → 生成配置 → Pulsar2 量化编译。
# 前置条件：
#   1) Python 环境里有 ultralytics（导出用）
#   2) 量化前先激活客户的 Pulsar2 环境，确保 `pulsar2` 命令可用（见 05_compile.sh）
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${DIR}/01_export_onnx.sh"
bash "${DIR}/02_split_onnx.sh"
bash "${DIR}/03_prepare_calibration.sh"
bash "${DIR}/04_make_configs.sh"
bash "${DIR}/05_compile.sh"

echo
echo "全部完成：axmodel 在 $(cd "${DIR}/.." && pwd)/build/{u8,u16,mix}/ 下，同名文件也复制在 models/"
