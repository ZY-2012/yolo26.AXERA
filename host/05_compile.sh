#!/usr/bin/env bash
# 主机执行 5/5：量化编译（AX620E / NPU2），产出 models/*.axmodel。
#
# 重要：请先激活客户的 Pulsar2 环境，确保 `pulsar2` 命令可用，例如：
#   source /path/to/pulsar2_env/bin/activate     # Python venv / conda
#   source /path/to/npu_dev                      # 或客户自己的环境激活脚本
# 然后执行： bash host/05_compile.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v pulsar2 >/dev/null 2>&1; then
    echo "未找到 pulsar2：请先激活 Pulsar2 环境再执行本脚本" >&2
    exit 1
fi

mkdir -p "${ROOT}/logs" "${ROOT}/models"
cd "${ROOT}/configs"

for name in u8 u16 mix; do
    echo "=== 量化编译 ${name} ==="
    pulsar2 build --config "${name}.json" 2>&1 | tee "${ROOT}/logs/${name}.log"
done

cp "${ROOT}/build/u8/yolo26n_u8.axmodel" \
   "${ROOT}/build/u16/yolo26n_u16.axmodel" \
   "${ROOT}/build/mix/yolo26n_e3_mix.axmodel" \
   "${ROOT}/models/"

echo
echo "axmodel 已生成：${ROOT}/models/"
ls -l "${ROOT}/models/"
