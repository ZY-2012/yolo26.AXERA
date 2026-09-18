#!/usr/bin/env bash
# 主机执行 5/8：用 Pulsar2 7.0 docker 编译三份配置（AX620E / NPU2）。
# 授权无需 export：优先 <repo>/.p2_home/.hasplm，其次用镜像内置 /root/*.v2c。
# 用法：bash host/05_compile.sh [u8.json [u16.json [mix.json]]]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/env.sh"
exec "${PYTHON_BIN}" "${ROOT}/scripts/compile.py" "$@"
