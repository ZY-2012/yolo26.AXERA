#!/usr/bin/env bash
# 主机执行 4/8：从客户 exp_e3_mix.json 生成 u8 / u16 / mix 三份 Pulsar2 配置。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/env.sh"
exec "${PYTHON_BIN}" "${ROOT}/scripts/make_configs.py" "$@"
