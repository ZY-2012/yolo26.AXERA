#!/usr/bin/env bash
# 主机执行 8/8：汇总 results/ → comparison.json 并打印对比表。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/env.sh"
"${PYTHON_BIN}" "${ROOT}/scripts/collect_results.py"
exec "${PYTHON_BIN}" "${ROOT}/scripts/summarize.py" "$@"
