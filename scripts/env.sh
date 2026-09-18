#!/usr/bin/env bash
# Workdir-local env: keep every temp/cache path inside the project, never /tmp.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BENCH_ROOT="${ROOT}"
export WORK_TMP="${ROOT}/.work_tmp"
export TMPDIR="${WORK_TMP}/tmp"
export MPLCONFIGDIR="${WORK_TMP}/matplotlib"
export XDG_CACHE_HOME="${WORK_TMP}/xdg_cache"
export XDG_CONFIG_HOME="${WORK_TMP}/xdg_config"
export YOLO_CONFIG_DIR="${WORK_TMP}/ultralytics"
# 导出/拆图需要 ultralytics：pip install ultralytics 或设置 ULTRALYTICS_REPO=/path/to/ultralytics
if [ -n "${ULTRALYTICS_REPO:-}" ]; then
  export PYTHONPATH="${ULTRALYTICS_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${YOLO_CONFIG_DIR}"
