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
_ULTRALYTICS_DEFAULT="/data/shared/huyuan/YOLO/ultralytics"
if [ -n "${ULTRALYTICS_REPO:-}" ]; then
  export PYTHONPATH="${ULTRALYTICS_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
elif [ -d "${_ULTRALYTICS_DEFAULT}" ]; then
  export PYTHONPATH="${_ULTRALYTICS_DEFAULT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
unset _ULTRALYTICS_DEFAULT
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${YOLO_CONFIG_DIR}"
