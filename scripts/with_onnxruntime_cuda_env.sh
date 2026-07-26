#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 /path/to/python <python arguments...>" >&2
    exit 2
fi

PYTHON_BIN="$1"
shift

cd "$(dirname "$0")/.."
source scripts/onnxruntime_cuda_env.sh

PYTHON_PREFIX="$("${PYTHON_BIN}" -c 'import sys; from pathlib import Path; print(Path(sys.executable).resolve().parents[1])')"
SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
prepend_onnxruntime_cuda_ld_library_path "${PYTHON_PREFIX}" "${SITE_PACKAGES}"
unset LD_PRELOAD || true

exec "${PYTHON_BIN}" "$@"
