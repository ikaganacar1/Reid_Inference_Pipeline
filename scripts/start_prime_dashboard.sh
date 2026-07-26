#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

JETSON_ENV_FILE="${JETSON_ENV_FILE:-${HOME}/.config/reid-pipeline/jetson.env}"
if [ -r "${JETSON_ENV_FILE}" ]; then
    set -a
    source "${JETSON_ENV_FILE}"
    set +a
fi

mkdir -p outputs/realtime

if pgrep -f "[s]cripts/realtime_prime.py" >/dev/null; then
    echo "Prime dashboard is already running."
    exit 0
fi

PYTHON_RUNNER=()
DIRECT_PYTHON=""
JETSON_VENV="${JETSON_VENV:-$(pwd)/.venv-jetson}"
REALTIME_CONFIG="${REALTIME_CONFIG:-configs/realtime_config.yaml}"
CONFIG_DIR="${CONFIG_DIR:-configs}"

if [ -n "${PYTHON_BIN:-}" ]; then
    DIRECT_PYTHON="${PYTHON_BIN}"
elif [ -x "${JETSON_VENV}/bin/python" ]; then
    DIRECT_PYTHON="${JETSON_VENV}/bin/python"
    echo "Using Jetson virtual environment: ${JETSON_VENV}"
elif [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
    source scripts/onnxruntime_cuda_env.sh
    if CONDA_PREFIX_PATH="$(conda_env_prefix "${CONDA_ENV}" 2>/dev/null)"; then
        CONDA_SITE_PACKAGES="$(conda_env_site_packages "${CONDA_ENV}")"
        prepend_onnxruntime_cuda_ld_library_path "${CONDA_PREFIX_PATH}" "${CONDA_SITE_PACKAGES}"
        PYTHON_RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" env -u LD_PRELOAD LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" python)
        echo "Using conda env for prime dashboard: ${CONDA_ENV}"
    fi
fi
if [ "${#PYTHON_RUNNER[@]}" -eq 0 ]; then
    DIRECT_PYTHON="${DIRECT_PYTHON:-python3}"
    PYTHON_RUNNER=(scripts/with_onnxruntime_cuda_env.sh "${DIRECT_PYTHON}")
fi

nohup "${PYTHON_RUNNER[@]}" -u scripts/realtime_prime.py \
    --config "${REALTIME_CONFIG}" \
    --config-dir "${CONFIG_DIR}" \
    > outputs/realtime/prime.log 2>&1 &
echo "Prime dashboard started: pid=$!"
echo "Dashboard URL: http://$(hostname -I | awk '{print $1}'):8765/"
