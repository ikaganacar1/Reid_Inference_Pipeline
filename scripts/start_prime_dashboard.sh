#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source scripts/runtime_env.sh
load_runtime_env
runtime_select_python

RUNTIME_LOG_DIR="${RUNTIME_LOG_DIR:-outputs}"
PRIME_LOG_DIR="${RUNTIME_LOG_DIR}/realtime"
mkdir -p "${PRIME_LOG_DIR}"

if pgrep -f "[s]cripts/realtime_prime.py" >/dev/null; then
    echo "Prime dashboard is already running."
    exit 0
fi

REALTIME_CONFIG="${REALTIME_CONFIG:-configs/realtime_config.yaml}"
CONFIG_DIR="${CONFIG_DIR:-configs}"
PRIME_PORT="${PRIME_PORT:-8765}"

nohup scripts/with_onnxruntime_cuda_env.sh "${RUNTIME_PYTHON_BIN}" \
    -u scripts/realtime_prime.py \
    --config "${REALTIME_CONFIG}" \
    --config-dir "${CONFIG_DIR}" \
    > "${PRIME_LOG_DIR}/prime.log" 2>&1 &
echo "Prime dashboard started: pid=$!"
echo "Dashboard URL: http://$(hostname -I | awk '{print $1}'):${PRIME_PORT}/"
