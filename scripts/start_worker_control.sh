#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source scripts/runtime_env.sh
load_runtime_env
runtime_select_python

RUNTIME_LOG_DIR="${RUNTIME_LOG_DIR:-outputs}"
WORKER_LOG_DIR="${RUNTIME_LOG_DIR}/realtime_worker_logs"
mkdir -p "${WORKER_LOG_DIR}"

PROFILE="${1:-${WORKER_PROFILE:-${PIPELINE_ROLE:-prime}}}"
case "${PROFILE}" in
    prime)
        export REALTIME_CONFIG="${REALTIME_CONFIG:-configs/realtime_config.yaml}"
        export YOLO_CONFIG="${YOLO_CONFIG:-configs/yolo_config.yaml}"
        ;;
    worker)
        export REALTIME_CONFIG="${REALTIME_CONFIG:-configs/realtime_config.worker.yaml}"
        export YOLO_CONFIG="${YOLO_CONFIG:-configs/yolo_config.worker.yaml}"
        ;;
    *)
        echo "ERROR: Unknown profile '${PROFILE}'. Use 'prime' or 'worker'."
        exit 2
        ;;
esac
WORKER_CONTROL_PORT="${WORKER_CONTROL_PORT:-8787}"

if pgrep -f "[s]cripts/realtime_worker_control.py" >/dev/null; then
    echo "Worker control API is already running."
    exit 0
fi

nohup "${RUNTIME_PYTHON_BIN}" -u scripts/realtime_worker_control.py \
    --config "${REALTIME_CONFIG}" \
    > "${WORKER_LOG_DIR}/control.log" 2>&1 &
echo "Worker control API started: pid=$!"
echo "Profile: ${PROFILE} realtime_config=${REALTIME_CONFIG} yolo_config=${YOLO_CONFIG}"
echo "Control URL: http://$(hostname -I | awk '{print $1}'):${WORKER_CONTROL_PORT}/status"
