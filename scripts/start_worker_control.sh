#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

JETSON_ENV_FILE="${JETSON_ENV_FILE:-${HOME}/.config/reid-pipeline/jetson.env}"
if [ -r "${JETSON_ENV_FILE}" ]; then
    set -a
    source "${JETSON_ENV_FILE}"
    set +a
fi

mkdir -p outputs/realtime_worker_logs

PROFILE="${1:-${WORKER_PROFILE:-prime}}"
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
JETSON_VENV="${JETSON_VENV:-$(pwd)/.venv-jetson}"
if [ -z "${PYTHON_BIN:-}" ] && [ -x "${JETSON_VENV}/bin/python" ]; then
    PYTHON_BIN="${JETSON_VENV}/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
export PYTHON_BIN

if pgrep -f "[s]cripts/realtime_worker_control.py" >/dev/null; then
    echo "Worker control API is already running."
    exit 0
fi

nohup "${PYTHON_BIN}" -u scripts/realtime_worker_control.py \
    --config "${REALTIME_CONFIG}" \
    > outputs/realtime_worker_logs/control.log 2>&1 &
echo "Worker control API started: pid=$!"
echo "Profile: ${PROFILE} realtime_config=${REALTIME_CONFIG} yolo_config=${YOLO_CONFIG}"
echo "Control URL: http://$(hostname -I | awk '{print $1}'):8787/status"
