#!/usr/bin/env bash
set -euo pipefail

# Start camera workers on one Jetson.
# Override sources by passing arguments, e.g.:
#   scripts/start_4_realtime_workers.sh /dev/video0 /dev/video1 /dev/video2 /dev/video3
# Stable camera IDs can be supplied as camera_id=source:
#   scripts/start_4_realtime_workers.sh cam1=/dev/video0 cam2=/dev/video2

cd "$(dirname "$0")/.."

source scripts/runtime_env.sh
load_runtime_env
runtime_select_python

sources=("${@}")
if [ "${#sources[@]}" -eq 0 ]; then
    if [ -z "${CAMERA_SOURCES:-}" ]; then
        echo "ERROR: No sources were supplied and CAMERA_SOURCES is empty."
        echo "Use scripts/start_worker_control.sh for automatic camera scanning."
        exit 2
    fi
    IFS=',' read -r -a configured_sources <<< "${CAMERA_SOURCES}"
    IFS=',' read -r -a configured_ids <<< "${CAMERA_IDS:-}"
    if [ "${#configured_sources[@]}" -ne "${#configured_ids[@]}" ]; then
        echo "ERROR: CAMERA_SOURCES and CAMERA_IDS must contain the same number of entries."
        exit 2
    fi
    for idx in "${!configured_sources[@]}"; do
        sources+=("${configured_ids[$idx]}=${configured_sources[$idx]}")
    done
fi

if [ "${PIPELINE_ROLE:-prime}" = "worker" ]; then
    REALTIME_CONFIG="${REALTIME_CONFIG:-configs/realtime_config.worker.yaml}"
    YOLO_CONFIG="${YOLO_CONFIG:-configs/yolo_config.worker.yaml}"
else
    REALTIME_CONFIG="${REALTIME_CONFIG:-configs/realtime_config.yaml}"
    YOLO_CONFIG="${YOLO_CONFIG:-configs/yolo_config.yaml}"
fi

RUNTIME_LOG_DIR="${RUNTIME_LOG_DIR:-outputs}"
WORKER_LOG_DIR="${RUNTIME_LOG_DIR}/realtime_worker_logs"
mkdir -p "${WORKER_LOG_DIR}"

for idx in "${!sources[@]}"; do
    camera_num=$((idx + 1))
    spec="${sources[$idx]}"
    if [[ "${spec}" =~ ^[A-Za-z][A-Za-z0-9_-]*= ]]; then
        camera_id="${spec%%=*}"
        source="${spec#*=}"
    else
        camera_id="cam${camera_num}"
        source="${spec}"
    fi
    log_path="${WORKER_LOG_DIR}/${camera_id}.log"

    display_source="$(printf '%s' "${source}" | sed -E 's#(://)[^/@]+@#\1***@#')"
    echo "Starting ${camera_id} source=${display_source} log=${log_path}"
    "${RUNTIME_PYTHON_BIN}" -u scripts/realtime_worker.py \
        --config "${REALTIME_CONFIG}" \
        --yolo-config "${YOLO_CONFIG}" \
        --camera-id "${camera_id}" \
        --source "${source}" \
        > "${log_path}" 2>&1 &
done

echo "Workers started. PIDs:"
jobs -p
wait
