#!/usr/bin/env bash
set -euo pipefail

# Start camera workers on one Jetson.
# Override sources by passing arguments, e.g.:
#   scripts/start_4_realtime_workers.sh /dev/video0 /dev/video1 /dev/video2 /dev/video3
# Stable camera IDs can be supplied as camera_id=source:
#   scripts/start_4_realtime_workers.sh cam1=/dev/video0 cam2=/dev/video2

cd "$(dirname "$0")/.."

sources=("${@}")
REALTIME_CONFIG="${REALTIME_CONFIG:-configs/realtime_config.yaml}"
YOLO_CONFIG="${YOLO_CONFIG:-configs/yolo_config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ "${#sources[@]}" -eq 0 ]; then
    sources=(0 1 2 3)
fi

if [ "${#sources[@]}" -lt 1 ]; then
    echo "Usage: $0 [source1 source2 ...]"
    exit 2
fi

mkdir -p outputs/realtime_worker_logs

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
    log_path="outputs/realtime_worker_logs/${camera_id}.log"

    echo "Starting ${camera_id} source=${source} log=${log_path}"
    "${PYTHON_BIN}" -u scripts/realtime_worker.py \
        --config "${REALTIME_CONFIG}" \
        --yolo-config "${YOLO_CONFIG}" \
        --camera-id "${camera_id}" \
        --source "${source}" \
        > "${log_path}" 2>&1 &
done

echo "Workers started. PIDs:"
jobs -p
wait
