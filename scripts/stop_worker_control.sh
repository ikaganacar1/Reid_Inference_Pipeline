#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/runtime_env.sh
load_runtime_env

WORKER_CONTROL_PORT="${WORKER_CONTROL_PORT:-8787}"
curl -fsS -X POST "http://127.0.0.1:${WORKER_CONTROL_PORT}/control" \
    -H 'Content-Type: application/json' \
    -d '{"action":"stop"}' >/dev/null 2>&1 || true
if command -v systemctl >/dev/null 2>&1 \
    && systemctl --user is-active --quiet reid-camera.service 2>/dev/null; then
    systemctl --user stop reid-camera.service
    echo "Camera worker systemd service stopped."
    exit 0
fi
pkill -TERM -f "[s]cripts/realtime_worker_control.py" || true
echo "Worker control API and workers stop signal sent."
