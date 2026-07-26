#!/usr/bin/env bash
set -euo pipefail

curl -fsS -X POST http://127.0.0.1:8787/control \
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
