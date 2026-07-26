#!/usr/bin/env bash
set -euo pipefail

if command -v systemctl >/dev/null 2>&1 \
    && systemctl --user is-active --quiet reid-prime.service 2>/dev/null; then
    systemctl --user stop reid-prime.service
    echo "Prime systemd service stopped."
    exit 0
fi

pkill -TERM -f "[s]cripts/realtime_prime.py" || true
echo "Prime dashboard stop signal sent."
