#!/bin/bash
#
# Stop Local UI Services
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_DIR="$PROJECT_ROOT/.pids"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping Local UI Services...${NC}"

# Stop API
if [ -f "$PID_DIR/api.pid" ]; then
    PID=$(cat "$PID_DIR/api.pid")
    if kill $PID 2>/dev/null; then
        echo -e "${GREEN}Stopped API (PID: $PID)${NC}"
    fi
    rm -f "$PID_DIR/api.pid"
fi

# Stop Worker
if [ -f "$PID_DIR/worker.pid" ]; then
    PID=$(cat "$PID_DIR/worker.pid")
    if kill $PID 2>/dev/null; then
        echo -e "${GREEN}Stopped Worker (PID: $PID)${NC}"
    fi
    rm -f "$PID_DIR/worker.pid"
fi

# Stop Frontend
if [ -f "$PID_DIR/frontend.pid" ]; then
    PID=$(cat "$PID_DIR/frontend.pid")
    if kill $PID 2>/dev/null; then
        echo -e "${GREEN}Stopped Frontend (PID: $PID)${NC}"
    fi
    rm -f "$PID_DIR/frontend.pid"
fi

# Also kill any lingering processes
pkill -f "uvicorn local_main:app" 2>/dev/null
pkill -f "local_worker.py" 2>/dev/null

echo -e "${GREEN}All services stopped${NC}"
