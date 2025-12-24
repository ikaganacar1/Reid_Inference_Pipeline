#!/bin/bash
#
# Local Stop Script for ReID Pipeline
# Stops all local services
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_DIR="$PROJECT_ROOT/.pids"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  ReID Pipeline - Stopping Services${NC}"
echo -e "${BLUE}============================================${NC}"

stop_service() {
    local name=$1
    local pid_file="$PID_DIR/${name}.pid"

    if [ ! -f "$pid_file" ]; then
        echo -e "${YELLOW}[$name] Not running (no pid file)${NC}"
        return 0
    fi

    local pid=$(cat "$pid_file")

    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${GREEN}[$name] Stopping (PID: $pid)...${NC}"
        kill "$pid" 2>/dev/null

        # Wait for graceful shutdown
        local count=0
        while kill -0 "$pid" 2>/dev/null && [ $count -lt 10 ]; do
            sleep 1
            count=$((count + 1))
        done

        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}[$name] Force killing...${NC}"
            kill -9 "$pid" 2>/dev/null
        fi

        echo -e "${GREEN}[$name] Stopped${NC}"
    else
        echo -e "${YELLOW}[$name] Not running${NC}"
    fi

    rm -f "$pid_file"
}

# Stop all services
stop_service "worker"
stop_service "api"

echo ""
echo -e "${GREEN}All services stopped${NC}"
