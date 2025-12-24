#!/bin/bash
#
# Local Status Script for ReID Pipeline
# Shows status of all local services
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_DIR="$PROJECT_ROOT/.pids"
LOG_DIR="$PROJECT_ROOT/logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  ReID Pipeline - Service Status${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

check_service() {
    local name=$1
    local port=$2
    local pid_file="$PID_DIR/${name}.pid"

    printf "%-15s" "[$name]"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}Running${NC} (PID: $pid)"

            # Check port if specified
            if [ -n "$port" ]; then
                if netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port "; then
                    echo -e "               Listening on port $port"
                fi
            fi
        else
            echo -e "${RED}Stopped${NC} (stale pid file)"
        fi
    else
        echo -e "${YELLOW}Not running${NC}"
    fi
}

check_service "api" "8000"
check_service "worker" ""

echo ""
echo -e "${BLUE}Log Files:${NC}"
echo "----------------------------------------"

if [ -d "$LOG_DIR" ]; then
    for log in "$LOG_DIR"/*.log; do
        if [ -f "$log" ]; then
            size=$(du -h "$log" | cut -f1)
            lines=$(wc -l < "$log")
            echo -e "  $(basename $log): $size ($lines lines)"
        fi
    done
else
    echo "  No log files found"
fi

echo ""
echo -e "${BLUE}Quick Commands:${NC}"
echo "  View API logs:    tail -f logs/api_$(date +%Y%m%d).log"
echo "  View Worker logs: tail -f logs/worker_$(date +%Y%m%d).log"
echo ""
