#!/bin/bash
#
# Local Start Script for ReID Pipeline
# Starts all local services (API, Worker) without Docker
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
PID_DIR="$PROJECT_ROOT/.pids"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directories
mkdir -p "$LOG_DIR" "$PID_DIR"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  ReID Pipeline - Local Startup${NC}"
echo -e "${BLUE}============================================${NC}"

# Check conda environment
CONDA_ENV="${CONDA_DEFAULT_ENV:-}"
if [ "$CONDA_ENV" != "tensorrt_blackwell" ]; then
    echo -e "${YELLOW}Warning: Expected conda environment 'tensorrt_blackwell', got '$CONDA_ENV'${NC}"
    echo -e "${YELLOW}Please run: conda activate tensorrt_blackwell${NC}"
fi

# Function to start a service
start_service() {
    local name=$1
    local command=$2
    local log_file="$LOG_DIR/${name}_$(date +%Y%m%d).log"
    local pid_file="$PID_DIR/${name}.pid"

    # Check if already running
    if [ -f "$pid_file" ]; then
        local old_pid=$(cat "$pid_file")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo -e "${YELLOW}[$name] Already running (PID: $old_pid)${NC}"
            return 0
        fi
    fi

    echo -e "${GREEN}[$name] Starting...${NC}"
    cd "$PROJECT_ROOT"

    # Start in background with logging
    nohup $command >> "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$pid_file"

    sleep 1

    if kill -0 $pid 2>/dev/null; then
        echo -e "${GREEN}[$name] Started (PID: $pid)${NC}"
        echo -e "${BLUE}[$name] Logs: $log_file${NC}"
    else
        echo -e "${RED}[$name] Failed to start${NC}"
        return 1
    fi
}

# Start API server
echo ""
echo -e "${BLUE}Starting API Server...${NC}"
start_service "api" "python -m uvicorn services.api.main:app --host 0.0.0.0 --port 8000"

# Start Worker (optional - for background job processing)
echo ""
echo -e "${BLUE}Starting Worker...${NC}"
start_service "worker" "python services/pipeline_worker/worker.py"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  All Services Started${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "API Server:  http://localhost:8000"
echo -e "API Docs:    http://localhost:8000/docs"
echo ""
echo -e "Logs:        $LOG_DIR/"
echo -e "Stop:        ./scripts/local_stop.sh"
echo -e "Status:      ./scripts/local_status.sh"
echo ""
