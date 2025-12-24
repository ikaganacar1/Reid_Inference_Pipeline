#!/bin/bash
#
# Start Local UI - API, Worker, and Frontend
# No Docker required - uses SQLite and in-memory job queue
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
echo -e "${BLUE}  Starting Local ReID Pipeline UI${NC}"
echo -e "${BLUE}============================================${NC}"

# Create directories
mkdir -p "$PID_DIR" "$LOG_DIR"

# Activate conda environment
echo -e "${BLUE}Activating conda environment: tensorrt_blackwell${NC}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tensorrt_blackwell
export PYTHONPATH="$PROJECT_ROOT"

cd "$PROJECT_ROOT"

# Kill existing processes
if [ -f "$PID_DIR/api.pid" ]; then
    OLD_PID=$(cat "$PID_DIR/api.pid")
    kill $OLD_PID 2>/dev/null && echo -e "${YELLOW}Killed old API process${NC}"
fi

if [ -f "$PID_DIR/worker.pid" ]; then
    OLD_PID=$(cat "$PID_DIR/worker.pid")
    kill $OLD_PID 2>/dev/null && echo -e "${YELLOW}Killed old worker process${NC}"
fi

if [ -f "$PID_DIR/frontend.pid" ]; then
    OLD_PID=$(cat "$PID_DIR/frontend.pid")
    kill $OLD_PID 2>/dev/null && echo -e "${YELLOW}Killed old frontend process${NC}"
fi

sleep 1

# Start API
echo -e "${GREEN}Starting API server on port 8000...${NC}"
cd "$PROJECT_ROOT/services/api"
python -m uvicorn local_main:app --host 0.0.0.0 --port 8000 --reload > "$LOG_DIR/api_local.log" 2>&1 &
API_PID=$!
echo $API_PID > "$PID_DIR/api.pid"
echo -e "  API PID: $API_PID"

# Wait for API to start
sleep 3

# Check if API is running
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo -e "  ${GREEN}API is healthy${NC}"
else
    echo -e "  ${RED}API failed to start - check $LOG_DIR/api_local.log${NC}"
fi

# Start Worker
echo -e "${GREEN}Starting Pipeline Worker...${NC}"
cd "$PROJECT_ROOT"
python services/pipeline_worker/local_worker.py > "$LOG_DIR/worker_local.log" 2>&1 &
WORKER_PID=$!
echo $WORKER_PID > "$PID_DIR/worker.pid"
echo -e "  Worker PID: $WORKER_PID"

# Start Frontend (if npm is available)
if command -v npm &> /dev/null; then
    echo -e "${GREEN}Starting Frontend on port 8009...${NC}"
    cd "$PROJECT_ROOT/services/frontend"

    # Use clean environment wrapper to avoid conda variable conflicts
    ./start-clean.sh > "$LOG_DIR/frontend_local.log" 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$PID_DIR/frontend.pid"
    echo -e "  Frontend PID: $FRONTEND_PID"
else
    echo -e "${YELLOW}npm not found - skipping frontend${NC}"
    echo -e "${YELLOW}You can access the API directly at http://localhost:8000/docs${NC}"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Services Started${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "  API:       http://localhost:8000"
echo -e "  API Docs:  http://localhost:8000/docs"
echo -e "  Frontend:  http://localhost:8009"
echo ""
echo -e "  Logs:"
echo -e "    API:      $LOG_DIR/api_local.log"
echo -e "    Worker:   $LOG_DIR/worker_local.log"
echo -e "    Frontend: $LOG_DIR/frontend_local.log"
echo ""
echo -e "  To stop: ./scripts/stop_local_ui.sh"
echo ""
