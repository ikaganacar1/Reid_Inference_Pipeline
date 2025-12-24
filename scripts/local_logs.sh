#!/bin/bash
#
# Local Logs Viewer for ReID Pipeline
# View logs from different components
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"

# Colors
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    echo "Usage: $0 [component] [options]"
    echo ""
    echo "Components:"
    echo "  api         - API server logs"
    echo "  worker      - Worker process logs"
    echo "  pipeline    - Pipeline execution logs"
    echo "  evaluation  - Evaluation process logs"
    echo "  all         - All logs (interleaved)"
    echo ""
    echo "Options:"
    echo "  -f, --follow    Follow log output (like tail -f)"
    echo "  -n, --lines N   Show last N lines (default: 50)"
    echo ""
    echo "Examples:"
    echo "  $0 api -f           # Follow API logs"
    echo "  $0 worker -n 100    # Show last 100 worker log lines"
    echo "  $0 all              # Show recent logs from all components"
}

COMPONENT="${1:-all}"
FOLLOW=false
LINES=50

shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--follow)
            FOLLOW=true
            shift
            ;;
        -n|--lines)
            LINES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

DATE_STR=$(date +%Y%m%d)

show_log() {
    local component=$1
    local log_file="$LOG_DIR/${component}_${DATE_STR}.log"

    if [ ! -f "$log_file" ]; then
        echo "No log file for $component today"
        return 1
    fi

    echo -e "${BLUE}=== $component logs ===${NC}"

    if [ "$FOLLOW" = true ]; then
        tail -f "$log_file"
    else
        tail -n "$LINES" "$log_file"
    fi
}

case $COMPONENT in
    api|worker|pipeline|evaluation)
        show_log "$COMPONENT"
        ;;
    all)
        if [ "$FOLLOW" = true ]; then
            # Follow all logs
            tail -f "$LOG_DIR"/*_${DATE_STR}.log 2>/dev/null || echo "No log files found"
        else
            # Show recent from all
            for log in "$LOG_DIR"/*_${DATE_STR}.log; do
                if [ -f "$log" ]; then
                    component=$(basename "$log" | sed "s/_${DATE_STR}.log//")
                    echo -e "${BLUE}=== $component ===${NC}"
                    tail -n 10 "$log"
                    echo ""
                fi
            done
        fi
        ;;
    *)
        echo "Unknown component: $COMPONENT"
        usage
        exit 1
        ;;
esac
