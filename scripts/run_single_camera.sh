#!/bin/bash
#
# Run Single Camera Pipeline Locally
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"

# Default values
INPUT_VIDEO=""
OUTPUT_VIDEO=""
REID_MODEL="$PROJECT_ROOT/models/lttc_0.1.4.49.onnx"
YOLO_MODEL="$PROJECT_ROOT/models/yolo11n.pt"
DISPLAY="true"
LOG_LEVEL="INFO"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    echo "Usage: $0 -i INPUT [-o OUTPUT] [options]"
    echo ""
    echo "Required:"
    echo "  -i, --input FILE     Input video file or camera index (0, 1, etc.)"
    echo ""
    echo "Optional:"
    echo "  -o, --output FILE    Output video file"
    echo "  -m, --model FILE     ReID model path (default: models/lttc_0.1.4.49.onnx)"
    echo "  -y, --yolo FILE      YOLO model path (default: models/yolo11n.pt)"
    echo "  --no-display         Disable display window"
    echo "  -v, --verbose        Enable debug logging"
    echo ""
    echo "Examples:"
    echo "  $0 -i uploads/video.mp4 -o output.mp4"
    echo "  $0 -i 0  # Use webcam"
    echo "  $0 -i video.mp4 --no-display -o result.avi"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_VIDEO="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_VIDEO="$2"
            shift 2
            ;;
        -m|--model)
            REID_MODEL="$2"
            shift 2
            ;;
        -y|--yolo)
            YOLO_MODEL="$2"
            shift 2
            ;;
        --no-display)
            DISPLAY="false"
            shift
            ;;
        -v|--verbose)
            LOG_LEVEL="DEBUG"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Validate input
if [ -z "$INPUT_VIDEO" ]; then
    echo -e "${RED}Error: Input video is required${NC}"
    usage
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Single Camera Pipeline${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Input:      $INPUT_VIDEO"
echo -e "Output:     ${OUTPUT_VIDEO:-None}"
echo -e "ReID Model: $REID_MODEL"
echo -e "YOLO Model: $YOLO_MODEL"
echo -e "Display:    $DISPLAY"
echo -e "Log File:   $LOG_FILE"
echo ""

cd "$PROJECT_ROOT"

# Activate conda environment and set PYTHONPATH
echo -e "${BLUE}Activating conda environment: tensorrt_blackwell${NC}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tensorrt_blackwell
export PYTHONPATH="$PROJECT_ROOT"

# Build command
CMD="python reid_pipeline/main.py run"
CMD="$CMD --input \"$INPUT_VIDEO\""
CMD="$CMD --reid-model \"$REID_MODEL\""
CMD="$CMD --yolo-model \"$YOLO_MODEL\""
CMD="$CMD --log-level $LOG_LEVEL"
CMD="$CMD --log-file \"$LOG_FILE\""

if [ -n "$OUTPUT_VIDEO" ]; then
    CMD="$CMD --output \"$OUTPUT_VIDEO\""
fi

if [ "$DISPLAY" = "false" ]; then
    CMD="$CMD --no-display"
fi

echo -e "${GREEN}Running...${NC}"
echo ""

# Run pipeline
eval $CMD

echo ""
echo -e "${GREEN}Pipeline completed${NC}"
echo -e "Log saved to: $LOG_FILE"
