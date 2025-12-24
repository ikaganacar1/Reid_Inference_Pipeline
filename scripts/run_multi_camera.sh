#!/bin/bash
#
# Run Multi Camera Pipeline Locally (4 streams)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"

# Default values
REID_MODEL="$PROJECT_ROOT/models/lttc_0.1.4.49.onnx"
YOLO_MODEL="$PROJECT_ROOT/models/yolo11n.pt"
OUTPUT_VIDEO=""
DISPLAY_SCALE="0.5"
LOG_LEVEL="INFO"
ENABLE_DISPLAY="true"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    echo "Usage: $0 VIDEO1 VIDEO2 VIDEO3 VIDEO4 [options]"
    echo ""
    echo "Required:"
    echo "  VIDEO1-4             Four input video files or camera indices"
    echo ""
    echo "Optional:"
    echo "  -o, --output FILE    Output video file (.avi format recommended)"
    echo "  -m, --model FILE     ReID model path (default: models/lttc_0.1.4.49.onnx)"
    echo "  -y, --yolo FILE      YOLO model path (default: models/yolo11n.pt)"
    echo "  -s, --scale FLOAT    Display scale (default: 0.5)"
    echo "  --no-display         Disable display window (headless mode)"
    echo "  -v, --verbose        Enable debug logging"
    echo ""
    echo "Examples:"
    echo "  $0 v1.mp4 v2.mp4 v3.mp4 v4.mp4 -o multi_output.avi"
    echo "  $0 0 1 2 3  # Use 4 webcams"
}

# Parse positional arguments (video files)
VIDEOS=()
while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
    VIDEOS+=("$1")
    shift
done

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        -s|--scale)
            DISPLAY_SCALE="$2"
            shift 2
            ;;
        --no-display)
            ENABLE_DISPLAY="false"
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

# Validate inputs
if [ ${#VIDEOS[@]} -ne 4 ]; then
    echo -e "${RED}Error: Exactly 4 video inputs required, got ${#VIDEOS[@]}${NC}"
    usage
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/multi_camera_$(date +%Y%m%d_%H%M%S).log"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Multi Camera Pipeline (4 Streams)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Video 1:    ${VIDEOS[0]}"
echo -e "Video 2:    ${VIDEOS[1]}"
echo -e "Video 3:    ${VIDEOS[2]}"
echo -e "Video 4:    ${VIDEOS[3]}"
echo -e "Output:     ${OUTPUT_VIDEO:-None}"
echo -e "ReID Model: $REID_MODEL"
echo -e "YOLO Model: $YOLO_MODEL"
echo -e "Scale:      $DISPLAY_SCALE"
echo -e "Log File:   $LOG_FILE"
echo ""

cd "$PROJECT_ROOT"

# Activate conda environment and set PYTHONPATH
echo -e "${BLUE}Activating conda environment: tensorrt_blackwell${NC}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tensorrt_blackwell
export PYTHONPATH="$PROJECT_ROOT"

# Build command
CMD="python reid_pipeline/multi_camera_pipeline.py"
CMD="$CMD --videos \"${VIDEOS[0]}\" \"${VIDEOS[1]}\" \"${VIDEOS[2]}\" \"${VIDEOS[3]}\""
CMD="$CMD --reid \"$REID_MODEL\""
CMD="$CMD --yolo \"$YOLO_MODEL\""
CMD="$CMD --display-scale $DISPLAY_SCALE"

if [ -n "$OUTPUT_VIDEO" ]; then
    CMD="$CMD --output \"$OUTPUT_VIDEO\""
fi

if [ "$ENABLE_DISPLAY" = "false" ]; then
    CMD="$CMD --no-display"
fi

echo -e "${GREEN}Running...${NC}"
echo ""

# Run pipeline with logging
eval $CMD 2>&1 | tee "$LOG_FILE"

echo ""
echo -e "${GREEN}Pipeline completed${NC}"
echo -e "Log saved to: $LOG_FILE"
