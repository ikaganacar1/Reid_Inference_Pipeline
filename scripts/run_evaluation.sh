#!/bin/bash
#
# Run Dataset Evaluation Locally
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"

# Default values
DATASET_PATH="$PROJECT_ROOT/data"
REID_MODEL="$PROJECT_ROOT/models/lttc_0.1.4.49.onnx"
SUBSET_SIZE=""
BATCH_SIZE="32"
LOG_LEVEL="INFO"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Optional:"
    echo "  -d, --dataset PATH   Dataset path (default: ./data)"
    echo "  -m, --model FILE     ReID model path (default: models/lttc_0.1.4.49.onnx)"
    echo "  -s, --subset N       Subset size for quick testing"
    echo "  -b, --batch N        Batch size (default: 32)"
    echo "  -v, --verbose        Enable debug logging"
    echo ""
    echo "Examples:"
    echo "  $0                           # Full evaluation"
    echo "  $0 -s 100                    # Quick test with 100 samples"
    echo "  $0 -d /path/to/market1501    # Custom dataset path"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -m|--model)
            REID_MODEL="$2"
            shift 2
            ;;
        -s|--subset)
            SUBSET_SIZE="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_SIZE="$2"
            shift 2
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

# Create log directory
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/evaluation_$(date +%Y%m%d_%H%M%S).log"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Market-1501 Dataset Evaluation${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Dataset:    $DATASET_PATH"
echo -e "Model:      $REID_MODEL"
echo -e "Subset:     ${SUBSET_SIZE:-Full dataset}"
echo -e "Batch Size: $BATCH_SIZE"
echo -e "Log File:   $LOG_FILE"
echo ""

cd "$PROJECT_ROOT"

# Activate conda environment and set PYTHONPATH
echo -e "${BLUE}Activating conda environment: tensorrt_blackwell${NC}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tensorrt_blackwell
export PYTHONPATH="$PROJECT_ROOT"

# Build command
CMD="python run_local_evaluation.py"
CMD="$CMD --dataset-path \"$DATASET_PATH\""
CMD="$CMD --model \"$REID_MODEL\""
CMD="$CMD --batch-size $BATCH_SIZE"

if [ -n "$SUBSET_SIZE" ]; then
    CMD="$CMD --subset-size $SUBSET_SIZE"
fi

if [ "$LOG_LEVEL" = "DEBUG" ]; then
    CMD="$CMD --verbose"
fi

echo -e "${GREEN}Running evaluation...${NC}"
echo ""

# Run evaluation with logging
eval $CMD 2>&1 | tee "$LOG_FILE"

echo ""
echo -e "${GREEN}Evaluation completed${NC}"
echo -e "Log saved to: $LOG_FILE"
