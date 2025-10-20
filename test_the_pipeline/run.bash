#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory so relative paths work
cd "$SCRIPT_DIR"

# Run the main.py with proper Python path
PYTHONPATH="$SCRIPT_DIR/.." python ../reid_pipeline/main.py run \
  --config custom_config_fixed.yaml \
  --input MOT16-05.mp4 \
  --output result_fixed.mp4 \
  --log-level INFO