#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"

PYTHONPATH="$SCRIPT_DIR/.." python /home/ika/yzlm/Reid_Inference_Pipeline/reid_pipeline/multi_camera_pipeline.py \
  --videos \
    test_videos/test_video_1.mp4 \
    test_videos/test_video_2.mp4 \
    test_videos/test_video_3.mp4 \
    test_videos/test_video_4.mp4 \
  --output multicam_output.mp4 \
  --yolo yolo11n.pt \
  --reid reid_ltcc.engine \
  --device cuda \
  --conf 0.3 \
  --match-thresh 0.5 \
  --new-thresh 0.7 \
  --tensorrt \
  --display-scale 0.2