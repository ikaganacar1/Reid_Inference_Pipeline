# ReID Inference Pipeline - Complete Guide

**Version:** 0.2
**Last Updated:** 2025-12-26
**Author:** Reid Inference Team

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Configuration Guide](#configuration-guide)
5. [Usage Instructions](#usage-instructions)
6. [Component Details](#component-details)
7. [Data Formats](#data-formats)
8. [Performance Tuning](#performance-tuning)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## System Overview

### What is this pipeline?

The **ReID Inference Pipeline** is a real-time person detection and tracking system that:

1. **Detects** all people in video frames using YOLO11n
2. **Extracts embeddings** using TAO ReID model (via Triton)
3. **Tracks** people across frames using BoxMOT BoTSORT
4. **Logs** all results in reproducible JSONL format
5. **Visualizes** tracks with unique IDs and bounding boxes

### Key Features

- ✅ **Real-time:** 12-20 FPS on consumer GPUs (RTX 5070, RTX 4080)
- ✅ **Modular:** Each component (YOLO, ReID, Tracker) can be replaced
- ✅ **Reproducible:** All outputs logged with model hashes and configs
- ✅ **Scalable:** Handles multiple people per frame with dynamic batching
- ✅ **Production-Ready:** Comprehensive error handling and health checks

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA RTX 3080 or better (11GB VRAM)
- CPU: 8-core modern processor
- RAM: 16 GB
- Storage: 50 GB free (models + outputs)

**Recommended:**
- GPU: NVIDIA RTX 4090 / RTX 5070 / RTX 5080 (24GB+ VRAM)
- CPU: 16-core modern processor
- RAM: 32 GB
- Storage: 200 GB free

**Software:**
- CUDA 12.x
- cuDNN 8.x
- Docker (for Triton)
- Python 3.10+
- Miniconda/Anaconda

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT VIDEO STREAM                          │
│                    (MP4, AVI, MOV, etc.)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
   │   YOLO11n   │   │   TAO ReID   │   │   BoTSORT    │
   │ Detector    │   │   (Triton)   │   │   Tracker    │
   │             │   │              │   │              │
   │ • Detects   │   │ • Extracts   │   │ • Associates │
   │   people    │   │   256-dim    │   │   detections │
   │ • Extracts  │   │   embeddings │   │ • Maintains  │
   │   crops     │   │ • ONNX model │   │   track IDs  │
   └────────┬────┘   └──────┬───────┘   └──────┬───────┘
            │                │                 │
            └────────────────┼─────────────────┘
                             │
                    ┌────────▼────────┐
                    │   VISUALIZATION │
                    │   + LOGGING     │
                    │                 │
                    │ • Draw tracks   │
                    │ • Save JSONL    │
                    │ • Log metrics   │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
  OUTPUT VIDEO        EXPERIMENT LOGS         METRICS
  (Annotated)         (JSONL format)          (Performance)
```

### Data Flow Through Pipeline

```
Frame 0 (1920x1080 BGR image)
    ↓
[YOLO Detection] (6.7 ms)
    ├─ Input: Raw frame
    ├─ Process: Forward pass + NMS
    └─ Output: 4 detections [x1,y1,x2,y2,conf,cls]

    ├─ Crop 1: [256x384] →
    ├─ Crop 2: [224x380] →
    ├─ Crop 3: [263x390] →
    └─ Crop 4: [180x400] →

[Preprocessing] (5 ms)
    └─ All crops → Resize to 384x192, BGR→RGB, normalize

[TAO ReID via Triton] (22.6 ms)
    ├─ Input: [4, 3, 384, 192] tensor
    ├─ Process: ONNX model inference on GPU
    └─ Output: [4, 256] embeddings

[BoTSORT Tracker] (3.2 ms)
    ├─ Input: Detections + embeddings + frame
    ├─ Process: Hungarian matching on IoU + embedding similarity
    └─ Output: 4 tracks with IDs [1, 2, 3, 4]

[Visualization & Logging] (15 ms)
    ├─ Draw bboxes + track IDs on frame
    ├─ Log detections.jsonl
    ├─ Log embeddings.jsonl
    ├─ Log tracks.jsonl
    └─ Write annotated frame to output video

TOTAL TIME: ~50-80 ms = 12-20 FPS
```

### Component Interactions

```
┌──────────────────────────────────┐
│   ReIDPipeline (Master Orchestrator)
│                                  │
│  process_video() {               │
│    for each frame:               │
│      - Call detector.detect()    │
│      - Call reid_client.infer()  │
│      - Call tracker.update()     │
│      - Call visualizer.draw()    │
│      - Call logger.log_*()       │
│  }                               │
└──────────────────────────────────┘
        ▲           ▲        ▲
        │           │        │
    ┌───┴────┐  ┌───┴───┐  ┌┴──────┐
    │Detector│  │ReID   │  │Tracker│
    │        │  │Client │  │       │
    │YOLO11n │  │Triton │  │BoTSORT│
    └────────┘  └───────┘  └───────┘
```

---

## Installation & Setup

### Step 1: Clone Repository & Install Dependencies

```bash
# Navigate to project directory
cd Reid_Inference_Pipeline_0.2

# Create conda environment
conda create -n reid python=3.10 -y
conda activate reid

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Models

#### YOLO Model (5.35 MB)
```bash
# Option 1: Auto-download (first run)
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Option 2: Manual download
cd models
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolo11n.pt
cd ..
```

#### TAO ReID Model (92 MB)
```bash
# Requires access to NVIDIA TAO or custom trained model
# File should be: models/lttc_0.1.4.49.onnx

# If you have NVIDIA TAO credentials:
tao export -m models/lttc_0.1.4.49.pth -k your_key -e onnx
```

### Step 3: Setup Triton Inference Server

#### Option A: Docker (Recommended)

```bash
# Pull Triton image
docker pull nvcr.io/nvidia/tritonserver:25.04-py3

# Start Triton server
bash scripts/start_triton_server.sh
```

#### Option B: Native Installation

```bash
# Install TensorRT (version must match Triton)
pip install tensorrt==8.x.x

# Copy model to Triton directory
cp models/lttc_0.1.4.49.onnx triton_models/lttc_reid/1/

# Start Triton
tritonserver --model-repository=triton_models
```

### Step 4: Verify Installation

```bash
# Validate all components
python scripts/validate_models.py

# Expected output:
# ✓ PASS: YOLO
# ✓ PASS: ONNX
# ✓ PASS: TENSORRT
# ✓ PASS: TRITON
# ✓ All validations passed!
```

---

## Configuration Guide

### Configuration Files Location

```
configs/
├── yolo_config.yaml          # YOLO detection settings
├── reid_config.yaml          # TAO ReID + Triton settings
├── tracker_config.yaml       # BoxMOT BoTSORT parameters
└── pipeline_config.yaml      # I/O and logging settings
```

### YOLO Configuration (`configs/yolo_config.yaml`)

```yaml
model:
  path: "models/yolo11n.pt"        # Model file path
  device: "cuda:0"                  # GPU device (cuda:0, cuda:1, etc.)

detection:
  conf_threshold: 0.5               # Confidence threshold (0-1)
  classes: [0]                      # Class IDs to detect (0=person)
  iou_threshold: 0.45              # NMS IoU threshold
```

**Parameters Explained:**
- `path`: Path to YOLO weights file
- `device`: GPU to use (cuda:0 for first GPU, etc.)
- `conf_threshold`: Filter detections below this confidence
  - **0.3:** More detections, more false positives
  - **0.5:** Default, balanced
  - **0.7:** Fewer detections, higher precision
- `classes`: COCO class IDs (0=person, 1=bicycle, etc.)
- `iou_threshold`: NMS overlap threshold (lower = more aggressive)

**Tuning Tips:**
```yaml
# For crowded scenes (many people)
conf_threshold: 0.4
iou_threshold: 0.3

# For outdoor scenes (fewer people)
conf_threshold: 0.6
iou_threshold: 0.5

# For real-time processing (speed over accuracy)
conf_threshold: 0.7
iou_threshold: 0.6
```

### ReID Configuration (`configs/reid_config.yaml`)

```yaml
triton:
  server_url: "localhost:8100"      # Triton HTTP endpoint
  model_name: "lttc_reid"           # Model name in Triton
  model_version: "1"                # Model version

model:
  onnx_path: "models/lttc_0.1.4.49.onnx"  # ONNX model file
  engine_path: "triton_models/lttc_reid/1/model.plan"  # TensorRT engine
  input_shape: [384, 192]           # [Height, Width]
  embedding_dim: 256                # Output embedding dimension

preprocessing:
  mean: [0.485, 0.456, 0.406]      # ImageNet normalization mean
  std: [0.229, 0.224, 0.225]       # ImageNet normalization std
  color_space: "RGB"                # Input color space
  channel_order: "CHW"              # Channel order (Channels-Height-Width)

tensorrt:
  min_batch: 1                       # Minimum batch size
  opt_batch: 8                       # Optimal batch size
  max_batch: 16                      # Maximum batch size
  precision: "fp16"                  # FP32 or FP16
  workspace_mb: 2048                 # Workspace size in MB

inference:
  max_retry: 3                       # Retry failed inferences
  timeout_ms: 5000                   # Timeout in milliseconds
```

**Parameters Explained:**
- `server_url`: Address where Triton is running
- `input_shape`: Must match ONNX model input dimensions
- `mean/std`: Preprocessing normalization (ImageNet standard)
- `precision`: FP16 for speed, FP32 for accuracy
- `opt_batch`: Should match Triton dynamic_batching preferred_batch_size

### Tracker Configuration (`configs/tracker_config.yaml`)

```yaml
botsort:
  conf_thresh: 0.3                  # Min confidence for tracking
  max_age: 30                        # Keep tracks for 30 frames without detection
  min_hits: 3                        # Need 3 detections to confirm track
  iou_threshold: 0.3                # IoU threshold for association
  track_buffer: 30                  # Track buffer size
  match_thresh: 0.8                 # Association matching threshold
  proximity_thresh: 0.5             # Spatial proximity threshold
  appearance_thresh: 0.25           # Embedding similarity threshold
  track_high_thresh: 0.5            # High confidence threshold
  track_low_thresh: 0.1             # Low confidence threshold
  new_track_thresh: 0.6             # Threshold for new tracks
```

**Parameters Explained:**
- `max_age`: How long to keep track without seeing person (30 frames @ 25 FPS = 1.2 seconds)
- `min_hits`: Track must be seen 3 times before becoming "active"
- `appearance_thresh`: How similar embeddings must be (0=different person, 1=same person)
  - **0.15:** Very strict, may lose tracks
  - **0.25:** Default, balanced
  - **0.35:** Lenient, may merge different people

**Tuning Tips:**
```yaml
# For crowded scenes (many occlusions)
max_age: 15                    # Don't keep tracks long
appearance_thresh: 0.2        # Be strict about identity

# For sparse scenes (few people)
max_age: 60                    # Keep tracks longer
appearance_thresh: 0.3        # Be more lenient

# For tracking quality
min_hits: 5                    # Require 5 detections
match_thresh: 0.9             # High matching quality
```

### Pipeline Configuration (`configs/pipeline_config.yaml`)

```yaml
io:
  save_visualization: true           # Save annotated video
  display: false                     # Show video while processing

logging:
  log_every_n_frames: 30             # Log metrics every 30 frames
  save_crops: false                  # Save person crops (uses lots of disk)
  save_embeddings: false             # Save embeddings (uses lots of disk)

processing:
  batch_size: 8                      # Batch size for ReID inference
  skip_frames: 0                     # Process every Nth frame (0=all)
```

**Parameters Explained:**
- `log_every_n_frames`: Log performance metrics less frequently to reduce I/O
- `save_crops`: Saves all person crops to disk (~100 MB per hour)
- `save_embeddings`: Saves all 256-dim embeddings (~50 MB per hour)
- `skip_frames`: Process every 5th frame for speed (processes 1/5 data)

---

## Usage Instructions

### Quick Start (5 minutes)

```bash
# 1. Ensure Triton is running
docker ps | grep triton-reid-server || bash scripts/start_triton_server.sh

# 2. Validate setup
python scripts/validate_models.py

# 3. Run pipeline
conda activate tensorrt_blackwell
python scripts/run_pipeline.py \
  --video test_videos/MOT16-02.mp4 \
  --output outputs/tracked.mp4 \
  --experiment-dir experiments/my_test

# 4. Check results
ls -lh outputs/tracked.mp4
ls -lh experiments/my_test/
```

### Command-Line Options

```bash
python scripts/run_pipeline.py \
  --video <input_video>              # Required: Input video path
  --output <output_video>            # Optional: Output annotated video
  --experiment-dir <output_dir>      # Optional: Experiment logs directory
  --max-frames <N>                   # Optional: Process only first N frames
  --display                          # Optional: Show video while processing
```

**Examples:**

```bash
# Example 1: Process entire video with all outputs
python scripts/run_pipeline.py \
  --video my_video.mp4 \
  --output my_video_tracked.mp4 \
  --experiment-dir experiments/test001

# Example 2: Quick test on first 50 frames
python scripts/run_pipeline.py \
  --video my_video.mp4 \
  --max-frames 50 \
  --experiment-dir experiments/quick_test

# Example 3: Batch processing
for video in videos/*.mp4; do
  python scripts/run_pipeline.py \
    --video "$video" \
    --output "results/$(basename $video)" \
    --experiment-dir "experiments/$(basename $video .mp4)"
done

# Example 4: Real-time monitoring (RTSP stream)
python scripts/run_pipeline.py \
  --video "rtsp://camera.local:554/stream" \
  --output live_tracking.mp4 \
  --experiment-dir experiments/live
```

### Managing Triton Server

```bash
# Start server
bash scripts/start_triton_server.sh

# Check if running
docker ps | grep triton-reid-server

# View logs
docker logs -f triton-reid-server

# Check model status
curl http://localhost:8100/v2/models/lttc_reid

# Stop server
docker stop triton-reid-server

# Remove container
docker rm triton-reid-server
```

### Working with Results

```bash
# View output video
ffplay outputs/tracked.mp4

# Get video statistics
ffprobe -v error -select_streams v:0 -show_entries \
  stream=width,height,r_frame_rate,duration outputs/tracked.mp4

# List all experiment outputs
ls experiments/test001/

# Analyze detections
python -c "
import json
with open('experiments/test001/detections.jsonl') as f:
    for line in f:
        data = json.loads(line)
        print(f'Frame {data[\"frame_idx\"]}: {data[\"num_detections\"]} people')
"

# Analyze tracks
python -c "
import json
max_track_id = 0
with open('experiments/test001/tracks.jsonl') as f:
    for line in f:
        data = json.loads(line)
        track_ids = [int(t[4]) for t in data['tracks']]
        max_track_id = max(max_track_id, max(track_ids) if track_ids else 0)
print(f'Total unique people tracked: {int(max_track_id)}')
"
```

---

## Component Details

### 1. YOLO Detector

**File:** `src/detector.py`

**Purpose:** Detect all people in video frames

**How it works:**
```
Input Frame (1920x1080)
    ↓
[YOLO Forward Pass]
    • Reduces to 416x416 internally
    • Detects objects in 3 scales
    • Applies NMS (Non-Maximum Suppression)
    ↓
[Filter for Person Class]
    • Keep only class 0 (person)
    • Filter by confidence threshold
    ↓
[Extract Crops]
    • Crop bounding boxes from original frame
    • Preserve original resolution
    ↓
Output:
    • Detections: [x1, y1, x2, y2, conf, cls]
    • Crops: List of person crop images
```

**API:**
```python
from src.detector import YOLOPersonDetector

detector = YOLOPersonDetector(config['yolo'])
detections, crops = detector.detect(frame)

# Returns:
# detections: np.ndarray [N, 6] where N = num people
# crops: list of N np.ndarray images
```

**Performance:**
- Speed: 6.7 ms per frame (RTX 5070)
- Throughput: 150 frames/second
- Model size: 5.35 MB

### 2. TAO ReID Client

**File:** `src/reid_client.py`

**Purpose:** Extract 256-dimensional embeddings for person re-identification

**How it works:**
```
Crops (Variable sizes)
    ↓
[Resize All to 384x192]
    • Bilinear interpolation
    • Maintain aspect ratio with padding
    ↓
[Convert BGR → RGB]
    • YOLO outputs BGR (OpenCV default)
    • TAO model expects RGB
    ↓
[Normalize]
    • Divide by 255.0 to get [0, 1]
    • Apply ImageNet normalization:
      (x - mean) / std
    ↓
[Convert HWC → CHW]
    • Transpose from (H, W, C) to (C, H, W)
    • Required by ONNX model
    ↓
[Batch and Send to Triton]
    • Create batch of [N, 3, 384, 192]
    • Send via HTTP REST API
    ↓
[Triton ONNX Inference]
    • Run on GPU via ONNX Runtime
    ↓
[Extract Embeddings]
    • Output: [N, 256] floating point
    • Already normalized (distance metric learned)
    ↓
Output: Embeddings [N, 256]
```

**API:**
```python
from src.reid_client import TritonReIDClient

client = TritonReIDClient(config['reid'])
embeddings = client.infer(crops)

# Returns:
# embeddings: np.ndarray [N, 256]
# where N = number of crops

# Computing similarity between embeddings:
from scipy.spatial.distance import cosine
similarity = 1 - cosine(emb1, emb2)  # Range: [0, 1]
```

**Performance:**
- Speed: 22.6 ms per batch (average 50 people)
- Throughput: ~2200 embeddings/second
- Model size: 92 MB ONNX

**Triton Configuration:**
```protobuf
# triton_models/lttc_reid/config.pbtxt
name: "lttc_reid"
platform: "onnxruntime_onnx"       # Using ONNX Runtime backend
max_batch_size: 16
input:
  name: "input"
  data_type: TYPE_FP32
  dims: [3, 384, 192]              # Dynamic batch, 3 channels, 384x192
output:
  name: "fc_pred"
  data_type: TYPE_FP32
  dims: [256]                       # 256-dim embedding
dynamic_batching:
  preferred_batch_size: [1, 4, 8]  # Optimize for these batch sizes
  max_queue_delay_microseconds: 100
```

### 3. BoTSORT Tracker

**File:** `src/tracker.py`

**Purpose:** Associate detections across frames to create persistent track IDs

**How it works:**
```
Previous Tracks (State from frame t-1):
    ├─ Track 1: bbox=[580, 445, 670, 710], emb=[...256 values...]
    ├─ Track 2: bbox=[1330, 410, 1500, 780], emb=[...256 values...]
    └─ Track 3: bbox=[1450, 430, 1600, 770], emb=[...256 values...]

Current Detections (Frame t):
    ├─ Det 1: bbox=[583, 449, 673, 708], conf=0.85, emb=[...256 values...]
    ├─ Det 2: bbox=[1335, 415, 1495, 775], conf=0.83, emb=[...256 values...]
    ├─ Det 3: bbox=[1455, 435, 1600, 770], conf=0.77, emb=[...256 values...]
    └─ Det 4: bbox=[200, 300, 250, 400], conf=0.60, emb=[...256 values...]

[Hungarian Algorithm with Combined Cost Matrix]:
    Cost = w_iou * (1 - IoU) + w_emb * (1 - cosine_sim)

    # Match old tracks to detections
    Track 1 → Det 1 (cost: 0.05)     ✓ MATCH
    Track 2 → Det 2 (cost: 0.08)     ✓ MATCH
    Track 3 → Det 3 (cost: 0.12)     ✓ MATCH
    Det 4 → No match                 → CREATE NEW TRACK 4

[Track State Updates]:
    Track 1: Update position, refresh appearance
    Track 2: Update position, refresh appearance
    Track 3: Update position, refresh appearance
    Track 4: Create new track (age=1)
    (Unmatched old tracks: increment age, remove if age > 30)

Output Tracks (Frame t):
    ├─ Track 1 (continues)
    ├─ Track 2 (continues)
    ├─ Track 3 (continues)
    └─ Track 4 (new, age=1)
```

**API:**
```python
from src.tracker import ReIDTracker

tracker = ReIDTracker(config['tracker'])
tracks = tracker.update(detections, frame, embeddings)

# Input:
#   detections: [N, 6] = [x1, y1, x2, y2, conf, cls]
#   frame: BGR image (for background subtraction if enabled)
#   embeddings: [N, 256] = embedding vectors

# Returns:
#   tracks: [M, 8] = [x1, y1, x2, y2, track_id, conf, cls, index]
#   where M = number of active tracks
```

**Matching Cost Function:**
```
cost_matrix[i, j] = w_iou * (1 - IoU(track_i, det_j))
                  + w_emb * (1 - cosine_similarity(emb_i, emb_j))

where:
    w_iou = spatial weight (default implicit)
    w_emb = appearance weight (controlled by appearance_thresh)
    IoU   = Intersection over Union (spatial overlap)
    cosine_similarity = dot product of normalized embeddings
```

**Key Parameters:**
- `max_age=30`: Track lives 30 frames without detection
- `min_hits=3`: Needs 3 detections to become confirmed
- `appearance_thresh=0.25`: Similarity threshold for matching

### 4. Logger & Visualization

**Files:** `src/logger.py`, `src/utils/visualization.py`

**Logger Output Structure:**

```
experiments/test001/
├── detections.jsonl           # Per-frame detections
├── embeddings.jsonl           # Per-detection embeddings
├── tracks.jsonl               # Per-frame tracking results
├── metrics.jsonl              # Per-frame performance metrics
├── config_snapshot.json       # All configs used
├── model_versions.json        # SHA256 hashes for reproducibility
├── system_info.json           # Hardware/software info
└── video_metadata.json        # Input video properties
```

**Sample detections.jsonl entry:**
```json
{
  "frame_idx": 0,
  "timestamp": "2025-12-26T09:32:24.480610",
  "num_detections": 4,
  "detections": [
    [582.0, 447.0, 672.0, 709.5, 0.851, 0.0],
    [1336.5, 414.75, 1498.5, 781.5, 0.831, 0.0],
    [1455.0, 433.5, 1602.0, 772.5, 0.774, 0.0],
    [455.625, 438.75, 552.375, 720.75, 0.654, 0.0]
  ],
  "inference_time_ms": 3156.74
}
```

**Visualization Features:**
- Colored bounding boxes (unique color per track ID)
- Track ID labels with confidence
- Trajectory lines (optional, if enabled)
- Per-frame metrics overlay (optional)

---

## Data Formats

### Input Video

**Supported Formats:** MP4, AVI, MOV, MKV, WEBM, RTSP streams

**Requirements:**
- Resolution: Any (internally resized for processing)
- FPS: Any
- Codec: H.264, H.265, VP8, VP9, etc.

**Example:**
```bash
# Using FFmpeg to convert
ffmpeg -i input.mov -vcodec libx264 -crf 23 -acodec aac output.mp4

# Check video properties
ffprobe -v error -select_streams v:0 -show_entries \
  stream=width,height,r_frame_rate,duration -of csv=p=0 input.mp4
```

### Output Video

**Format:** H.264 codec, MP4 container

**Properties:**
- Resolution: Same as input
- FPS: Same as input
- Codec: H.264 (libx264)
- Quality: High (CRF=20)

**File Size Estimation:**
```
1 hour of 1080p @ 30 FPS ≈ 2-3 GB
1 hour of 4K @ 30 FPS ≈ 8-12 GB
```

### JSONL Experiment Logs

**Format:** JSON Lines (one JSON object per line, newline-delimited)

**Reading in Python:**
```python
import json

def read_jsonl(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))
    return data

detections = read_jsonl('experiments/test001/detections.jsonl')
for frame_data in detections:
    print(f"Frame {frame_data['frame_idx']}: "
          f"{frame_data['num_detections']} people")
```

**File Size Estimates (per 1 hour @ 30 FPS):**
- detections.jsonl: ~100 KB (200 people/frame avg)
- embeddings.jsonl: ~200 MB (200 embeddings x 256 floats)
- tracks.jsonl: ~150 KB
- metrics.jsonl: ~2 KB

### Reproducibility Files

**model_versions.json:**
```json
{
  "yolo11n.pt": {
    "path": "models/yolo11n.pt",
    "sha256": "0ebbc80d4a7680d1d4daf7a2b2c3d4e5f6a7b8c9",
    "size_mb": 5.35
  },
  "lttc_0.1.4.49.onnx": {
    "path": "models/lttc_0.1.4.49.onnx",
    "sha256": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b",
    "size_mb": 92.07
  }
}
```

---

## Performance Tuning

### Understanding Latency

```
Per-Frame Breakdown (1080p video):

1. Video Read/Write:        ~5 ms
2. YOLO Detection:          ~6.7 ms
3. ReID Preprocessing:      ~5 ms
4. ReID Inference (Triton): ~15 ms
5. BoTSORT Tracking:        ~3.2 ms
6. Visualization:           ~5 ms
7. Logging I/O:            ~10 ms
────────────────────────────────
TOTAL:                     ~50-80 ms per frame

At 30 FPS (33 ms/frame):
- Can't keep up → Processing slower than video
- Would drop frames in real-time

Recommendations:
- Reduce video resolution (720p instead of 1080p)
- Increase detection confidence threshold
- Use skip_frames=2 (process every other frame)
- Use smaller YOLO model (yolo8n instead of yolo11n)
```

### Optimization Strategies

#### 1. Reduce Input Resolution

```bash
# Resize video to 720p before processing
ffmpeg -i input.mp4 -vf "scale=1280:720" input_720p.mp4

# Config remains the same
python scripts/run_pipeline.py --video input_720p.mp4
```

**Impact:**
- YOLO: ~20% faster
- ReID: No change (crops still resized to 384x192)
- Overall: ~15% faster

#### 2. Skip Frames

```yaml
# configs/pipeline_config.yaml
processing:
  skip_frames: 2  # Process every 3rd frame
```

**Impact:**
- Overall throughput: ~3x faster
- Tracking continuity: Slightly reduced (may lose tracks in fast motion)

#### 3. Reduce YOLO Confidence

```yaml
# configs/yolo_config.yaml
detection:
  conf_threshold: 0.6  # More strict (skip borderline detections)
```

**Impact:**
- Fewer detections → Faster ReID
- May miss some people

#### 4. Use Smaller YOLO Model

```yaml
# configs/yolo_config.yaml
model:
  path: "models/yolo8n.pt"  # Nano model (3.2 MB)
```

**Impact:**
- Detection speed: ~2x faster
- Accuracy: Slightly reduced

#### 5. Batch ReID Inference

```python
# In pipeline.py, instead of:
embeddings = self.reid_client.infer(crops)

# Batch in groups:
batch_size = 16
embeddings = []
for i in range(0, len(crops), batch_size):
    batch = crops[i:i+batch_size]
    batch_embs = self.reid_client.infer(batch)
    embeddings.append(batch_embs)
embeddings = np.vstack(embeddings)
```

**Impact:**
- Triton dynamic batching more effective
- ~10% faster inference

#### 6. Disable Expensive Logging

```yaml
# configs/pipeline_config.yaml
logging:
  save_embeddings: false  # Don't save 256-dim vectors
  save_crops: false       # Don't save person crops
  log_every_n_frames: 60  # Log less frequently
```

**Impact:**
- Disk I/O: ~50% less
- Overall: ~5% faster

### GPU Memory Optimization

```python
# If GPU memory error:

# 1. Reduce batch size
configs['tracker']['max_batch_size'] = 8  # was 16

# 2. Clear cache
import torch
torch.cuda.empty_cache()

# 3. Use FP16 (already configured)
# configs/reid_config.yaml: precision: "fp16"

# 4. Process smaller chunks
python scripts/run_pipeline.py --max-frames 100  # Process in batches
```

### Benchmarking

```bash
# Profile a single component
python -c "
import time
import cv2
import numpy as np
from src.detector import YOLOPersonDetector
import yaml

with open('configs/yolo_config.yaml') as f:
    config = yaml.safe_load(f)

detector = YOLOPersonDetector(config['yolo'])
frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

times = []
for i in range(100):
    t0 = time.time()
    detections, crops = detector.detect(frame)
    times.append((time.time() - t0) * 1000)

print(f'YOLO Detection:')
print(f'  Mean: {np.mean(times):.2f} ms')
print(f'  Std:  {np.std(times):.2f} ms')
print(f'  Min:  {np.min(times):.2f} ms')
print(f'  Max:  {np.max(times):.2f} ms')
"
```

---

## Troubleshooting

### Common Issues

#### 1. Triton Server Won't Start

**Error:** `docker: command not found`

```bash
# Solution: Install Docker
# Ubuntu/Debian
sudo apt-get install docker.io docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
docker ps
```

**Error:** `Bind for 0.0.0.0:8100 failed: port is already allocated`

```bash
# Solution: Find and stop process using port
lsof -i :8100
kill -9 <PID>

# Or use different port
docker run -p 8200:8000 -p 8201:8001 ...
# Update configs/reid_config.yaml:
# server_url: "localhost:8200"
```

**Error:** `GPU not available in container`

```bash
# Solution: Install nvidia-docker
docker run --rm --gpus all nvidia/cuda:12.2.0-runtime-ubuntu22.04 \
  nvidia-smi

# Or use nvidia-docker
nvidia-docker run --rm nvcr.io/nvidia/tritonserver:25.04-py3 \
  nvidia-smi
```

#### 2. YOLO Model Not Found

**Error:** `FileNotFoundError: models/yolo11n.pt`

```bash
# Solution: Download model
cd models
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolo11n.pt
cd ..

# Or auto-download on first use
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

#### 3. ReID Model Connection Failed

**Error:** `urllib.error.URLError: <urlopen error Connection refused>`

```bash
# Solution: Start Triton server
bash scripts/start_triton_server.sh

# Verify it's running
curl http://localhost:8100/v2/health/ready
docker ps | grep triton

# Check logs
docker logs triton-reid-server
```

**Error:** `Model 'lttc_reid' is not loaded`

```bash
# Solution: Check model file exists
ls -la triton_models/lttc_reid/1/

# Should have:
# - model.onnx (92 MB)
# - config.pbtxt

# Verify config
cat triton_models/lttc_reid/config.pbtxt

# Restart Triton
docker stop triton-reid-server
bash scripts/start_triton_server.sh
```

#### 4. CUDA Out of Memory

**Error:** `CUDA out of memory. Tried to allocate 2.00 GiB`

```bash
# Solution 1: Reduce input resolution
ffmpeg -i input.mp4 -vf "scale=1280:720" input_720p.mp4

# Solution 2: Process fewer frames at once
python scripts/run_pipeline.py --video input.mp4 --max-frames 100

# Solution 3: Reduce batch size
# configs/reid_config.yaml:
# max_batch_size: 8  (was 16)

# Solution 4: Use smaller YOLO model
# configs/yolo_config.yaml:
# path: "models/yolo8n.pt"
```

#### 5. Output Video Looks Corrupted

**Error:** `FFmpeg error when writing video`

```bash
# Solution: Check codec support
ffmpeg -codecs | grep h264

# Or use different codec
# In src/pipeline.py, change:
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # instead of 'mp4v'

# Or convert output
ffmpeg -i corrupted.mp4 -vcodec libx264 -crf 20 output.mp4
```

#### 6. Embeddings Are All Zeros

**Error:** `embeddings = np.zeros((N, 256))`

```bash
# Solution: Verify TAO model is loaded
curl http://localhost:8100/v2/models/lttc_reid

# Check Triton logs
docker logs triton-reid-server | grep "lttc_reid"

# Verify ONNX file
python -c "
import onnx
model = onnx.load('models/lttc_0.1.4.49.onnx')
print(f'Inputs: {[i.name for i in model.graph.input]}')
print(f'Outputs: {[o.name for o in model.graph.output]}')
"
```

#### 7. Slow Processing Speed

**Error:** `Overall FPS: 2.5` (too slow)

```bash
# Solution: Profile to find bottleneck
python scripts/run_pipeline.py --video input.mp4 --max-frames 10

# Check output metrics:
# Which component takes most time?
# - YOLO detection: 6.7 ms
# - ReID inference: 22.6 ms (likely culprit)
# - Tracking: 3.2 ms
# - I/O: variable

# If ReID slow:
# 1. Check GPU utilization: nvidia-smi -l 1
# 2. Verify Triton is using GPU: docker exec triton-reid-server nvidia-smi
# 3. Consider TensorRT engine (needs compatible TensorRT)
```

### Debugging Commands

```bash
# Check GPU status
nvidia-smi
nvidia-smi -l 1  # Update every 1 second

# Check Docker container resource usage
docker stats triton-reid-server

# View Triton metrics
curl http://localhost:8102/metrics

# Check Python memory usage
python -c "
import psutil
import os
pid = os.getpid()
p = psutil.Process(pid)
print(f'Memory: {p.memory_info().rss / 1024**2:.1f} MB')
"

# Validate video file
ffprobe -v error -show_format -show_streams input.mp4

# Check model file integrity
md5sum models/yolo11n.pt
md5sum models/lttc_0.1.4.49.onnx
```

---

## Advanced Usage

### Custom Detector Implementation

```python
# Create custom detector (e.g., using different YOLO version)

from src.detector import YOLOPersonDetector
import numpy as np

class CustomDetector(YOLOPersonDetector):
    def detect(self, frame):
        # Your custom detection code
        detections = np.array([...])  # [N, 6]
        crops = [...]  # List of crop images
        return detections, crops

# In scripts/run_pipeline.py:
# Replace:
#   self.detector = YOLOPersonDetector(configs['yolo'])
# With:
#   self.detector = CustomDetector(configs['yolo'])
```

### Custom Embedding Model

```python
# Create custom ReID client (e.g., using different TAO model)

from src.reid_client import TritonReIDClient
import numpy as np

class CustomReIDClient(TritonReIDClient):
    def infer(self, crops, retry=3):
        # Your custom embedding extraction
        embeddings = np.array([...])  # [N, 256] or [N, D]
        return embeddings

# In scripts/run_pipeline.py:
# Replace:
#   self.reid_client = TritonReIDClient(configs['reid'])
# With:
#   self.reid_client = CustomReIDClient(configs['reid'])
```

### Custom Tracker

```python
# Create custom tracker (e.g., using different algorithm)

from src.tracker import ReIDTracker
import numpy as np

class CustomTracker(ReIDTracker):
    def update(self, detections, frame, embeddings):
        # Your custom tracking algorithm
        tracks = np.array([...])  # [M, 8]
        return tracks

# In scripts/run_pipeline.py:
# Replace:
#   self.tracker = ReIDTracker(configs['tracker'])
# With:
#   self.tracker = CustomTracker(configs['tracker'])
```

### Streaming Output

```python
# Stream results instead of saving video

def stream_results(video_path):
    pipeline = ReIDPipeline(configs, experiment_dir)

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, crops = pipeline.detector.detect(frame)
        embeddings = pipeline.reid_client.infer(crops)
        tracks = pipeline.tracker.update(detections, frame, embeddings)

        frame_annotated = pipeline.visualizer.draw_tracks(frame, tracks)

        # Stream to client (e.g., HTTP, RTMP)
        yield frame_annotated
```

### Batch Processing

```bash
#!/bin/bash
# Process multiple videos in parallel

videos=(
    "video1.mp4"
    "video2.mp4"
    "video3.mp4"
    "video4.mp4"
)

for video in "${videos[@]}"; do
    (
        python scripts/run_pipeline.py \
            --video "$video" \
            --output "results/${video%.mp4}_tracked.mp4" \
            --experiment-dir "experiments/${video%.mp4}"
    ) &

    # Limit to 2 parallel processes
    if (( $(jobs -r | wc -l) >= 2 )); then
        wait -n
    fi
done

wait
echo "All videos processed"
```

### Post-Processing Analysis

```python
# Analyze tracking results

import json
import numpy as np

def analyze_experiment(exp_dir):
    # Load all results
    with open(f'{exp_dir}/tracks.jsonl') as f:
        tracks_data = [json.loads(line) for line in f]

    with open(f'{exp_dir}/metrics.jsonl') as f:
        metrics_data = [json.loads(line) for line in f]

    # Compute statistics
    total_tracks = set()
    total_detections = 0
    track_lengths = {}

    for frame_data in tracks_data:
        for track in frame_data['tracks']:
            track_id = int(track[4])
            total_tracks.add(track_id)
            track_lengths[track_id] = track_lengths.get(track_id, 0) + 1

        total_detections += frame_data['num_tracks']

    # Print results
    print(f"Total Unique People: {len(total_tracks)}")
    print(f"Total Track Events: {total_detections}")
    print(f"Average Track Length: {np.mean(list(track_lengths.values())):.1f} frames")
    print(f"Max Track Length: {max(track_lengths.values())} frames")

    # Performance
    fps_values = [m['fps'] for m in metrics_data if 'fps' in m]
    print(f"\nPerformance:")
    print(f"  Average FPS: {np.mean(fps_values):.1f}")
    print(f"  Min FPS: {np.min(fps_values):.1f}")
    print(f"  Max FPS: {np.max(fps_values):.1f}")

# Usage
analyze_experiment('experiments/test001')
```

---

## FAQ

**Q: Can I use a different YOLO version?**
A: Yes. Update `configs/yolo_config.yaml` with path to any Ultralytics YOLO model.

**Q: Can I use a different ReID model?**
A: Yes. Convert your model to ONNX format and place in `models/`. Update `configs/reid_config.yaml` with correct paths and input/output shapes.

**Q: How do I deploy to production?**
A: Use Docker Compose to orchestrate Triton + application containers.

**Q: Can I process 4K video?**
A: Yes, but processing will be slower (4x resolution = slower YOLO). Consider using smaller YOLO model.

**Q: How do I handle occlusions?**
A: Current tracker loses tracks when occluded. Use longer `max_age` in config to recover faster.

**Q: Can I track other classes besides people?**
A: Yes. Change `classes: [0]` in `configs/yolo_config.yaml` to other COCO class IDs.

---

## Support & References

### Documentation
- YOLO: https://docs.ultralytics.com/
- Triton: https://docs.nvidia.com/deeplearning/triton-inference-server/
- BoxMOT: https://github.com/mikel-brostrom/yolo_tracking

### GitHub Issues
Report bugs: [Your repo URL]

### Contact
For TAO ReID models: [Your contact]

---

**Document Version:** 1.0
**Last Updated:** 2025-12-26
**Maintainers:** Reid Inference Team
