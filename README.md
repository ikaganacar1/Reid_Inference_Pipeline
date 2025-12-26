# YOLO + TAO ReID Inference Pipeline

A comprehensive two-stage person re-identification pipeline integrating YOLO person detection, TAO-trained ReID models (via Triton Inference Server), and BoxMOT tracking with full experimental logging.

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────┐
│    YOLO     │───▶│  Triton ReID     │───▶│   BoxMOT     │
│  Detector   │    │  (TensorRT FP16) │    │   Tracker    │
└─────────────┘    └──────────────────┘    └──────────────┘
     │                      │                      │
     └──────────────────────┴──────────────────────┘
                            │
                   ┌────────▼────────┐
                   │  Experiment     │
                   │  Logger         │
                   └─────────────────┘
```

### Components

1. **YOLO11n Person Detector** - Detects persons and extracts crops
2. **TAO ReID Model** - Extracts 256-dim embeddings via Triton Inference Server
3. **BoxMOT (BoTSORT)** - Multi-object tracking with external ReID features
4. **Experimental Logger** - Comprehensive logging for reproducibility

## Features

- ✅ FP16 TensorRT inference for TAO ReID model
- ✅ Dynamic batching (1-16) via Triton Inference Server
- ✅ Real-time multi-object tracking with ReID features
- ✅ Comprehensive experimental logging (detections, embeddings, tracks, metrics)
- ✅ Model versioning with SHA256 hashing
- ✅ Video visualization with track IDs
- ✅ Performance metrics (FPS, latency, GPU memory)

## Quick Start

### 1. Setup Environment

```bash
# Activate conda environment
conda activate tensorrt_blackwell

# Install dependencies
pip install -r requirements.txt

# Pull Triton Docker container (if user hasn't already)
# docker pull nvcr.io/nvidia/tritonserver:25.04-py3
```

### 2. Generate TensorRT Engine

```bash
# Convert ONNX to TensorRT engine
python scripts/export_to_tensorrt.py \\
    --onnx models/lttc_0.1.4.49.onnx \\
    --output triton_models/lttc_reid/1/model.plan \\
    --config configs/reid_config.yaml
```

### 3. Start Triton Server

```bash
# Start Triton Inference Server in Docker
bash scripts/start_triton_server.sh

# Wait for server to be ready (script will wait automatically)
```

### 4. Validate Setup

```bash
# Validate all components
python scripts/validate_models.py
```

### 5. Run Pipeline

```bash
# Process a video
python main.py \\
    --video data/videos/your_video.mp4 \\
    --output data/outputs/result.mp4 \\
    --experiment-name my_test_run
```

## Directory Structure

```
Reid_Inference_Pipeline_0.2/
├── configs/                    # Configuration files
│   ├── yolo_config.yaml       # YOLO detection settings
│   ├── reid_config.yaml       # ReID + Triton settings
│   ├── tracker_config.yaml    # BoxMOT tracker settings
│   └── pipeline_config.yaml   # Pipeline orchestration
│
├── models/                     # Model files
│   ├── yolo11n.pt             # YOLO model
│   ├── lttc_0.1.4.49.pth      # TAO ReID PyTorch checkpoint
│   ├── lttc_0.1.4.49.onnx     # TAO ReID ONNX export
│   └── lttc_0.1.4.49.engine   # Original TensorRT engine
│
├── triton_models/              # Triton model repository
│   └── lttc_reid/
│       ├── config.pbtxt        # Triton model config
│       └── 1/
│           └── model.plan      # TensorRT engine (FP16, dynamic batch)
│
├── src/                        # Source code
│   ├── detector.py            # YOLO wrapper
│   ├── reid_client.py         # Triton HTTP client
│   ├── tracker.py             # BoxMOT integration
│   ├── logger.py              # Experimental logging
│   ├── pipeline.py            # Main orchestration
│   └── utils/                 # Utilities
│
├── scripts/                    # Helper scripts
│   ├── export_to_tensorrt.py  # ONNX → TensorRT converter
│   ├── setup_triton_model.py  # Triton model repo setup
│   ├── start_triton_server.sh # Triton server launcher
│   └── validate_models.py     # Model validation
│
├── data/                       # Data directories
│   ├── videos/                # Input videos
│   └── outputs/               # Output visualizations
│
├── experiments/                # Experiment logs
│   └── exp_YYYYMMDD_HHMMSS/  # Auto-generated per run
│       ├── config_snapshot.json
│       ├── detections.jsonl
│       ├── embeddings.jsonl
│       ├── tracks.jsonl
│       ├── metrics.jsonl
│       └── video_metadata.json
│
├── main.py                     # Main CLI entry point
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Configuration

### YOLO Configuration (`configs/yolo_config.yaml`)

```yaml
model:
  path: "models/yolo11n.pt"
  device: "cuda:0"

detection:
  conf_threshold: 0.5
  iou_threshold: 0.7
  classes: [0]  # Person only
  imgsz: 640
```

### ReID Configuration (`configs/reid_config.yaml`)

```yaml
triton:
  server_url: "localhost:8000"
  model_name: "lttc_reid"
  model_version: "1"

model:
  input_shape: [256, 128]  # H x W
  embedding_dim: 256

preprocessing:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

tensorrt:
  min_batch: 1
  opt_batch: 8
  max_batch: 16
  precision: "fp16"
```

### Tracker Configuration (`configs/tracker_config.yaml`)

```yaml
botsort:
  track_buffer: 30
  appearance_thresh: 0.25
  with_reid: false  # Use external TAO embeddings
```

## Usage Examples

### Basic Usage

```bash
# Process single video
python main.py --video test_video.mp4
```

### Custom Experiment Name

```bash
# Use custom experiment name
python main.py --video test_video.mp4 --experiment-name exp_yolo11n_tao_fp16
```

### Limit Frames (Testing)

```bash
# Process only first 100 frames
python main.py --video test_video.mp4 --max-frames 100
```

### No Visualization

```bash
# Skip visualization output (faster)
python main.py --video test_video.mp4 --no-visualization
```

## Experimental Logging

Each pipeline run creates a unique experiment directory with comprehensive logs:

### Log Files

1. **detections.jsonl** - Per-frame YOLO detections
2. **embeddings.jsonl** - ReID embeddings for each person
3. **tracks.jsonl** - Tracking results with track IDs
4. **metrics.jsonl** - Performance metrics (FPS, GPU memory, latency)
5. **config_snapshot.json** - Complete configuration used
6. **model_versions.json** - Model file SHA256 hashes
7. **video_metadata.json** - Input video metadata

### View Logs

```bash
# View detections
cat experiments/exp_*/detections.jsonl | jq '.'

# View tracking results
cat experiments/exp_*/tracks.jsonl | jq '.tracks'

# View performance metrics
cat experiments/exp_*/metrics.jsonl | jq '.fps'

# Get experiment summary
cat experiments/exp_*/config_snapshot.json | jq '.pipeline'
```

## Performance

### Expected Performance (RTX 5070, 12GB VRAM)

- **YOLO Detection**: ~15-20 ms per frame
- **ReID Inference (Triton)**: ~8-10 ms per batch of 8 crops
- **Tracking**: ~2-5 ms per frame
- **Overall FPS**: > 20 FPS on 640x640 videos

### Performance Tuning

1. **Batch Size**: Adjust `opt_batch` in `reid_config.yaml`
2. **YOLO Input Size**: Change `imgsz` in `yolo_config.yaml`
3. **Tracking Buffer**: Modify `track_buffer` in `tracker_config.yaml`

## Troubleshooting

### Triton Server Not Starting

```bash
# Check Docker logs
docker logs triton-reid-server

# Verify model repository
python scripts/setup_triton_model.py --validate-only

# Restart server
docker stop triton-reid-server
bash scripts/start_triton_server.sh
```

### CUDA Out of Memory

- Reduce `max_batch` in `reid_config.yaml`
- Reduce YOLO `imgsz` in `yolo_config.yaml`
- Close other GPU applications

### Low FPS

- Enable FP16 in YOLO config
- Increase Triton batch size
- Use smaller YOLO model (yolo11n vs yolo11x)

### Model Not Found Errors

```bash
# Validate all models
python scripts/validate_models.py

# Re-generate TensorRT engine
python scripts/export_to_tensorrt.py
```

## Model Files

### Required Models

1. **YOLO11n** - `models/yolo11n.pt` (already present)
2. **TAO ReID ONNX** - `models/lttc_0.1.4.49.onnx` (already present)

### Generated Files

- **TensorRT Engine** - `triton_models/lttc_reid/1/model.plan` (generated by script)

## Development

### Testing Individual Components

```bash
# Test YOLO detector
python src/detector.py

# Test ReID client (requires Triton running)
python src/reid_client.py

# Test tracker
python src/tracker.py

# Test logger
python src/logger.py
```

### Running Tests

```bash
# Validate setup
python scripts/validate_models.py

# Test pipeline (no Triton)
python scripts/validate_models.py --skip-triton
```

## Citation

If you use this pipeline in your research, please cite:

- **YOLO**: Ultralytics YOLOv11
- **TAO Toolkit**: NVIDIA TAO Toolkit
- **Triton**: NVIDIA Triton Inference Server
- **BoxMOT**: BoxMOT Multi-Object Tracking

## License

This project is provided as-is for research and development purposes.

## Support

For issues or questions:
1. Check troubleshooting section above
2. Validate setup with `python scripts/validate_models.py`
3. Review experiment logs in `experiments/` directory

## Changelog

### Version 0.2.0
- Initial implementation with Triton Inference Server
- FP16 TensorRT engine support
- BoxMOT tracking integration
- Comprehensive experimental logging
