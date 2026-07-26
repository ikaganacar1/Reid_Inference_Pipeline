# YOLO + TAO ReID Inference Pipeline

A person re-identification pipeline integrating YOLO person detection,
TAO-trained ReID models, BoxMOT tracking, synchronized offline replay, and a
distributed realtime Jetson deployment.

> **Realtime Jetson path:** the active prime uses ONNX Runtime CUDA directly,
> not Triton. Start with [docs/JETSON_DEPLOYMENT.md](docs/JETSON_DEPLOYMENT.md).
> Triton remains available for legacy/offline configurations only.

## Architecture

```
Camera Orin: camera -> YOLO26m -> detections/crops -> WebSocket
                                                      |
Prime Orin:  ONNX Runtime CUDA ReID -> per-camera BoTSORT
                                      -> global identity gallery
                                      -> dashboard + segmented recordings
```

### Components

1. **YOLO26m Person Detector** - Detects persons and extracts crops
2. **Generalized Swin ReID Model** - Extracts 1024-dim embeddings through direct ONNX Runtime CUDA in realtime
3. **BoxMOT (BoTSORT)** - Multi-object tracking with appearance-based re-identification
4. **Experimental Logger** - Comprehensive logging for reproducibility

## Features

- Real-time person re-identification across camera exits/entries
- GPU ONNX Runtime inference for the active ReID model
- Bounded frame-level ReID request chunking
- Appearance-based matching for long-term tracking
- Bounded local lost-track retention
- Comprehensive experimental logging (detections, embeddings, tracks, metrics)
- Model versioning with SHA256 hashing
- Video visualization with track IDs
- Dataset evaluation tools (mAP, CMC metrics)

## Quick Start

### 1. Setup Environment

```bash
# Activate conda environment
conda activate tensorrt_blackwell

# Install dependencies
pip install -r requirements.txt
```

This quick start is for a workstation. Jetsons must use the role-specific
installer in [docs/JETSON_DEPLOYMENT.md](docs/JETSON_DEPLOYMENT.md) so generic
pip packages do not replace JetPack's CUDA-enabled PyTorch and OpenCV.

### 2. Optional Legacy Triton Server

```bash
# Only for a config whose ReID backend is explicitly set to Triton.
bash scripts/start_triton_server.sh

# Verify model is loaded
curl http://localhost:8100/v2/models/swin_base_reid
```

### 3. Validate Setup

```bash
# Validate all components
python scripts/validate_models.py
```

### 4. Run Pipeline

```bash
# Process a video
python main.py \
    --video data/videos/your_video.mp4 \
    --experiment-name my_test_run
```

## Directory Structure

```
Reid_Inference_Pipeline/
├── configs/                    # Configuration files
│   ├── yolo_config.yaml       # YOLO detection settings
│   ├── reid_config.yaml       # ReID backend and preprocessing
│   ├── tracker_config.yaml    # BoxMOT tracker settings
│   ├── pipeline_config.yaml   # Pipeline orchestration
│   └── evaluation_config.yaml # Dataset evaluation settings
│
├── deploy/                     # Jetson service templates and model manifest
├── triton_models/              # Optional legacy Triton model repository
│
├── src/                        # Source code
│   ├── detector.py            # YOLO wrapper
│   ├── reid_client.py         # ReID backend factory / Triton client
│   ├── onnx_reid_client.py    # Active direct ONNX Runtime backend
│   ├── realtime/              # Distributed worker, prime, and global IDs
│   ├── tracker.py             # BoxMOT integration with external ReID
│   ├── logger.py              # Experimental logging
│   ├── pipeline.py            # Main orchestration
│   ├── evaluation/            # Dataset evaluation module
│   │   ├── dataset.py         # Market1501/LTCC dataset loader
│   │   ├── metrics.py         # CMC and mAP computation
│   │   └── evaluator.py       # Main evaluator
│   └── utils/                 # Utilities
│
├── scripts/                    # Helper scripts
│   ├── export_to_tensorrt.py  # ONNX → TensorRT converter
│   ├── setup_triton_model.py  # Triton model repo setup
│   ├── start_triton_server.sh # Triton server launcher
│   ├── validate_models.py     # Model validation
│   ├── import_model.py        # Import new ReID models
│   ├── evaluate_dataset.py    # Dataset evaluation
│   └── benchmark_triton_model.py # Performance benchmarking
│
├── docs/                       # Documentation
│   ├── IMPORTING_MODELS.md    # Guide for importing new models
│   └── MODEL_IMPORT_QUICKSTART.md # Quick reference
│
├── experiments/                # Experiment logs
│   └── <experiment_name>/     # Per-run results
│
├── main.py                     # Main CLI entry point
├── requirements.txt            # Workstation dependencies
├── requirements_*_jetson.txt   # Jetson role dependencies
└── README.md                   # This file
```

## Configuration

### ReID Configuration (`configs/reid_config.yaml`)

```yaml
backend: "onnxruntime_direct"

model:
  onnx_path: "TwinProject_models/reid_generalized_yolo11n/generalized_reid_swin_epoch119.onnx"
  input_shape: [256, 128]  # H x W
  embedding_dim: 1024

preprocessing:
  mean: [0.5, 0.5, 0.5]
  std: [0.5, 0.5, 0.5]

onnxruntime:
  providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

### Tracker Configuration (`configs/tracker_config.yaml`)

```yaml
botsort:
  # Re-identification settings
  with_reid: true           # Enable appearance-based matching
  proximity_thresh: 0.5     # Require spatial support for local matching
  appearance_thresh: 0.3    # Strict local embedding distance
  track_buffer: 30          # About 3 seconds at the 10 FPS target

  # Standard tracking parameters
  track_high_thresh: 0.5
  track_low_thresh: 0.1
  new_track_thresh: 0.5
  match_thresh: 0.8
```

**Key settings for re-identification:**
- `with_reid: true` - Enables appearance-based matching using ReID embeddings
- `proximity_thresh: 0.5` - Keeps local association spatially constrained
- `track_buffer: 30` - Retains a lost local track for about three seconds at 10 FPS
- `appearance_thresh: 0.3` - Cosine distance threshold (higher is more lenient)

Long-gap and cross-camera identity are handled by the global gallery, not by
keeping stale local motion tracks alive.

## Usage Examples

### Basic Usage

```bash
# Process single video
python main.py --video test_video.mp4
```

### Realtime Multi-Jetson Mode

The current deployment uses one camera on each of two Orin devices. The same
subsystem can be extended with globally unique camera IDs:

```bash
# Prime Orin: centralized ReID, tracking, recording, LAN viewer, and cam1
scripts/start_prime_dashboard.sh
scripts/start_worker_control.sh prime

# Worker Orin: cam2
scripts/start_worker_control.sh worker
```

See [docs/REALTIME_PIPELINE.md](docs/REALTIME_PIPELINE.md) for the network
layout, worker commands, browser viewer, and recording behavior. Use
[docs/JETSON_DEPLOYMENT.md](docs/JETSON_DEPLOYMENT.md) for installation,
preflight, model staging, and boot services.

### Custom Experiment Name

```bash
python main.py --video test_video.mp4 --experiment-name my_experiment
```

### Limit Frames (Testing)

```bash
python main.py --video test_video.mp4 --max-frames 100
```

### No Visualization (Faster)

```bash
python main.py --video test_video.mp4 --no-visualization
```

## Dataset Evaluation

Evaluate ReID model performance on Market1501-format datasets:

```bash
# Evaluate on dataset
python scripts/evaluate_dataset.py --data-root data --experiment-name eval_run

# Re-evaluate from saved embeddings (faster)
python scripts/evaluate_dataset.py --from-embeddings experiments/evaluation/eval_run
```

**Metrics:**
- mAP (mean Average Precision) - Primary retrieval metric
- CMC (Cumulative Matching Characteristics) - Rank-1, 5, 10, 20 accuracy

## Importing New Models

Import new ReID models to Triton:

```bash
# Automated import with TensorRT conversion
python scripts/import_model.py \
    --onnx models/new_model.onnx \
    --model-name new_model \
    --test \
    --benchmark
```

See `docs/IMPORTING_MODELS.md` for detailed guide or `docs/MODEL_IMPORT_QUICKSTART.md` for quick reference.

## Experimental Logging

Each pipeline run creates an experiment directory with:

- `detections.jsonl` - Per-frame YOLO detections
- `embeddings.jsonl` - ReID embeddings for each person
- `tracks.jsonl` - Tracking results with track IDs
- `metrics.jsonl` - Performance metrics (FPS, GPU memory, latency)
- `config_snapshot.json` - Complete configuration used
- `video_metadata.json` - Input video metadata

### View Logs

```bash
# View tracking results
cat experiments/<name>/tracks.jsonl | jq '.tracks'

# View performance metrics
cat experiments/<name>/metrics.jsonl | jq '.fps'
```

## Performance

### Expected Performance (RTX 3090)

- **YOLO Detection**: ~6 ms per frame
- **ReID Inference**: ~18 ms per batch
- **Tracking**: ~8 ms per frame
- **Overall FPS**: 30-40 FPS

## Troubleshooting

### Triton Server Not Starting

```bash
# Check Docker logs
docker logs triton-reid-server

# Verify model repository
ls triton_models/swin_base_reid/

# Restart server
docker stop triton-reid-server
bash scripts/start_triton_server.sh
```

### Track IDs Changing on Re-entry

Ensure tracker config has:
```yaml
with_reid: true
proximity_thresh: 0.5
track_buffer: 30
```

For longer retention or cross-camera deployments, use the global identity
gallery instead of increasing the local BoTSORT buffer.

### CUDA Out of Memory

- Reduce `max_batch` in `reid_config.yaml`
- Reduce YOLO `imgsz` in `yolo_config.yaml`

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

## License

This project is provided as-is for research and development purposes.

## Changelog

### Version 0.4.0
- Bounded local lost-track retention to prevent unbounded matching-state growth
- Added bounded Triton request chunking and applied configured timeouts
- Fixed IoU-only mode, YOLOE mask mapping, and track-history embedding mapping
- Applied configured sparse optical-flow camera-motion compensation

### Version 0.3.0
- Switched to Swin Base ReID model (1024-dim embeddings)
- Fixed re-identification for track persistence across camera exits/entries
- Added appearance-based matching with configurable thresholds
- Added unlimited track buffer for long-term tracking
- Added dataset evaluation module (mAP, CMC metrics)
- Added model import scripts for deploying new ReID models

### Version 0.2.0
- Initial implementation with Triton Inference Server
- FP16 TensorRT engine support
- BoxMOT tracking integration
- Comprehensive experimental logging
