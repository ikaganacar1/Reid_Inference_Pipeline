# YOLO + TAO ReID Inference Pipeline

A comprehensive person re-identification pipeline integrating YOLO person detection, TAO-trained ReID models (via Triton Inference Server), and BoxMOT tracking with full experimental logging.

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────┐
│    YOLO     │───▶│  Triton ReID     │───▶│   BoxMOT     │
│  Detector   │    │  (Swin Base)     │    │   Tracker    │
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
2. **Swin Base ReID Model** - Extracts 1024-dim embeddings via Triton Inference Server
3. **BoxMOT (BoTSORT)** - Multi-object tracking with appearance-based re-identification
4. **Experimental Logger** - Comprehensive logging for reproducibility

## Features

- Real-time person re-identification across camera exits/entries
- FP16 TensorRT inference for ReID model
- Dynamic batching (1-16) via Triton Inference Server
- Appearance-based matching for long-term tracking
- Unlimited track memory (never forgets a person)
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

### 2. Start Triton Server

```bash
# Start Triton Inference Server
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
│   ├── reid_config.yaml       # ReID + Triton settings
│   ├── tracker_config.yaml    # BoxMOT tracker settings
│   ├── pipeline_config.yaml   # Pipeline orchestration
│   └── evaluation_config.yaml # Dataset evaluation settings
│
├── models/                     # Model files
│   └── yolo11n.pt             # YOLO model
│
├── triton_models/              # Triton model repository
│   ├── lttc_reid/             # LTTC ReID model
│   └── swin_base_reid/        # Swin Base ReID model (default)
│
├── src/                        # Source code
│   ├── detector.py            # YOLO wrapper
│   ├── reid_client.py         # Triton HTTP client
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
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Configuration

### ReID Configuration (`configs/reid_config.yaml`)

```yaml
triton:
  server_url: "localhost:8100"
  model_name: "swin_base_reid"
  model_version: "1"

model:
  input_shape: [256, 128]  # H x W
  embedding_dim: 1024      # Swin Base output

preprocessing:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
```

### Tracker Configuration (`configs/tracker_config.yaml`)

```yaml
botsort:
  # Re-identification settings
  with_reid: true           # Enable appearance-based matching
  proximity_thresh: 1.0     # Allow matching without IoU overlap
  appearance_thresh: 0.45   # Embedding distance threshold
  track_buffer: 999999999   # Unlimited - never forget a person

  # Standard tracking parameters
  track_high_thresh: 0.5
  track_low_thresh: 0.1
  new_track_thresh: 0.6
  match_thresh: 0.8
```

**Key settings for re-identification:**
- `with_reid: true` - Enables appearance-based matching using ReID embeddings
- `proximity_thresh: 1.0` - Allows re-identification when person re-enters at different location
- `track_buffer: 999999999` - Never removes lost tracks (unlimited memory)
- `appearance_thresh: 0.45` - Cosine distance threshold (higher = more lenient)

## Usage Examples

### Basic Usage

```bash
# Process single video
python main.py --video test_video.mp4
```

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
proximity_thresh: 1.0
track_buffer: 999999999
```

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
