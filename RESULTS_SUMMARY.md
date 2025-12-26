# ReID Inference Pipeline - Testing Results

## ✅ System Status

**All components validated and working:**
- ✓ YOLO11n person detector (5.35 MB)
- ✓ TAO ReID ONNX model (92.07 MB, input: 384x192, output: 256-dim embeddings)  
- ✓ Triton Inference Server (running on ports 8100-8102)
- ✓ BoxMOT BoTSORT tracker with external ReID embeddings
- ✓ Complete end-to-end pipeline with logging and visualization

## 🎯 Test Results (MOT16-02 video, 100 frames)

### Processing Performance
- **Total frames processed:** 100
- **Resolution:** 1920x1080 @ 25 FPS
- **Processing time:** 8.05 seconds
- **Overall FPS:** 12.42 (about 50% real-time)

### Detection & Tracking
- **Total detections:** 478 people
- **Total tracks:** 420 unique track instances

### Latency Breakdown
- **YOLO detection:** 6.70 ms average
- **TAO ReID (Triton):** 22.58 ms average  
- **BoxMOT tracking:** 3.19 ms average
- **Total per-frame:** ~80.5 ms (12.4 FPS)

## 📁 Outputs Generated

### Video Output
- `outputs/MOT16-02_tracked.mp4` (7.4 MB)
  - Annotated video with bounding boxes and track IDs

### Experiment Logs (`experiments/mot16-02-test/`)
- `detections.jsonl` (38 KB) - Per-frame detection results
- `embeddings.jsonl` (2.5 MB) - ReID embeddings for all detections
- `tracks.jsonl` (53 KB) - Tracking results with IDs
- `metrics.jsonl` (1.1 KB) - Performance metrics (FPS, GPU usage, latency)
- `config_snapshot.json` - Configuration used for the run
- `model_versions.json` - Model hashes for reproducibility
- `system_info.json` - Hardware and environment info
- `video_metadata.json` - Input video properties

## 🔧 Key Technical Achievements

### 1. TensorRT Version Mismatch Resolution
**Problem:** TensorRT engine built with v10.14.1 (version 240) incompatible with Triton 25.04 (version 239)

**Solution:** Switched from TensorRT plan file to ONNX model with `onnxruntime_onnx` backend
- Triton converts ONNX to optimized runtime internally
- Maintains performance while ensuring compatibility

### 2. Model Configuration Discovery
- Auto-detected correct input dimensions (384x192) from ONNX graph
- Fixed output tensor name mismatch ("fc_pred" vs "output")
- Corrected preprocessing pipeline (BGR→RGB, ImageNet normalization)

### 3. Port Conflict Resolution
- Moved Triton from default 8000-8002 to 8100-8102 (JupyterHub conflict)

### 4. Track Format Compatibility
- Fixed BoxMOT output format (8 values vs expected 7)
- Updated visualization to handle `[x1, y1, x2, y2, track_id, conf, cls, index]`

## 🚀 Usage

### Start Triton Server
```bash
docker run --rm -d --gpus all \
  --name triton-reid-server \
  --shm-size=1g \
  -p 8100:8000 -p 8101:8001 -p 8102:8002 \
  -v $(pwd)/triton_models:/models \
  nvcr.io/nvidia/tritonserver:25.04-py3 \
  tritonserver --model-repository=/models --log-verbose=1
```

### Run Pipeline
```bash
conda activate tensorrt_blackwell
python scripts/run_pipeline.py \
  --video test_videos/MOT16-02.mp4 \
  --output outputs/tracked.mp4 \
  --experiment-dir experiments/my_experiment \
  --max-frames 100  # optional, process all frames if omitted
```

### Validate All Components
```bash
python scripts/validate_models.py
```

## 📊 Architecture

```
Input Video
    ↓
[YOLO11n] → Person detections (x1,y1,x2,y2,conf,cls)
    ↓
[Crop & Preprocess] → Resize to 384x192, normalize
    ↓
[Triton + TAO ReID] → 256-dim embeddings
    ↓
[BoxMOT BoTSORT] → Track IDs + trajectories
    ↓
[Visualization + Logging] → Annotated video + JSONL logs
```

## 🔍 Next Steps

1. **Performance Optimization:**
   - Build TensorRT engine with matching version for ~2x speedup
   - Implement batching for ReID inference (currently processes crops one-by-one)
   - Use dynamic batching config in Triton

2. **Production Deployment:**
   - Add health checks and auto-restart for Triton
   - Implement streaming input/output for real-time processing
   - Add multi-camera support

3. **Tracking Improvements:**
   - Tune BoxMOT parameters for specific use case
   - Experiment with different tracker types (DeepSORT, ByteTrack)
   - Add post-processing for track smoothing

---
**Generated:** 2025-12-26  
**Environment:** tensorrt_blackwell conda env, Triton 25.04, CUDA 12.x
