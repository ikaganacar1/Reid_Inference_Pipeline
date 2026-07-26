# ReID Pipeline - Quick Start Guide

## 🚀 60-Second Setup

```bash
# 1. Install dependencies
conda create -n reid python=3.10 -y && conda activate reid
pip install -r requirements.txt

# 2. Start Triton server (in another terminal)
bash scripts/start_triton_server.sh

# 3. Run pipeline
python scripts/run_pipeline.py \
  --video test_videos/MOT16-02.mp4 \
  --output outputs/tracked.mp4 \
  --experiment-dir experiments/test001

# 4. View results
ffplay outputs/tracked.mp4
```

---

## 📋 Common Commands

### Process a Video
```bash
python scripts/run_pipeline.py \
  --video your_video.mp4 \
  --output your_video_tracked.mp4 \
  --experiment-dir experiments/my_test
```

### Quick Test (10 frames only)
```bash
python scripts/run_pipeline.py \
  --video your_video.mp4 \
  --max-frames 10 \
  --experiment-dir experiments/quick_test
```

### Batch Process Multiple Videos
```bash
for video in videos/*.mp4; do
  python scripts/run_pipeline.py \
    --video "$video" \
    --output "results/$(basename $video)" \
    --experiment-dir "experiments/$(basename $video .mp4)"
done
```

### Validate Setup
```bash
python scripts/validate_models.py
```

---

## 🔧 Manage Triton Server

```bash
# Start
bash scripts/start_triton_server.sh

# Check status
docker ps | grep triton-reid-server

# View logs
docker logs -f triton-reid-server

# Stop
docker stop triton-reid-server

# Check model status
curl http://localhost:8100/v2/models/swin_base_reid
```

---

## ⚙️ Key Configuration Files

### Detection Confidence (YOLO)
```yaml
# configs/yolo_config.yaml
detection:
  conf_threshold: 0.5  # Higher = stricter, faster
```

### Tracking Parameters
```yaml
# configs/tracker_config.yaml
botsort:
  track_buffer: 30          # About 3 seconds at the 10 FPS realtime target
  proximity_thresh: 0.5     # Require spatial support for local association
  appearance_thresh: 0.3    # Maximum local cosine distance
```

### Pipeline Settings
```yaml
# configs/pipeline_config.yaml
logging:
  log_every_n_frames: 30  # Less frequent = faster
processing:
  batch_size: 8  # Maximum crops passed to one configured ReID request
```

---

## 📊 Understanding Outputs

```
experiments/test001/
├── tracked.mp4              # Annotated video with track IDs
├── detections.jsonl         # All detections per frame
├── embeddings.jsonl         # Optional 1024-dim vectors for each person
├── tracks.jsonl             # Track assignments per frame
├── metrics.jsonl            # FPS, GPU usage, latency
├── config_snapshot.json     # Configs used for run
└── model_versions.json      # Model hashes
```

### Quick Analysis
```bash
# Count total people detected
python -c "
import json
count = 0
with open('experiments/test001/detections.jsonl') as f:
    for line in f:
        count += json.loads(line)['num_detections']
print(f'Total detections: {count}')
"

# Count unique tracks
python -c "
import json
ids = set()
with open('experiments/test001/tracks.jsonl') as f:
    for line in f:
        for track in json.loads(line)['tracks']:
            ids.add(int(track[4]))
print(f'Unique people: {len(ids)}')
"
```

---

## 🎯 Optimization Tips

### Slow Processing?
```bash
# 1. Process smaller resolution
ffmpeg -i input.mp4 -vf "scale=1280:720" input_720p.mp4

# 2. Use smaller YOLO model
# Edit configs/yolo_config.yaml:
# path: "models/yolo8n.pt"

# 3. Disable expensive embedding-vector logging
# Edit configs/pipeline_config.yaml:
# save_embeddings: false
```

### GPU Out of Memory?
```bash
# 1. Reduce batch
# Edit configs/reid_config.yaml:
# max_batch_size: 8

# 2. Process fewer frames
python scripts/run_pipeline.py --max-frames 100

# 3. Disable expensive logging
# Edit configs/pipeline_config.yaml:
# save_embeddings: false
# save_crops: false
```

---

## ❌ Troubleshooting

### Triton won't start
```bash
docker logs triton-reid-server
# If: "port already in use"
docker stop $(docker ps -q)
bash scripts/start_triton_server.sh
```

### YOLO model not found
```bash
cd models
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolo11n.pt
cd ..
```

### Can't connect to Triton
```bash
# Check if running
docker ps | grep triton

# Start it
bash scripts/start_triton_server.sh

# Check health
curl http://localhost:8100/v2/health/ready
```

### CUDA out of memory
```bash
# Reduce to 720p
ffmpeg -i input.mp4 -vf "scale=1280:720" input_720p.mp4
python scripts/run_pipeline.py --video input_720p.mp4
```

---

## 📈 Performance Expectations

| Resolution | Model | FPS | GPU Memory |
|------------|-------|-----|-----------|
| 720p | YOLO11n | 18 | 1.1 GB |
| 1080p | YOLO11n | 12 | 1.2 GB |
| 1440p | YOLO11n | 8 | 1.4 GB |
| 720p | YOLO8n | 22 | 0.9 GB |

---

## 🔍 Debugging

```bash
# Profile one component
python -c "
from src.detector import YOLOPersonDetector
import yaml, time, numpy as np

with open('configs/yolo_config.yaml') as f:
    config = yaml.safe_load(f)

detector = YOLOPersonDetector(config['yolo'])
frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

times = []
for _ in range(10):
    t0 = time.time()
    detector.detect(frame)
    times.append((time.time() - t0) * 1000)

print(f'YOLO: {np.mean(times):.1f}±{np.std(times):.1f} ms')
"

# Check GPU
nvidia-smi

# Monitor in real-time
watch -n 1 nvidia-smi
```

---

## 📚 Full Documentation

See **PIPELINE_GUIDE.md** for:
- Detailed architecture explanation
- Component API reference
- Data format specifications
- Advanced customization
- Complete troubleshooting guide

---

## 🎬 Example: Process Your Own Video

```bash
# 1. Prepare video
ffmpeg -i my_video.mov -vcodec libx264 -crf 23 my_video.mp4

# 2. Process
python scripts/run_pipeline.py \
  --video my_video.mp4 \
  --output my_video_tracked.mp4 \
  --experiment-dir experiments/my_video

# 3. View results
ffplay my_video_tracked.mp4

# 4. Analyze logs
python -c "
import json
with open('experiments/my_video/metrics.jsonl') as f:
    metrics = [json.loads(line) for line in f]
    print(f'Average FPS: {sum(m[\"fps\"] for m in metrics) / len(metrics):.1f}')
"
```

---

**For detailed documentation:** See PIPELINE_GUIDE.md
**For architecture details:** See RESULTS_SUMMARY.md
**For help:** Check PIPELINE_GUIDE.md Troubleshooting section
