# Quick Start Guide

Get up and running with the Production ReID Pipeline in 5 minutes!

## 🚀 Installation (5 minutes)

### 1. Setup Environment

```bash
cd reid_pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy opencv-python ultralytics pyyaml
```

### 2. Download YOLO Model

```bash
# Download YOLOv11n
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
```

## ⚡ Usage (1 minute)

### Option 1: Command Line (Easiest)

```bash
# Process video file
python main.py run --preset development --input video.mp4 --output result.mp4

# Use webcam
python main.py run --preset development --input 0

# Use custom model
python main.py run \
  --preset development \
  --input video.mp4 \
  --output result.mp4 \
  --yolo-model yolo11n.pt \
  --reid-model your_reid_model.pth
```

### Option 2: Python Script

```python
from pipeline.production_pipeline import ProductionReIDPipeline

# Create pipeline
pipeline = ProductionReIDPipeline(
    yolo_model_path='yolo11n.pt',
    reid_model_path=None,  # Use default
    detection_conf=0.3,
    reid_threshold_match=0.70,
    enable_display=True
)

# Run on video
pipeline.run(
    video_source='video.mp4',
    output_path='output.mp4'
)
```

## 🎯 For Your Specific Setup

### If you have the files from your upload:

```bash
# Copy your models
cp /path/to/yolo11n.pt .
cp /path/to/model_epoch_039_step_21863.pth .

# Run with your models
python main.py run \
  --preset development \
  --input /path/to/MOT16-05.mp4 \
  --output MOT16-05_result.mp4 \
  --yolo-model yolo11n.pt \
  --reid-model model_epoch_039_step_21863.pth
```

## 📊 Keyboard Controls During Execution

- `q` - Quit processing
- `s` - Save current gallery state

## 🔧 Configuration Presets

### For Testing/Development (Desktop GPU)
```bash
python main.py run --preset development --input video.mp4
```

### For Jetson Xavier NX
```bash
python main.py run --preset xavier_nx --input video.mp4
```

### For Jetson Orin NX
```bash
python main.py run --preset orin_nx --input video.mp4
```

## ⚙️ Generate Custom Configuration

```bash
# Generate config file
python main.py config --preset xavier_nx --output my_config.yaml

# Edit as needed
nano my_config.yaml

# Use your config
python main.py run --config my_config.yaml --input video.mp4
```

## 🚄 TensorRT Acceleration (Optional)

For maximum performance on Jetson:

```bash
# Convert YOLO to TensorRT
python main.py convert \
  --yolo-model yolo11n.pt \
  --precision fp16 \
  --output-dir tensorrt_models/

# Convert ReID to TensorRT
python main.py convert \
  --reid-model your_reid_model.pth \
  --precision fp16 \
  --output-dir tensorrt_models/

# Use TensorRT models
python main.py run \
  --preset xavier_nx \
  --input video.mp4 \
  --yolo-model tensorrt_models/yolo_fp16.engine \
  --reid-model tensorrt_models/reid_fp16.engine
```

## 💾 Gallery Persistence

```bash
# Save gallery for later use
python main.py run \
  --input video1.mp4 \
  --output result1.mp4 \
  --save-state gallery.pkl

# Continue with saved gallery
python main.py run \
  --input video2.mp4 \
  --output result2.mp4 \
  --load-state gallery.pkl \
  --save-state gallery.pkl
```

## 📈 Monitoring Output

The pipeline displays real-time statistics:
- **FPS**: Current frames per second
- **Processing Time**: Time per frame (ms)
- **Detections**: Current frame detections
- **Gallery Size**: Active persons in gallery
- **Total Persons**: All unique persons seen
- **Queue**: Queue depths (I:input, D:detection, O:output)

## 🐛 Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size and gallery size
python main.py config --preset development --output config.yaml
# Edit config.yaml: reduce reid.batch_size and gallery.max_size
python main.py run --config config.yaml --input video.mp4
```

### "Model file not found"
```bash
# Ensure model paths are correct
ls -la *.pt *.pth
# Use absolute paths if needed
python main.py run --yolo-model /full/path/to/yolo11n.pt --input video.mp4
```

### Low FPS
```bash
# Use TensorRT conversion (see above)
# Or reduce resolution
# Or increase detection confidence threshold
```

## 📚 Next Steps

1. **Read Full README**: See [README.md](README.md) for detailed documentation
2. **Check Examples**: See `examples/example_usage.py` for more usage patterns
3. **Optimize Performance**: Follow TensorRT conversion guide for 2-10x speedup
4. **Customize Config**: Edit YAML configs for your specific use case

## 🆘 Getting Help

- Check [README.md](README.md) for detailed documentation
- Review example scripts in `examples/`
- Open an issue on GitHub

---

**Happy Tracking! 🎉**
