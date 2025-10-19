# Production Person Re-Identification Pipeline for NVIDIA Jetson

A production-grade Person Re-Identification (ReID) system optimized for NVIDIA Jetson edge AI devices, featuring real-time tracking, gallery management, and TensorRT acceleration.

## 🎯 Key Features

- **Production-Ready Architecture**: Asynchronous multi-threaded pipeline with queue management
- **Advanced Gallery System**: Dynamic circular buffer with three-tier matching (Match/Uncertain/New)
- **TensorRT Optimization**: FP16/INT8 acceleration for maximum performance on Jetson
- **Batch Processing**: Optimized ReID feature extraction with automatic batching
- **Graceful Degradation**: Three-tier system for handling resource constraints
- **State Persistence**: Gallery save/load for warm restart capability
- **Hardware Presets**: Optimized configurations for Xavier NX, Orin NX, and AGX Orin

## 🚀 Performance Targets

| Hardware | Streams | Resolution | Target FPS | Optimization |
|----------|---------|------------|------------|--------------|
| Xavier NX | 4 | 1080p | 20-35 | FP16 |
| Orin NX | 8-12 | 1080p | 30-50 | INT8 |
| AGX Orin | 32+ | 4K | 50+ | INT8 + Aggressive Batching |

## 📋 Requirements

### Hardware
- NVIDIA Jetson Xavier NX, Orin NX, or AGX Orin
- Minimum 8GB RAM
- CUDA-capable GPU

### Software
```bash
# Core dependencies
Python 3.8+
PyTorch 2.0+
CUDA 11.0+
cuDNN 8.0+

# Python packages
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.19.0
pyyaml>=5.4.0

# Optional (for TensorRT)
tensorrt>=8.0.0
pycuda>=2021.1
```

## 🔧 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/reid-pipeline.git
cd reid-pipeline
```

### 2. Install Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (Jetson-specific)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip3 install -r requirements.txt
```

### 3. Download Models
```bash
# Download YOLOv11n model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt

# Download or train your ReID model
# Place in models/ directory
```

## 💻 Quick Start

### Basic Usage

```bash
# Run with development preset (no TensorRT)
python main.py run --preset development --input video.mp4 --output result.mp4

# Run with hardware preset
python main.py run --preset xavier_nx --input 0 --output webcam.mp4

# Run with custom configuration
python main.py run --config my_config.yaml --input video.mp4
```

### Generate Configuration

```bash
# Generate configuration for Xavier NX
python main.py config --preset xavier_nx --output config_xavier.yaml

# Edit the configuration file as needed
nano config_xavier.yaml

# Run with your configuration
python main.py run --config config_xavier.yaml --input video.mp4
```

## 🔄 TensorRT Conversion

### Convert Models for Maximum Performance

```bash
# Convert YOLO model to TensorRT FP16
python main.py convert --yolo-model yolo11n.pt --precision fp16 --output-dir tensorrt_models/

# Convert ReID model to TensorRT INT8
python main.py convert --reid-model reid_model.pth --precision int8 --output-dir tensorrt_models/
```

### Manual Conversion (Advanced)

```python
from optimization.tensorrt_converter import TensorRTConverter
from pathlib import Path

converter = TensorRTConverter()

# Convert YOLO
engine = converter.convert_yolo_to_tensorrt(
    Path('yolo11n.pt'),
    Path('tensorrt_models/'),
    precision='fp16'
)

# Convert ReID with INT8 calibration
calibration_data = prepare_calibration_dataset(
    Path('calibration_images/'),
    num_images=1000
)

engine = converter.convert_reid_to_tensorrt(
    reid_model,
    Path('tensorrt_models/'),
    precision='int8',
    calibration_images=calibration_data
)
```

## 📊 Configuration

### Configuration File Structure

```yaml
device: cuda
enable_display: true

detection:
  model_path: yolo11n.pt
  conf_threshold: 0.3
  iou_threshold: 0.55
  use_tensorrt: true

reid:
  model_path: reid_model.pth
  embedding_dim: 2048
  batch_size: 16
  use_tensorrt: true
  tensorrt_precision: fp16

gallery:
  max_size: 500
  similarity_threshold_match: 0.70
  similarity_threshold_new: 0.50
  quality_admission_threshold: 0.91
  ttl_frames_sparse: 150

queues:
  input_queue_size: 10
  processing_queue_size: 50
  output_queue_size: 20
```

### Hardware Presets

#### Xavier NX (4 streams, FP16)
```bash
python main.py run --preset xavier_nx --input video.mp4
```

#### Orin NX (8-12 streams, INT8)
```bash
python main.py run --preset orin_nx --input video.mp4
```

#### AGX Orin (32+ streams, INT8)
```bash
python main.py run --preset agx_orin --input video.mp4
```

## 🏗️ Architecture

### Pipeline Flow

```
Camera/Video → Input Queue → Detection → Detection Queue → ReID + Tracking → Output Queue → Display/Save
   Thread 1       (10)       Thread 2      (50)           Thread 3          (20)        Thread 4
```

### Gallery Management

- **Circular Buffer**: Last 5-10 observations per identity
- **Centroid Embedding**: Confidence-weighted aggregation
- **Three-Tier Matching**:
  - **Match** (>0.70): Associate with existing identity
  - **Uncertain** (0.50-0.70): Temporal analysis
  - **New** (<0.50): Create new gallery entry
- **Adaptive Pruning**: TTL-based (30-150 frames) + LRU eviction

### Component Overview

```
reid_pipeline/
├── models/                 # Detection and ReID models
│   ├── detector.py        # Enhanced YOLO detector with TensorRT
│   └── reid_model.py      # Batch ReID extractor
├── gallery/               # Gallery management system
│   ├── circular_buffer.py # Memory-efficient buffer
│   ├── gallery_manager.py # Dynamic gallery with pruning
│   └── similarity.py      # Cosine/Euclidean matching + k-reciprocal
├── pipeline/              # Main pipeline orchestration
│   └── production_pipeline.py  # Async multi-threaded pipeline
├── optimization/          # Performance optimization
│   └── tensorrt_converter.py   # ONNX→TensorRT conversion
└── utils/                 # Utilities and configuration
    └── config.py          # Configuration management
```

## 🎮 Advanced Usage

### Gallery Persistence

```bash
# Save gallery state during processing (press 's' while running)
# Or automatically save on exit

# Load existing gallery for continued processing
python main.py run --input video2.mp4 --load-state gallery_state.pkl

# Save gallery on exit
python main.py run --input video.mp4 --save-state final_gallery.pkl
```

### Multi-Camera Processing

```python
from pipeline.production_pipeline import ProductionReIDPipeline

# Create pipeline with shared gallery
pipeline = ProductionReIDPipeline(
    yolo_model_path='tensorrt_models/yolo_fp16.engine',
    reid_model_path='tensorrt_models/reid_fp16.engine',
    gallery_max_size=1000
)

# Process multiple streams
# (Implementation depends on your multi-camera setup)
```

### Custom ReID Model

```python
from models.reid_model import ReIDModel
import torch

# Define custom model
class CustomReIDModel(ReIDModel):
    def __init__(self):
        super().__init__(embedding_dim=512)
        # Add your custom layers

# Train and save
model = CustomReIDModel()
# ... training code ...
torch.save(model.state_dict(), 'custom_reid.pth')

# Use in pipeline
python main.py run --reid-model custom_reid.pth --input video.mp4
```

## 📈 Monitoring and Statistics

### Runtime Monitoring

The pipeline displays real-time statistics:
- FPS (Frames Per Second)
- Processing time per frame
- Active detections
- Gallery size
- Queue depths
- Total persons tracked

### Statistics Export

Statistics are automatically saved to JSON:
```json
{
  "frames_processed": 10000,
  "avg_fps": 32.5,
  "total_persons_tracked": 47,
  "gallery_size": 35,
  "processing_times": {...}
}
```

## 🛠️ Troubleshooting

### Low FPS

1. **Enable TensorRT**: Convert models to TensorRT FP16/INT8
2. **Adjust Batch Size**: Increase ReID batch size
3. **Reduce Gallery Size**: Lower max_gallery_size
4. **Lower Detection Threshold**: Reduce detection_conf
5. **Use Hardware Preset**: Ensure you're using the correct preset for your device

### High Memory Usage

1. **Reduce Gallery Size**: Lower max_gallery_size
2. **Reduce Queue Sizes**: Decrease queue_size_*
3. **Reduce Buffer Capacity**: Lower buffer_capacity in gallery config
4. **Enable Aggressive Pruning**: Reduce ttl_frames_sparse

### ID Switching

1. **Increase Match Threshold**: Raise similarity_threshold_match to 0.75
2. **Increase Gallery Admission Threshold**: Raise quality_admission_threshold to 0.93
3. **Increase Detection Confidence**: Raise conf_threshold to 0.35
4. **Use Better ReID Model**: Train on domain-specific data

### TensorRT Errors

1. **Check CUDA/cuDNN**: Ensure compatible versions
2. **Update TensorRT**: Install latest TensorRT for your Jetpack
3. **Verify Model**: Test ONNX export first
4. **Check Calibration**: For INT8, ensure calibration data is representative

## 📝 Performance Tuning Guide

### For Xavier NX

```yaml
# Optimal settings for Xavier NX
detection:
  conf_threshold: 0.3
  use_tensorrt: true

reid:
  batch_size: 8
  tensorrt_precision: fp16

gallery:
  max_size: 300
  buffer_capacity: 5

queues:
  input_queue_size: 5
  processing_queue_size: 30
```

### For Orin NX

```yaml
# Optimal settings for Orin NX
detection:
  conf_threshold: 0.3
  use_tensorrt: true

reid:
  batch_size: 16
  tensorrt_precision: int8

gallery:
  max_size: 500
  buffer_capacity: 10

queues:
  input_queue_size: 10
  processing_queue_size: 50
```

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Performance Benchmark
```bash
python scripts/benchmark.py --config config.yaml --video test_video.mp4
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv11 by Ultralytics
- Person Re-Identification research community
- NVIDIA Jetson team for optimization guidelines

## 📞 Support

For issues and questions:
- GitHub Issues: [github.com/yourusername/reid-pipeline/issues](https://github.com/yourusername/reid-pipeline/issues)
- Documentation: [docs link]
- Email: support@example.com

## 🔗 References

- [Person Re-identification: Past, Present and Future](https://arxiv.org/abs/1610.02984)
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Jetson Developer Guide](https://developer.nvidia.com/embedded/develop/software)
