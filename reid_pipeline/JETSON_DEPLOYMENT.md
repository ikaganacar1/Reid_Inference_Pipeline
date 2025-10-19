# Jetson Deployment Guide

Complete guide for deploying the Production ReID Pipeline on NVIDIA Jetson devices.

## 🎯 Target Devices

- **Jetson Xavier NX**: 4 streams @ 1080p, 20-35 FPS
- **Jetson Orin NX**: 8-12 streams @ 1080p, 30-50 FPS
- **Jetson AGX Orin**: 32+ streams @ 4K, 50+ FPS

## 📋 Prerequisites

### Hardware Requirements
- NVIDIA Jetson device (Xavier NX, Orin NX, or AGX Orin)
- Minimum 32GB storage (64GB+ recommended)
- Sufficient RAM (8GB minimum, 16GB+ recommended)
- Camera(s) or video files for input

### Software Requirements
- JetPack 5.0+ (includes CUDA, cuDNN, TensorRT)
- Python 3.8+
- Internet connection for initial setup

## 🚀 Installation on Jetson

### Step 1: System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
    python3-pip \
    python3-dev \
    libopencv-dev \
    python3-opencv \
    git

# Install Jetson-specific packages
sudo apt install -y \
    nvidia-jetpack \
    python3-libnvinfer \
    python3-libnvinfer-dev
```

### Step 2: Python Environment

```bash
# Create virtual environment
python3 -m venv ~/reid_venv
source ~/reid_venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch for Jetson
# Get the latest wheel from: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
wget <pytorch_wheel_url>
pip install torch*.whl

# Install torchvision
sudo apt install -y libjpeg-dev zlib1g-dev
git clone --branch v0.15.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.15.0
python setup.py install --user
cd ..
```

### Step 3: Install ReID Pipeline

```bash
# Clone repository
git clone <your_repo_url> reid_pipeline
cd reid_pipeline

# Install requirements
pip install -r requirements.txt

# Install ultralytics for YOLO
pip install ultralytics
```

### Step 4: Download Models

```bash
# Download YOLOv11n
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt

# Copy your trained ReID model
cp /path/to/your/reid_model.pth models/
```

## ⚡ TensorRT Optimization

### Convert YOLO Model

```bash
# FP16 (recommended for Xavier NX)
python main.py convert \
    --yolo-model yolo11n.pt \
    --precision fp16 \
    --output-dir tensorrt_models/

# INT8 (recommended for Orin NX+)
python main.py convert \
    --yolo-model yolo11n.pt \
    --precision int8 \
    --output-dir tensorrt_models/
```

### Convert ReID Model

```bash
# Prepare calibration dataset (for INT8)
mkdir calibration_images
# Copy 500-1000 representative person images to calibration_images/

# FP16 conversion
python main.py convert \
    --reid-model models/reid_model.pth \
    --precision fp16 \
    --output-dir tensorrt_models/

# INT8 conversion (with calibration)
python main.py convert \
    --reid-model models/reid_model.pth \
    --precision int8 \
    --output-dir tensorrt_models/ \
    --calibration-dir calibration_images/
```

## 🎛️ Configuration

### Generate Device-Specific Config

```bash
# For Xavier NX
python main.py config --preset xavier_nx --output config_xavier.yaml

# For Orin NX
python main.py config --preset orin_nx --output config_orin.yaml

# For AGX Orin
python main.py config --preset agx_orin --output config_agx.yaml
```

### Customize Configuration

Edit the generated YAML file:

```yaml
# config_xavier.yaml
device: cuda
enable_display: true  # Set to false for headless deployment

detection:
  model_path: tensorrt_models/yolo_fp16.engine
  conf_threshold: 0.3
  use_tensorrt: true

reid:
  model_path: tensorrt_models/reid_fp16.engine
  batch_size: 8  # Adjust based on memory
  use_tensorrt: true
  tensorrt_precision: fp16

gallery:
  max_size: 300  # Adjust based on expected crowd size
  similarity_threshold_match: 0.70  # Tune for your accuracy needs

queues:
  input_queue_size: 5
  processing_queue_size: 30
  output_queue_size: 15
```

## 🏃 Running the Pipeline

### Single Video Processing

```bash
python main.py run \
    --config config_xavier.yaml \
    --input video.mp4 \
    --output result.mp4
```

### Real-time Camera Processing

```bash
# USB Camera (typically /dev/video0)
python main.py run \
    --config config_xavier.yaml \
    --input 0 \
    --output camera_output.mp4

# CSI Camera (use GStreamer pipeline)
python main.py run \
    --config config_xavier.yaml \
    --input "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink" \
    --output camera_output.mp4
```

### Headless Deployment (No Display)

```bash
python main.py run \
    --config config_xavier.yaml \
    --input video.mp4 \
    --output result.mp4 \
    --no-display \
    --log-file pipeline.log
```

## 📊 Performance Tuning

### Monitor Resource Usage

```bash
# In another terminal
sudo tegrastats

# Key metrics to watch:
# - GPU utilization: Target 70-90%
# - RAM usage: Should stay under 90%
# - CPU usage: Should be low (most work on GPU)
```

### Optimize Batch Size

Start with preset values, then adjust:

```yaml
# For Xavier NX
reid:
  batch_size: 8  # Start here
  # If GPU usage low: increase to 12-16
  # If memory issues: decrease to 4-6

# For Orin NX
reid:
  batch_size: 16  # Start here
  # Can go up to 24-32 if GPU underutilized
```

### Optimize Queue Sizes

Monitor queue depths during runtime:

```bash
# Check pipeline statistics
# Look for "Queue: I:X D:Y O:Z" in display

# If queues always full → increase sizes
# If queues always empty → decrease sizes (save memory)
```

### Optimize Gallery Size

```yaml
gallery:
  max_size: 300  # Xavier NX (conservative)
  max_size: 500  # Orin NX (balanced)
  max_size: 1000 # AGX Orin (aggressive)
  
  # Adjust based on:
  # - Expected number of people
  # - Memory constraints
  # - Performance requirements
```

## 🔍 Troubleshooting

### Issue: Low FPS

**Diagnosis:**
```bash
# Check GPU utilization
sudo tegrastats
```

**Solutions:**
1. Ensure TensorRT models are being used
2. Increase batch size
3. Reduce detection confidence threshold
4. Lower input resolution

### Issue: High Memory Usage

**Diagnosis:**
```bash
# Monitor memory
watch -n 1 free -h
```

**Solutions:**
1. Reduce gallery max_size
2. Reduce queue sizes
3. Decrease batch size
4. Clear GPU cache periodically

### Issue: TensorRT Conversion Fails

**Common causes:**
- Incompatible ONNX opset
- Insufficient memory during build
- Missing TensorRT components

**Solutions:**
```bash
# Increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Try with smaller batch size
# Convert with FP16 instead of INT8
```

### Issue: Camera Not Detected

**For USB cameras:**
```bash
# List video devices
ls -l /dev/video*

# Test camera
v4l2-ctl --list-devices
```

**For CSI cameras:**
```bash
# Test with gst-launch
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1' ! nvvidconv ! autovideosink
```

## 🚢 Production Deployment

### 1. Create Systemd Service

```bash
# Create service file
sudo nano /etc/systemd/system/reid-pipeline.service
```

```ini
[Unit]
Description=ReID Pipeline Service
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/reid_pipeline
Environment="PATH=/home/jetson/reid_venv/bin"
ExecStart=/home/jetson/reid_venv/bin/python main.py run --config /home/jetson/reid_pipeline/config.yaml --input 0 --no-display --log-file /var/log/reid_pipeline.log
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable reid-pipeline
sudo systemctl start reid-pipeline

# Check status
sudo systemctl status reid-pipeline

# View logs
sudo journalctl -u reid-pipeline -f
```

### 2. Set Up Log Rotation

```bash
sudo nano /etc/logrotate.d/reid-pipeline
```

```
/var/log/reid_pipeline.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 0640 jetson jetson
}
```

### 3. Monitoring Script

```bash
# Create monitoring script
nano monitor_pipeline.sh
```

```bash
#!/bin/bash
# Monitor ReID Pipeline

while true; do
    echo "=== $(date) ==="
    
    # Check if service is running
    systemctl is-active reid-pipeline
    
    # GPU stats
    sudo tegrastats --interval 1000 --logfile tegra.log &
    TEGRA_PID=$!
    sleep 5
    kill $TEGRA_PID
    tail -1 tegra.log
    
    # Memory usage
    free -h
    
    # Disk usage
    df -h /
    
    echo ""
    sleep 60
done
```

### 4. Set Up Remote Monitoring

```bash
# Install monitoring tools
sudo apt install -y prometheus-node-exporter

# Access metrics at http://jetson-ip:9100/metrics
```

## 🔒 Security Considerations

### 1. Network Security
```bash
# Configure firewall
sudo ufw enable
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 8080/tcp  # If using web interface
```

### 2. Data Privacy
- Implement data retention policies
- Encrypt sensitive gallery data
- Secure log files with appropriate permissions

### 3. Access Control
```bash
# Create dedicated user
sudo adduser reid-user
sudo usermod -aG video reid-user
```

## 📈 Performance Benchmarks

### Expected Performance

**Xavier NX (FP16 TensorRT):**
- Single stream: 35-45 FPS
- 4 streams: 20-25 FPS
- Detection latency: ~30ms
- ReID latency: ~15ms (batch of 8)

**Orin NX (INT8 TensorRT):**
- Single stream: 60-80 FPS
- 8 streams: 35-45 FPS
- Detection latency: ~8ms
- ReID latency: ~10ms (batch of 16)

**AGX Orin (INT8 TensorRT):**
- Single stream: 100+ FPS
- 32 streams: 40-50 FPS
- Detection latency: ~5ms
- ReID latency: ~8ms (batch of 32)

## 🎯 Optimization Checklist

- [ ] TensorRT models converted and tested
- [ ] Configuration tuned for target hardware
- [ ] Batch sizes optimized for GPU utilization
- [ ] Queue sizes adjusted for workload
- [ ] Gallery size set appropriately
- [ ] Monitoring enabled
- [ ] Logs configured and rotating
- [ ] Systemd service created (for production)
- [ ] Performance benchmarked
- [ ] Failover/restart configured

## 📚 Additional Resources

- [NVIDIA Jetson Developer Guide](https://developer.nvidia.com/embedded/learn/get-started-jetson)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

---

**Deployment successful! Your ReID pipeline is now running on Jetson hardware! 🎉**
