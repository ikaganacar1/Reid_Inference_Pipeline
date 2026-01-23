# Importing New ReID Models to Triton

This guide walks through deploying new person re-identification models to the Triton Inference Server.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Model Requirements](#model-requirements)
4. [Deployment Methods](#deployment-methods)
5. [Common Models](#common-models)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Automated Import (Recommended)

```bash
# Import ONNX model with automatic TensorRT conversion
python scripts/import_model.py \
    --onnx models/my_model.onnx \
    --model-name my_reid \
    --batch-sizes 1 8 16

# Import and test immediately
python scripts/import_model.py \
    --onnx models/swin_reid.onnx \
    --model-name swin_reid \
    --test
```

The script will:
1. ✓ Validate ONNX model
2. ✓ Extract input/output shapes and names
3. ✓ Convert to TensorRT (FP16)
4. ✓ Generate Triton config files
5. ✓ Deploy to Triton server
6. ✓ Run validation tests

### Manual Import

See [Deployment Methods](#deployment-methods) below.

---

## Prerequisites

### Software Requirements

- NVIDIA GPU (compute capability ≥ 7.0)
- CUDA 12.x
- TensorRT 8.6+
- Triton Inference Server container
- Python 3.8+

### Python Dependencies

```bash
pip install tensorrt onnx onnx-simplifier tritonclient[all]
```

### Triton Server Running

```bash
# Check if running
docker ps | grep triton

# If not running, start it
bash scripts/start_triton_server.sh
```

---

## Model Requirements

### Supported Formats

| Format | Extension | Platform | Speed | Recommended |
|--------|-----------|----------|-------|-------------|
| ONNX | `.onnx` | `onnxruntime_onnx` | 1x | Testing only |
| TensorRT | `.plan` | `tensorrt_plan` | 2-3x | ✓ Production |

### Model Constraints

**Input Requirements:**
- Format: `[batch, channels, height, width]`
- Data type: FP32 (float32)
- Color space: RGB (most models) or BGR
- Normalization: Model-specific (usually ImageNet stats)

**Output Requirements:**
- Format: `[batch, embedding_dim]`
- Data type: FP32
- Embedding dim: Typically 256, 512, 768, 1024, or 2048
- L2-normalized or raw (specify in config)

**Batch Size:**
- Minimum: 1
- Recommended max: 16-32
- Must be consistent across model and config

### Expected Model Architecture

```
Input: [B, 3, H, W] RGB image
   ↓
Preprocessing (outside model):
  - Resize to model input size
  - Normalize with mean/std
   ↓
Model Forward Pass:
  - Backbone (ResNet, Swin, ViT, etc.)
  - Global pooling
  - FC layer
   ↓
Output: [B, D] embedding vector
```

---

## Deployment Methods

### Method 1: Automated Script (Recommended)

```bash
python scripts/import_model.py \
    --onnx models/your_model.onnx \
    --model-name your_model \
    --input-size 384 192 \
    --embedding-dim 768 \
    --batch-sizes 1 8 16 \
    --precision fp16 \
    --test \
    --benchmark
```

**Options:**
- `--onnx`: Path to ONNX model file (required)
- `--model-name`: Name for Triton deployment (default: from filename)
- `--input-size`: Height Width (default: auto-detect)
- `--embedding-dim`: Output dimension (default: auto-detect)
- `--batch-sizes`: Min Opt Max batch sizes (default: 1 8 16)
- `--precision`: fp16 or fp32 (default: fp16)
- `--test`: Run validation after deployment
- `--benchmark`: Run performance benchmark
- `--deploy-onnx`: Also deploy ONNX version (for comparison)
- `--skip-tensorrt`: Skip TensorRT conversion (ONNX only)

### Method 2: Manual Deployment

#### Step 1: Inspect ONNX Model

```bash
python -c "
import onnx
import numpy as np

model = onnx.load('models/your_model.onnx')

# Validate
onnx.checker.check_model(model)
print('✓ ONNX model is valid\n')

# Get input info
for inp in model.graph.input:
    shape = [d.dim_value if d.dim_value > 0 else -1
             for d in inp.type.tensor_type.shape.dim]
    print(f'Input: {inp.name}')
    print(f'  Shape: {shape}')
    print(f'  Type: {inp.type.tensor_type.elem_type}\n')

# Get output info
for out in model.graph.output:
    shape = [d.dim_value if d.dim_value > 0 else -1
             for d in out.type.tensor_type.shape.dim]
    print(f'Output: {out.name}')
    print(f'  Shape: {shape}')
    print(f'  Type: {out.type.tensor_type.elem_type}')
"
```

#### Step 2: Convert to TensorRT

```bash
# Using built-in script
python scripts/export_to_tensorrt.py \
    --onnx models/your_model.onnx \
    --output triton_models/your_model/1/model.plan \
    --min-batch 1 \
    --opt-batch 8 \
    --max-batch 16 \
    --precision fp16
```

#### Step 3: Create Directory Structure

```bash
mkdir -p triton_models/your_model/1
cp models/your_model.onnx triton_models/your_model/1/model.onnx
# OR
mv <tensorrt_output>/model.plan triton_models/your_model/1/model.plan
```

#### Step 4: Create config.pbtxt

**For ONNX:**
```protobuf
name: "your_model"
platform: "onnxruntime_onnx"
max_batch_size: 16

input [
  {
    name: "input"              # MUST match ONNX input name!
    data_type: TYPE_FP32
    dims: [ 3, 384, 192 ]      # [C, H, W] - no batch dimension
  }
]

output [
  {
    name: "output"             # MUST match ONNX output name!
    data_type: TYPE_FP32
    dims: [ 768 ]              # Embedding dimension
  }
]

dynamic_batching {
  preferred_batch_size: [ 1, 4, 8, 16 ]
  max_queue_delay_microseconds: 100
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

**For TensorRT:**
```protobuf
name: "your_model"
platform: "tensorrt_plan"
max_batch_size: 16

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 384, 192 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 768 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 1, 4, 8, 16 ]
  max_queue_delay_microseconds: 100
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

optimization {
  cuda {
    graphs: true
  }
}
```

#### Step 5: Deploy to Triton

```bash
# Restart Triton to load new model
docker stop triton-reid-server
bash scripts/start_triton_server.sh

# Wait for server
sleep 10

# Verify model loaded
curl http://localhost:8100/v2/models/your_model
```

#### Step 6: Update Pipeline Config

```bash
# Create new config or modify existing
cp configs/reid_config.yaml configs/your_model_config.yaml

# Edit the new config
# Change:
#   triton.model_name: "your_model"
#   model.input_shape: [H, W]
#   model.embedding_dim: D
#   preprocessing.mean/std: (model-specific)
```

#### Step 7: Test

```bash
# Test with ReID client
python -c "
import yaml
from src.reid_client import TritonReIDClient
import numpy as np

with open('configs/your_model_config.yaml') as f:
    config = yaml.safe_load(f)

client = TritonReIDClient(config)

# Test inference
dummy = [np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)]
emb = client.infer(dummy)
print(f'✓ Output shape: {emb.shape}')
print(f'✓ Embedding range: [{emb.min():.3f}, {emb.max():.3f}]')
"

# Benchmark
python scripts/benchmark_triton_model.py --config configs/your_model_config.yaml
```

---

## Common Models

### 1. Swin Transformer ReID

**Characteristics:**
- Input: Typically 256×128 or 384×128
- Embedding: 768 or 1024 dims
- Preprocessing: ImageNet normalization

**Import:**
```bash
python scripts/import_model.py \
    --onnx models/swin_reid.onnx \
    --model-name swin_reid \
    --input-size 256 128 \
    --embedding-dim 768 \
    --batch-sizes 1 4 8 \
    --test
```

**Config:**
```yaml
preprocessing:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  color_space: "RGB"
```

### 2. ResNet-based ReID

**Characteristics:**
- Input: 256×128 or 384×192
- Embedding: 256, 512, or 2048 dims
- Preprocessing: ImageNet normalization

**Import:**
```bash
python scripts/import_model.py \
    --onnx models/resnet_reid.onnx \
    --model-name resnet_reid \
    --batch-sizes 1 8 16
```

### 3. ViT (Vision Transformer) ReID

**Characteristics:**
- Input: 224×224 or 256×256 (square)
- Embedding: 768 or 1024 dims
- Preprocessing: ViT-specific or ImageNet

**Import:**
```bash
python scripts/import_model.py \
    --onnx models/vit_reid.onnx \
    --model-name vit_reid \
    --input-size 224 224 \
    --embedding-dim 768
```

### 4. OSNet

**Characteristics:**
- Input: 256×128
- Embedding: 512 dims
- Lightweight, fast

**Import:**
```bash
python scripts/import_model.py \
    --onnx models/osnet.onnx \
    --model-name osnet \
    --batch-sizes 1 16 32
```

---

## Troubleshooting

### Model Won't Load

**Error:** `Model not found in repository`

```bash
# Check directory structure
ls -R triton_models/your_model/

# Should see:
# triton_models/your_model/
# ├── config.pbtxt
# └── 1/
#     └── model.plan  (or model.onnx)
```

**Error:** `Invalid input/output name`

```bash
# Get actual names from ONNX
python -c "
import onnx
m = onnx.load('models/your_model.onnx')
print('Inputs:', [i.name for i in m.graph.input])
print('Outputs:', [o.name for o in m.graph.output])
"

# Update config.pbtxt with exact names
```

**Error:** `Batch size mismatch`

```bash
# Ensure max_batch_size in config.pbtxt matches TensorRT engine
# If using TensorRT, must match --max-batch during conversion
```

### Inference Errors

**Error:** `[400] inference request batch-size must be <= X`

```bash
# Reduce batch size in evaluation config
# Edit configs/evaluation_config.yaml:
# evaluation.batch_size: 8  (or whatever max_batch is)
```

**Error:** `Shape mismatch`

```bash
# Check preprocessing in src/reid_client.py
# Model input_shape must match config:
# self.input_shape = config['model']['input_shape']  # [H, W]

# Verify with:
python -c "
import yaml
with open('configs/your_model_config.yaml') as f:
    cfg = yaml.safe_load(f)
print('Config input:', cfg['model']['input_shape'])
"
```

### Performance Issues

**Low FPS / High Latency:**

1. Use TensorRT instead of ONNX
2. Enable FP16 precision
3. Optimize batch sizes (test with benchmark script)
4. Check GPU utilization: `nvidia-smi`

**Out of Memory:**

1. Reduce max_batch_size
2. Use FP16 instead of FP32
3. Close other GPU applications

### Validation Failures

**Wrong embedding dimension:**

```bash
# Check model output
python -c "
import onnx
m = onnx.load('models/your_model.onnx')
for out in m.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f'{out.name}: {shape}')
"

# Update config:
# model.embedding_dim: <correct_value>
```

**Embeddings all zeros/NaN:**

1. Check preprocessing normalization (mean/std)
2. Verify color space (RGB vs BGR)
3. Test with actual image instead of random data

---

## Pipeline Integration

### Using New Model in Video Pipeline

```bash
# 1. Import model
python scripts/import_model.py --onnx models/new_model.onnx --model-name new_model

# 2. Create config
cp configs/reid_config.yaml configs/new_model_config.yaml
# Edit: change model_name, input_shape, embedding_dim

# 3. Update config loader (if needed)
# Edit src/utils/config_loader.py to load new config

# 4. Run pipeline with new model
python main.py \
    --video data/videos/test.mp4 \
    --reid-config configs/new_model_config.yaml
```

### Using New Model in Evaluation

```bash
# 1. Update reid config
# configs/reid_config.yaml -> point to new model

# 2. Run evaluation
python scripts/evaluate_dataset.py

# 3. Compare with baseline
python scripts/compare_models.py \
    --model1 configs/baseline_config.yaml \
    --model2 configs/new_model_config.yaml
```

---

## Best Practices

### Model Preparation

1. **Test ONNX first** - Validate on small dataset before TensorRT
2. **Simplify ONNX** - Use onnx-simplifier to remove unnecessary ops
3. **Fixed batch size** - Avoid dynamic shapes if possible
4. **Verify outputs** - Check embeddings are normalized and reasonable

### Deployment

1. **Use TensorRT in production** - 2-3x faster than ONNX
2. **Enable FP16** - Minimal accuracy loss, 2x faster
3. **Optimize batch size** - Run benchmarks to find sweet spot
4. **Version models** - Use Triton's version folders (1/, 2/, 3/)

### Testing

1. **Unit test** - Verify output shape and range
2. **Visual test** - Check on known images
3. **Benchmark** - Measure latency and throughput
4. **Evaluate** - Run full dataset evaluation (mAP, CMC)

### Monitoring

1. **Log versions** - Track model SHA256 hashes
2. **Monitor metrics** - Compare mAP/CMC across models
3. **Track latency** - Ensure inference time is acceptable
4. **GPU memory** - Monitor VRAM usage

---

## Appendix

### Useful Commands

```bash
# Check ONNX operators
python -c "
import onnx
m = onnx.load('model.onnx')
ops = set(n.op_type for n in m.graph.node)
print('Operators:', sorted(ops))
"

# Simplify ONNX
pip install onnx-simplifier
python -m onnxsim input.onnx output.onnx

# Test with trtexec (alternative to script)
trtexec \
  --onnx=model.onnx \
  --saveEngine=model.plan \
  --minShapes=input:1x3x256x128 \
  --optShapes=input:8x3x256x128 \
  --maxShapes=input:16x3x256x128 \
  --fp16

# Query Triton model
curl http://localhost:8100/v2/models/<model_name>/config

# Triton model statistics
curl http://localhost:8100/v2/models/<model_name>/stats
```

### Input Size Reference

| Model Type | Typical Input | Aspect Ratio |
|------------|---------------|--------------|
| ResNet-50 | 256×128, 384×192 | 2:1 |
| Swin-T/S | 256×128, 384×128 | 2:1 or 3:1 |
| ViT | 224×224, 256×256 | 1:1 |
| OSNet | 256×128 | 2:1 |
| TransReID | 256×128 | 2:1 |

### Embedding Dimension Reference

| Backbone | Output Dim |
|----------|------------|
| ResNet-18 | 512 |
| ResNet-50 | 2048 (or 256 w/ FC) |
| Swin-Tiny | 768 |
| Swin-Small | 768 |
| Swin-Base | 1024 |
| ViT-Small | 384 |
| ViT-Base | 768 |
| ViT-Large | 1024 |

### Color Space Check

Most models expect **RGB**, but verify:

```python
# If model was trained in PyTorch (default RGB):
preprocessing.color_space: "RGB"

# If model was trained with cv2.imread directly (BGR):
preprocessing.color_space: "BGR"
```

---

## Support

For issues or questions:
1. Check Triton logs: `docker logs triton-reid-server`
2. Validate model: `python scripts/import_model.py --validate-only`
3. Run diagnostics: `python scripts/diagnose_triton.py`
4. Review this guide's [Troubleshooting](#troubleshooting) section
