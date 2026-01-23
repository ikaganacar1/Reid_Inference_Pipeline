# Model Import Quick Start

## For Your Swin Transformer Model

### Option 1: Fully Automated (Recommended)

```bash
# Import Swin Transformer with auto-detection
python scripts/import_model.py \
    --onnx models/swin_reid.onnx \
    --model-name swin_reid \
    --test \
    --benchmark

# This will:
# ✓ Validate ONNX
# ✓ Extract input/output shapes
# ✓ Convert to TensorRT (FP16)
# ✓ Deploy to Triton
# ✓ Create config file
# ✓ Run tests
# ✓ Benchmark performance
```

### Option 2: With Manual Settings

```bash
# If auto-detection fails or you want specific settings
python scripts/import_model.py \
    --onnx models/swin_reid.onnx \
    --model-name swin_reid \
    --input-size 256 128 \
    --embedding-dim 768 \
    --batch-sizes 1 4 8 \
    --precision fp16 \
    --test
```

### Option 3: ONNX Only (No TensorRT)

```bash
# For quick testing without TensorRT conversion
python scripts/import_model.py \
    --onnx models/swin_reid.onnx \
    --model-name swin_reid_onnx \
    --skip-tensorrt \
    --deploy-onnx
```

---

## Common Swin Transformer Configurations

### Swin-Tiny ReID
```bash
python scripts/import_model.py \
    --onnx models/swin_tiny_reid.onnx \
    --input-size 256 128 \
    --embedding-dim 768 \
    --batch-sizes 1 8 16
```

### Swin-Small ReID
```bash
python scripts/import_model.py \
    --onnx models/swin_small_reid.onnx \
    --input-size 256 128 \
    --embedding-dim 768 \
    --batch-sizes 1 4 8  # Smaller batch due to larger model
```

### Swin-Base ReID
```bash
python scripts/import_model.py \
    --onnx models/swin_base_reid.onnx \
    --input-size 384 128 \
    --embedding-dim 1024 \
    --batch-sizes 1 4 8
```

---

## After Import

### 1. Verify Model Loaded

```bash
# Check Triton server
curl http://localhost:8100/v2/models/swin_reid

# Should return model config
```

### 2. Test Inference

```bash
# Quick test
python -c "
import yaml
import numpy as np
from src.reid_client import TritonReIDClient

with open('configs/swin_reid_config.yaml') as f:
    config = yaml.safe_load(f)

client = TritonReIDClient(config)
dummy = [np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)]
emb = client.infer(dummy)
print(f'✓ Embedding shape: {emb.shape}')
"
```

### 3. Run Evaluation

```bash
# Evaluate on LTCC dataset
python scripts/evaluate_dataset.py \
    --reid-config configs/swin_reid_config.yaml \
    --experiment-name swin_eval
```

### 4. Use in Video Pipeline

```bash
# Process video with new model
python main.py \
    --video data/videos/test.mp4 \
    --experiment-name swin_test
```

**Note:** You'll need to update `configs/reid_config.yaml` to point to `swin_reid`, or use:

```bash
# Temporarily use different config
cp configs/swin_reid_config.yaml configs/reid_config.yaml
```

---

## Troubleshooting

### Model Won't Load

```bash
# Check Triton logs
docker logs triton-reid-server

# Common issues:
# - Wrong batch size in config
# - Incorrect input/output names
# - Missing model file
```

### Check Input/Output Names

```bash
python -c "
import onnx
m = onnx.load('models/swin_reid.onnx')
print('Input:', [i.name for i in m.graph.input])
print('Output:', [o.name for o in m.graph.output])
"

# Update config.pbtxt if names don't match
```

### TensorRT Conversion Fails

```bash
# Try ONNX simplification first
pip install onnx-simplifier
python -m onnxsim models/swin_reid.onnx models/swin_reid_simplified.onnx

# Then import
python scripts/import_model.py --onnx models/swin_reid_simplified.onnx
```

### Out of Memory

```bash
# Use smaller batch sizes
python scripts/import_model.py \
    --onnx models/swin_reid.onnx \
    --batch-sizes 1 2 4  # Reduce max batch
```

---

## Model-Specific Settings

After import, you may need to adjust preprocessing in `configs/swin_reid_config.yaml`:

### For Models Trained with Different Normalization

```yaml
preprocessing:
  # Original ImageNet stats (default)
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

  # OR if your model uses different normalization:
  # mean: [0.5, 0.5, 0.5]
  # std: [0.5, 0.5, 0.5]
```

### For Models Expecting BGR Input

```yaml
preprocessing:
  color_space: "BGR"  # Change from RGB
```

### For Square Input Models (ViT, etc.)

```yaml
model:
  input_shape: [224, 224]  # Square instead of 2:1 ratio
```

---

## Performance Comparison

After importing, compare with baseline:

```bash
# Benchmark new model
python scripts/benchmark_triton_model.py \
    --config configs/swin_reid_config.yaml \
    --iterations 100

# Benchmark baseline
python scripts/benchmark_triton_model.py \
    --config configs/reid_config.yaml \
    --iterations 100

# Compare evaluation results
python scripts/evaluate_dataset.py --reid-config configs/swin_reid_config.yaml
python scripts/evaluate_dataset.py --reid-config configs/reid_config.yaml
```

---

## Typical Swin Transformer Characteristics

| Model | Input Size | Embed Dim | Batch Size | Speed vs ResNet50 |
|-------|------------|-----------|------------|-------------------|
| Swin-Tiny | 256×128 | 768 | 8-16 | ~0.8x (slower) |
| Swin-Small | 256×128 | 768 | 4-8 | ~0.6x |
| Swin-Base | 384×128 | 1024 | 2-4 | ~0.4x |

Swin models are typically:
- **More accurate** (+2-5% mAP over ResNet)
- **Slower** (larger models, more computation)
- **Better with clothing changes** (stronger features)

---

## Complete Workflow Example

```bash
# 1. Download your Swin model (already done)
# Assume: models/swin_transformer_reid.onnx

# 2. Import to Triton
python scripts/import_model.py \
    --onnx models/swin_transformer_reid.onnx \
    --model-name swin_reid \
    --test \
    --benchmark

# 3. Evaluate on dataset
python scripts/evaluate_dataset.py \
    --reid-config configs/swin_reid_config.yaml \
    --experiment-name swin_ltcc

# 4. Check results
cat experiments/evaluation/swin_ltcc/results.json | jq '.'

# 5. If better than baseline, use in production
cp configs/swin_reid_config.yaml configs/reid_config.yaml

# 6. Run video pipeline
python main.py --video data/videos/test.mp4
```

---

## References

- Full guide: `docs/IMPORTING_MODELS.md`
- Troubleshooting: Check Triton logs
- Support: Review import script output for errors
