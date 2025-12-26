#!/usr/bin/env python3
"""
Model Validation Script
Validates YOLO and TAO ReID models are properly set up
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))


def validate_yolo_model():
    """Validate YOLO model"""
    print("="*60)
    print("Validating YOLO Model")
    print("="*60)

    model_path = Path("models/yolo11n.pt")
    if not model_path.exists():
        print(f"✗ YOLO model not found: {model_path}")
        return False

    print(f"✓ YOLO model found: {model_path}")
    print(f"  Size: {model_path.stat().st_size / (1024**2):.2f} MB")

    # Try loading model
    try:
        from ultralytics import YOLO
        import numpy as np

        model = YOLO(str(model_path))
        print("✓ YOLO model loaded successfully")

        # Test inference on dummy image
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy_img, conf=0.5, classes=[0], verbose=False)

        print(f"✓ YOLO inference test passed")
        print(f"  Detected {len(results[0].boxes)} objects")

        return True

    except Exception as e:
        print(f"✗ YOLO validation failed: {e}")
        return False


def validate_triton_server():
    """Validate Triton server connection"""
    print("\n" + "="*60)
    print("Validating Triton Inference Server")
    print("="*60)

    try:
        import tritonclient.http as httpclient
        import yaml

        # Load config to get correct Triton URL
        config_path = Path("configs/reid_config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        triton_url = config['triton']['server_url']

        client = httpclient.InferenceServerClient(url=triton_url, verbose=False)

        # Check server health
        if not client.is_server_live():
            print("✗ Triton server is not live")
            print("  Start server with: bash scripts/start_triton_server.sh")
            return False

        print("✓ Triton server is live")

        if not client.is_server_ready():
            print("✗ Triton server is not ready")
            return False

        print("✓ Triton server is ready")

        # Check ReID model
        model_name = "lttc_reid"
        if not client.is_model_ready(model_name, "1"):
            print(f"✗ Model '{model_name}' is not loaded")
            print("  Generate TensorRT engine with: python scripts/export_to_tensorrt.py")
            return False

        print(f"✓ Model '{model_name}' is loaded and ready")

        # Get model metadata
        metadata = client.get_model_metadata(model_name, "1")
        print(f"  Model version: {metadata.get('versions', [])}")

        return True

    except Exception as e:
        print(f"✗ Triton validation failed: {e}")
        print("  Make sure Triton server is running")
        return False


def validate_tensorrt_engine():
    """Validate TensorRT engine"""
    print("\n" + "="*60)
    print("Validating TensorRT Engine")
    print("="*60)

    engine_path = Path("triton_models/lttc_reid/1/model.plan")
    if not engine_path.exists():
        print(f"✗ TensorRT engine not found: {engine_path}")
        print("  Generate with: python scripts/export_to_tensorrt.py")
        return False

    print(f"✓ TensorRT engine found: {engine_path}")
    print(f"  Size: {engine_path.stat().st_size / (1024**2):.2f} MB")

    # Check metadata
    metadata_path = engine_path.parent.parent / f"{engine_path.stem}_metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print("✓ Engine metadata found")
        print(f"  TensorRT version: {metadata.get('tensorrt_version')}")
        print(f"  Precision: {metadata.get('build_config', {}).get('precision')}")
        print(f"  Batch size range: [{metadata.get('build_config', {}).get('min_batch')}, "
              f"{metadata.get('build_config', {}).get('opt_batch')}, "
              f"{metadata.get('build_config', {}).get('max_batch')}]")

    return True


def validate_onnx_model():
    """Validate ONNX model"""
    print("\n" + "="*60)
    print("Validating ONNX Model")
    print("="*60)

    onnx_path = Path("models/lttc_0.1.4.49.onnx")
    if not onnx_path.exists():
        print(f"✗ ONNX model not found: {onnx_path}")
        return False

    print(f"✓ ONNX model found: {onnx_path}")
    print(f"  Size: {onnx_path.stat().st_size / (1024**2):.2f} MB")

    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

        print("✓ ONNX model is valid")

        # Print input/output info
        for input_tensor in model.graph.input:
            shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"  Input '{input_tensor.name}': {shape}")

        for output_tensor in model.graph.output:
            shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"  Output '{output_tensor.name}': {shape}")

        return True

    except Exception as e:
        print(f"✗ ONNX validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate pipeline models")
    parser.add_argument("--skip-triton", action="store_true", help="Skip Triton server check")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("Pipeline Model Validation")
    print("="*60)

    results = {}

    # Validate YOLO
    results['yolo'] = validate_yolo_model()

    # Validate ONNX
    results['onnx'] = validate_onnx_model()

    # Validate TensorRT engine
    results['tensorrt'] = validate_tensorrt_engine()

    # Validate Triton server
    if not args.skip_triton:
        results['triton'] = validate_triton_server()

    # Print summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name.upper()}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All validations passed! Ready to run pipeline.")
        return 0
    else:
        print("\n✗ Some validations failed. Please fix issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
