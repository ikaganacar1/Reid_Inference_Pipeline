#!/usr/bin/env python3
"""
Automated ReID Model Import Script
Streamlines deployment of new models to Triton Inference Server

Usage:
    python scripts/import_model.py --onnx models/my_model.onnx --model-name my_reid
    python scripts/import_model.py --onnx models/swin.onnx --test --benchmark
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import onnx
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ModelImporter:
    """Handles automated import and deployment of ReID models to Triton."""

    def __init__(self, args):
        self.args = args
        self.onnx_path = Path(args.onnx)
        self.model_name = args.model_name or self.onnx_path.stem
        self.triton_root = Path("triton_models")
        self.config_root = Path("configs")

        # Model metadata (extracted from ONNX)
        self.input_name = None
        self.output_name = None
        self.input_shape = None  # [C, H, W]
        self.output_shape = None  # [D]
        self.embedding_dim = None

    def run(self):
        """Execute full import pipeline."""
        print("=" * 70)
        print("ReID Model Import Tool")
        print("=" * 70)
        print(f"\nModel: {self.onnx_path}")
        print(f"Target name: {self.model_name}")

        # Step 1: Validate ONNX
        print("\n[1/7] Validating ONNX model...")
        if not self.validate_onnx():
            print("ERROR: ONNX validation failed")
            return False

        # Step 2: Extract metadata
        print("\n[2/7] Extracting model metadata...")
        self.extract_metadata()

        # Step 3: Convert to TensorRT (optional)
        tensorrt_path = None
        if not self.args.skip_tensorrt:
            print("\n[3/7] Converting to TensorRT...")
            tensorrt_path = self.convert_to_tensorrt()
            if tensorrt_path is None and not self.args.deploy_onnx:
                print("ERROR: TensorRT conversion failed and --deploy-onnx not set")
                return False
        else:
            print("\n[3/7] Skipping TensorRT conversion (--skip-tensorrt)")

        # Step 4: Create Triton directory structure
        print("\n[4/7] Creating Triton model repository...")
        self.create_model_directory(tensorrt_path)

        # Step 5: Generate config.pbtxt
        print("\n[5/7] Generating Triton configuration...")
        self.generate_triton_config(tensorrt_path is not None)

        # Step 6: Deploy to Triton
        print("\n[6/7] Deploying to Triton server...")
        if not self.deploy_to_triton():
            print("ERROR: Deployment failed")
            return False

        # Step 7: Create pipeline config
        print("\n[7/7] Creating pipeline configuration...")
        self.create_pipeline_config()

        print("\n" + "=" * 70)
        print("✓ Model import completed successfully!")
        print("=" * 70)
        print(f"\nModel deployed as: {self.model_name}")
        print(f"Config file: configs/{self.model_name}_config.yaml")

        # Optional: Test
        if self.args.test:
            print("\n" + "=" * 70)
            print("Running validation tests...")
            print("=" * 70)
            self.test_model()

        # Optional: Benchmark
        if self.args.benchmark:
            print("\n" + "=" * 70)
            print("Running performance benchmark...")
            print("=" * 70)
            self.benchmark_model()

        print("\n✓ Import complete!")
        print("\nNext steps:")
        print(f"  1. Test: python src/reid_client.py (update config)")
        print(f"  2. Evaluate: python scripts/evaluate_dataset.py --reid-config configs/{self.model_name}_config.yaml")
        print(f"  3. Video: python main.py --video <path> --reid-config configs/{self.model_name}_config.yaml")

        return True

    def validate_onnx(self) -> bool:
        """Validate ONNX model structure."""
        if not self.onnx_path.exists():
            print(f"ERROR: ONNX file not found: {self.onnx_path}")
            return False

        try:
            model = onnx.load(str(self.onnx_path))
            onnx.checker.check_model(model)
            print("  ✓ ONNX model is valid")
            return True
        except Exception as e:
            print(f"  ✗ ONNX validation failed: {e}")
            return False

    def extract_metadata(self):
        """Extract input/output shapes and names from ONNX model."""
        model = onnx.load(str(self.onnx_path))

        # Get input info
        if len(model.graph.input) == 0:
            raise ValueError("Model has no inputs!")

        input_tensor = model.graph.input[0]
        self.input_name = input_tensor.name
        input_shape = [d.dim_value if d.dim_value > 0 else -1
                      for d in input_tensor.type.tensor_type.shape.dim]

        # Assume [batch, C, H, W]
        if len(input_shape) == 4:
            _, C, H, W = input_shape
            self.input_shape = [C, H, W]
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")

        # Get output info
        if len(model.graph.output) == 0:
            raise ValueError("Model has no outputs!")

        output_tensor = model.graph.output[0]
        self.output_name = output_tensor.name
        output_shape = [d.dim_value if d.dim_value > 0 else -1
                       for d in output_tensor.type.tensor_type.shape.dim]

        # Assume [batch, D]
        if len(output_shape) == 2:
            _, D = output_shape
            self.embedding_dim = D
            self.output_shape = [D]
        else:
            raise ValueError(f"Unexpected output shape: {output_shape}")

        # Override with command line args if provided
        if self.args.input_size:
            H, W = self.args.input_size
            self.input_shape = [C, H, W]

        if self.args.embedding_dim:
            self.embedding_dim = self.args.embedding_dim
            self.output_shape = [self.embedding_dim]

        print(f"  Input: {self.input_name} -> {self.input_shape} (CHW)")
        print(f"  Output: {self.output_name} -> {self.output_shape} (embedding dim)")

    def convert_to_tensorrt(self) -> Optional[Path]:
        """Convert ONNX to TensorRT engine."""
        min_batch, opt_batch, max_batch = self.args.batch_sizes

        output_path = self.triton_root / self.model_name / "1" / "model.plan"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"  Converting with batch sizes: min={min_batch}, opt={opt_batch}, max={max_batch}")
        print(f"  Precision: {self.args.precision}")
        print(f"  Output: {output_path}")

        try:
            # Use existing conversion script
            cmd = [
                "python", "scripts/export_to_tensorrt.py",
                "--onnx", str(self.onnx_path),
                "--output", str(output_path),
                "--min-batch", str(min_batch),
                "--opt-batch", str(opt_batch),
                "--max-batch", str(max_batch),
                "--precision", self.args.precision,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("  ✓ TensorRT conversion successful")
                return output_path
            else:
                print(f"  ✗ TensorRT conversion failed:")
                print(result.stderr)
                return None

        except Exception as e:
            print(f"  ✗ Error during conversion: {e}")
            return None

    def create_model_directory(self, tensorrt_path: Optional[Path]):
        """Create Triton model repository structure."""
        model_dir = self.triton_root / self.model_name
        version_dir = model_dir / "1"
        version_dir.mkdir(parents=True, exist_ok=True)

        if self.args.deploy_onnx and not tensorrt_path:
            # Deploy ONNX model
            onnx_target = version_dir / "model.onnx"
            shutil.copy(self.onnx_path, onnx_target)
            print(f"  ✓ Copied ONNX to {onnx_target}")
        elif tensorrt_path and tensorrt_path.exists():
            print(f"  ✓ TensorRT engine at {tensorrt_path}")
        else:
            raise ValueError("No model file to deploy!")

    def generate_triton_config(self, use_tensorrt: bool):
        """Generate config.pbtxt for Triton."""
        config_path = self.triton_root / self.model_name / "config.pbtxt"

        platform = "tensorrt_plan" if use_tensorrt else "onnxruntime_onnx"
        _, _, max_batch = self.args.batch_sizes

        C, H, W = self.input_shape
        D = self.embedding_dim

        config_content = f'''name: "{self.model_name}"
platform: "{platform}"
max_batch_size: {max_batch}

input [
  {{
    name: "{self.input_name}"
    data_type: TYPE_FP32
    dims: [ {C}, {H}, {W} ]
  }}
]

output [
  {{
    name: "{self.output_name}"
    data_type: TYPE_FP32
    dims: [ {D} ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ 1, 4, 8, {max_batch} ]
  max_queue_delay_microseconds: 100
}}

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
'''

        if use_tensorrt:
            config_content += '''
optimization {
  cuda {
    graphs: true
  }
}
'''

        with open(config_path, 'w') as f:
            f.write(config_content)

        print(f"  ✓ Config written to {config_path}")

    def deploy_to_triton(self) -> bool:
        """Restart Triton server to load new model."""
        print("  Restarting Triton server...")

        try:
            # Stop existing server
            subprocess.run(["docker", "stop", "triton-reid-server"],
                         capture_output=True, check=False)
            time.sleep(2)

            # Start server
            result = subprocess.run(["bash", "scripts/start_triton_server.sh"],
                                  capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"  ✗ Failed to start Triton: {result.stderr}")
                return False

            # Wait for server to be ready
            print("  Waiting for server to be ready...")
            for i in range(30):
                try:
                    import tritonclient.http as httpclient
                    client = httpclient.InferenceServerClient(url="localhost:8100")
                    if client.is_server_ready():
                        break
                except:
                    pass
                time.sleep(1)
            else:
                print("  ✗ Triton server not ready after 30 seconds")
                return False

            # Check if model loaded
            import tritonclient.http as httpclient
            client = httpclient.InferenceServerClient(url="localhost:8100")

            if client.is_model_ready(self.model_name):
                print(f"  ✓ Model '{self.model_name}' loaded successfully")
                return True
            else:
                print(f"  ✗ Model '{self.model_name}' failed to load")
                print("  Check logs: docker logs triton-reid-server")
                return False

        except Exception as e:
            print(f"  ✗ Deployment error: {e}")
            return False

    def create_pipeline_config(self):
        """Create YAML config for pipeline integration."""
        config_path = self.config_root / f"{self.model_name}_config.yaml"

        _, _, max_batch = self.args.batch_sizes
        _, H, W = self.input_shape

        config = {
            'triton': {
                'server_url': 'localhost:8100',
                'model_name': self.model_name,
                'model_version': '1',
                'protocol': 'http',
            },
            'model': {
                'onnx_path': str(self.onnx_path),
                'input_shape': [H, W],  # Height x Width
                'embedding_dim': self.embedding_dim,
            },
            'preprocessing': {
                'mean': [0.485, 0.456, 0.406],  # ImageNet default
                'std': [0.229, 0.224, 0.225],
                'color_space': 'RGB',
                'channel_order': 'CHW',
            },
            'tensorrt': {
                'min_batch': self.args.batch_sizes[0],
                'opt_batch': self.args.batch_sizes[1],
                'max_batch': self.args.batch_sizes[2],
                'precision': self.args.precision,
                'workspace_mb': 2048,
            },
            'inference': {
                'max_retry': 3,
                'timeout_ms': 5000,
            }
        }

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"  ✓ Config written to {config_path}")
        print(f"  ⚠ NOTE: Verify preprocessing mean/std are correct for your model!")

    def test_model(self):
        """Run basic validation tests."""
        try:
            from src.reid_client import TritonReIDClient

            # Load config
            config_path = self.config_root / f"{self.model_name}_config.yaml"
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Create client
            client = TritonReIDClient(config)

            # Test with dummy data
            _, H, W = self.input_shape
            dummy_crop = np.random.randint(0, 255, (H*2, W*2, 3), dtype=np.uint8)

            print(f"\n  Testing with dummy crop ({H*2}x{W*2})...")

            # Single inference
            embeddings = client.infer([dummy_crop])
            print(f"  ✓ Single inference: {embeddings.shape}")
            print(f"    Range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
            print(f"    L2 norm: {np.linalg.norm(embeddings[0]):.3f}")

            # Batch inference
            batch_size = min(4, self.args.batch_sizes[2])
            batch_crops = [dummy_crop] * batch_size
            embeddings = client.infer(batch_crops)
            print(f"  ✓ Batch inference ({batch_size}): {embeddings.shape}")

            print("\n  ✓ All tests passed!")

        except Exception as e:
            print(f"\n  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()

    def benchmark_model(self):
        """Run performance benchmark."""
        try:
            config_path = self.config_root / f"{self.model_name}_config.yaml"

            cmd = [
                "python", "scripts/benchmark_triton_model.py",
                "--config", str(config_path),
                "--iterations", "50",
                "--batch-sizes"
            ] + [str(b) for b in [1, self.args.batch_sizes[1], self.args.batch_sizes[2]]]

            subprocess.run(cmd, check=True)

        except Exception as e:
            print(f"\n  ✗ Benchmark failed: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated ReID model import to Triton",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="Path to ONNX model file"
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name in Triton (default: from ONNX filename)"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="Input size [height width] (default: auto-detect from ONNX)"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        help="Embedding dimension (default: auto-detect from ONNX)"
    )

    # TensorRT options
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs=3,
        metavar=("MIN", "OPT", "MAX"),
        default=[1, 8, 16],
        help="Batch sizes: min opt max"
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="TensorRT precision"
    )
    parser.add_argument(
        "--skip-tensorrt",
        action="store_true",
        help="Skip TensorRT conversion (deploy ONNX only)"
    )
    parser.add_argument(
        "--deploy-onnx",
        action="store_true",
        help="Deploy ONNX model (instead of or in addition to TensorRT)"
    )

    # Testing options
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run validation tests after deployment"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark after deployment"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    importer = ModelImporter(args)

    try:
        success = importer.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nImport interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
