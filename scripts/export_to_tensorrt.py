#!/usr/bin/env python3
"""
TensorRT Engine Builder for TAO ReID Model
Converts ONNX model to TensorRT engine with FP16 precision and dynamic batching
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import tensorrt as trt
import onnx

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class TensorRTEngineBuilder:
    def __init__(self, onnx_path, engine_path, config):
        self.onnx_path = Path(onnx_path)
        self.engine_path = Path(engine_path)
        self.config = config

        # TensorRT logger
        self.logger = trt.Logger(trt.Logger.INFO)

    def validate_onnx(self):
        """Validate ONNX model structure"""
        print(f"Validating ONNX model: {self.onnx_path}")

        model = onnx.load(str(self.onnx_path))
        onnx.checker.check_model(model)

        # Print model info
        print(f"  Model IR version: {model.ir_version}")
        print(f"  Producer: {model.producer_name} {model.producer_version}")

        # Print input/output shapes
        for input_tensor in model.graph.input:
            shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"  Input '{input_tensor.name}': {shape}")

        for output_tensor in model.graph.output:
            shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"  Output '{output_tensor.name}': {shape}")

        print("ONNX model validation passed!")
        return True

    def build_engine(self):
        """Build TensorRT engine from ONNX model"""
        print(f"Building TensorRT engine: {self.engine_path}")
        print(f"  FP16 precision: {self.config['precision'] == 'fp16'}")
        print(f"  Batch size range: [{self.config['min_batch']}, {self.config['opt_batch']}, {self.config['max_batch']}]")
        print(f"  Workspace: {self.config['workspace_mb']} MB")

        # Create builder
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX model
        print("Parsing ONNX model...")
        with open(self.onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("ERROR: Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return False

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            self.config['workspace_mb'] * 1024 * 1024
        )

        # Enable FP16 if requested
        if self.config['precision'] == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 mode enabled")

        # Set dynamic batch size with optimization profile
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name

        # Get actual input dimensions from ONNX model
        input_tensor = network.get_input(0)
        _, C, H, W = input_tensor.shape  # Get actual shape from model

        # Input shape: [batch, channels, height, width]
        min_shape = (self.config['min_batch'], C, H, W)
        opt_shape = (self.config['opt_batch'], C, H, W)
        max_shape = (self.config['max_batch'], C, H, W)

        profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
        config.add_optimization_profile(profile)

        print(f"  Optimization profile set:")
        print(f"    Min shape: {min_shape}")
        print(f"    Opt shape: {opt_shape}")
        print(f"    Max shape: {max_shape}")

        # Build engine
        print("Building TensorRT engine (this may take a few minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            print("ERROR: Failed to build TensorRT engine")
            return False

        # Save engine
        self.engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.engine_path, 'wb') as f:
            f.write(serialized_engine)

        # Get engine size
        engine_size_mb = self.engine_path.stat().st_size / (1024 * 1024)
        print(f"TensorRT engine saved: {self.engine_path}")
        print(f"  Size: {engine_size_mb:.2f} MB")

        return True

    def compute_hash(self, file_path):
        """Compute SHA256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def save_metadata(self):
        """Save engine metadata for versioning"""
        metadata = {
            "onnx_model": {
                "path": str(self.onnx_path),
                "sha256": self.compute_hash(self.onnx_path),
                "size_mb": self.onnx_path.stat().st_size / (1024 * 1024)
            },
            "tensorrt_engine": {
                "path": str(self.engine_path),
                "sha256": self.compute_hash(self.engine_path),
                "size_mb": self.engine_path.stat().st_size / (1024 * 1024)
            },
            "build_config": self.config,
            "tensorrt_version": trt.__version__
        }

        metadata_path = self.engine_path.parent / f"{self.engine_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved: {metadata_path}")
        return metadata


def load_config(config_path):
    """Load configuration from YAML file"""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['tensorrt']


def main():
    parser = argparse.ArgumentParser(description="Export ONNX model to TensorRT engine")
    parser.add_argument(
        "--onnx",
        default="models/lttc_0.1.4.49.onnx",
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--output",
        default="triton_models/lttc_reid/1/model.plan",
        help="Output path for TensorRT engine"
    )
    parser.add_argument(
        "--config",
        default="configs/reid_config.yaml",
        help="Path to ReID config file"
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp16",
        help="TensorRT precision mode"
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=2048,
        help="Workspace size in MB"
    )
    parser.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="Minimum batch size"
    )
    parser.add_argument(
        "--opt-batch",
        type=int,
        default=8,
        help="Optimal batch size"
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=16,
        help="Maximum batch size"
    )

    args = parser.parse_args()

    # Load config or use command line args
    if Path(args.config).exists():
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
    else:
        config = {
            'precision': args.precision,
            'workspace_mb': args.workspace,
            'min_batch': args.min_batch,
            'opt_batch': args.opt_batch,
            'max_batch': args.max_batch
        }

    # Build engine
    builder = TensorRTEngineBuilder(args.onnx, args.output, config)

    # Validate ONNX
    if not builder.validate_onnx():
        sys.exit(1)

    # Build engine
    if not builder.build_engine():
        sys.exit(1)

    # Save metadata
    metadata = builder.save_metadata()

    print("\n" + "="*50)
    print("TensorRT engine build completed successfully!")
    print("="*50)
    print(f"Engine path: {args.output}")
    print(f"Engine SHA256: {metadata['tensorrt_engine']['sha256'][:16]}...")
    print(f"Ready for Triton Inference Server deployment")


if __name__ == "__main__":
    main()
