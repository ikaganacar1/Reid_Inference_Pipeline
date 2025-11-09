#!/usr/bin/env python3
"""
Script to rebuild TensorRT engines with the correct version.
Run this inside the worker container:
  docker compose exec worker python3 /app/rebuild_engines.py
"""
import sys
import torch
import tensorrt as trt
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from reid_pipeline.models.reid_model import ReIDModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_pytorch_to_onnx(
    pytorch_model_path: str,
    onnx_path: str,
    input_shape: tuple = (1, 3, 384, 192),
    device: str = 'cuda'
):
    """
    Export PyTorch model to ONNX format.

    Args:
        pytorch_model_path: Path to .pth model
        onnx_path: Output path for .onnx model
        input_shape: Input tensor shape (batch, channels, height, width)
        device: Device for export
    """
    logger.info(f"Exporting PyTorch model to ONNX: {pytorch_model_path} -> {onnx_path}")

    # Load PyTorch model
    model = ReIDModel(embedding_dim=2048).to(device)
    model.eval()

    try:
        checkpoint = torch.load(pytorch_model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        logger.info("✅ PyTorch model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load PyTorch model: {e}")
        return False

    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)

    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        logger.info(f"✅ ONNX model exported: {onnx_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to export ONNX: {e}")
        return False


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = 'fp16',
    max_batch_size: int = 32,
    min_batch_size: int = 1,
    opt_batch_size: int = 8,
    workspace_size: int = 2 << 30,  # 2GB
    verbose: bool = False
):
    """
    Build TensorRT engine from ONNX model with detailed timing and logging.

    Args:
        onnx_path: Path to .onnx model
        engine_path: Output path for .engine file
        precision: 'fp32', 'fp16', or 'int8'
        max_batch_size: Maximum batch size
        min_batch_size: Minimum batch size
        opt_batch_size: Optimal batch size
        workspace_size: Workspace size in bytes
        verbose: Enable verbose TensorRT logging
    """
    import time
    start_time = time.time()

    def log_time(message):
        elapsed = time.time() - start_time
        logger.info(f"[{elapsed:6.2f}s] {message}")

    log_time(f"Building TensorRT engine: {onnx_path} -> {engine_path}")
    logger.info(f"TensorRT version: {trt.__version__}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Batch size range: min={min_batch_size}, opt={opt_batch_size}, max={max_batch_size}")

    # Create builder with verbose logging if requested
    log_level = trt.Logger.VERBOSE if verbose else trt.Logger.INFO
    TRT_LOGGER = trt.Logger(log_level)

    log_time("Creating TensorRT builder...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    log_time(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        onnx_data = f.read()
        log_time(f"Read ONNX file ({len(onnx_data)/(1024*1024):.2f} MB)")
        if not parser.parse(onnx_data):
            logger.error("Failed to parse ONNX model")
            for error in range(parser.num_errors):
                logger.error(f"  Error {error}: {parser.get_error(error)}")
            return False
    log_time("✅ ONNX parsing complete")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    # Enable detailed profiling
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    # Set precision
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            log_time("✅ FP16 mode enabled")
        else:
            logger.warning("⚠️  FP16 not supported on this platform, using FP32")
    elif precision == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            log_time("✅ INT8 mode enabled (requires calibration)")
        else:
            logger.warning("⚠️  INT8 not supported on this platform, using FP32")

    # Create optimization profile for dynamic batch sizes
    log_time("Creating optimization profile for dynamic shapes...")
    profile = builder.create_optimization_profile()

    # Get input tensor name and shape from network
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = input_tensor.shape

    logger.info(f"Input tensor: name='{input_name}', shape={input_shape}")

    # Set dynamic batch dimension (first dimension)
    # For ReID models: input shape is [batch, 3, 384, 192]
    min_shape = (min_batch_size, input_shape[1], input_shape[2], input_shape[3])
    opt_shape = (opt_batch_size, input_shape[1], input_shape[2], input_shape[3])
    max_shape = (max_batch_size, input_shape[1], input_shape[2], input_shape[3])

    logger.info(f"Setting optimization profile:")
    logger.info(f"  Min shape: {min_shape}")
    logger.info(f"  Opt shape: {opt_shape}")
    logger.info(f"  Max shape: {max_shape}")

    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    log_time("✅ Optimization profile added")

    # Build engine
    log_time("Building engine (this may take 2-5 minutes on first build)...")
    logger.info("NOTE: TensorRT is compiling kernels for your GPU - this is normal on first build")
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            logger.error("❌ Failed to build engine")
            return False

        log_time("✅ Engine compilation complete! Saving to disk...")

        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        engine_size_mb = Path(engine_path).stat().st_size / (1024*1024)
        log_time(f"✅ TensorRT engine saved: {engine_path}")
        logger.info(f"   Engine size: {engine_size_mb:.2f} MB")
        logger.info(f"   Total build time: {time.time() - start_time:.1f} seconds")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to build engine: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to rebuild all engines."""
    logger.info("=" * 60)
    logger.info("TensorRT Engine Rebuild Script")
    logger.info("=" * 60)
    logger.info(f"TensorRT version: {trt.__version__}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    # Find all .pth models in /app/models
    models_dir = Path("/app/models")
    if not models_dir.exists():
        logger.error(f"❌ Models directory not found: {models_dir}")
        return

    pth_models = list(models_dir.glob("*.pth"))

    if not pth_models:
        logger.warning(f"⚠️  No .pth models found in {models_dir}")
        logger.info("If you have ONNX models, you can convert them directly:")
        for onnx_model in models_dir.glob("*.onnx"):
            engine_path = onnx_model.with_suffix('.engine')
            logger.info(f"\nConverting: {onnx_model.name}")
            build_tensorrt_engine(str(onnx_model), str(engine_path), precision='fp16')
        return

    logger.info(f"\nFound {len(pth_models)} PyTorch model(s) to convert:")
    for model in pth_models:
        logger.info(f"  - {model.name}")

    # Convert each model
    for pth_model in pth_models:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {pth_model.name}")
        logger.info(f"{'=' * 60}")

        onnx_path = pth_model.with_suffix('.onnx')
        engine_path = pth_model.with_suffix('.engine')

        # Step 1: Export to ONNX
        if not onnx_path.exists():
            logger.info("\nStep 1/2: Exporting to ONNX...")
            if not export_pytorch_to_onnx(str(pth_model), str(onnx_path)):
                logger.error(f"❌ Skipping {pth_model.name} due to ONNX export failure")
                continue
        else:
            logger.info(f"✅ ONNX model already exists: {onnx_path.name}")

        # Step 2: Build TensorRT engine
        logger.info("\nStep 2/2: Building TensorRT engine...")
        if not build_tensorrt_engine(str(onnx_path), str(engine_path), precision='fp16'):
            logger.error(f"❌ Failed to build engine for {pth_model.name}")
            continue

        logger.info(f"\n✅ Successfully rebuilt engine: {engine_path.name}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ All engines rebuilt successfully!")
    logger.info("=" * 60)
    logger.info("\nYou can now use these engines with the pipeline:")
    for engine in models_dir.glob("*.engine"):
        logger.info(f"  - {engine.name}")


if __name__ == "__main__":
    # Auto-detect if running inside Docker container
    import os
    if os.path.exists('/app/models'):
        # Running inside Docker container
        onnx_path = '/app/models/lttc_0.1.4.49.onnx'
        engine_path = '/app/models/lttc_0.1.4.49.engine'
    else:
        # Running on host machine
        onnx_path = '/home/ika/yzlm/Reid_Inference_Pipeline/models/lttc_0.1.4.49.onnx'
        engine_path = '/home/ika/yzlm/Reid_Inference_Pipeline/models/lttc_0.1.4.49.engine'

    build_tensorrt_engine(onnx_path, engine_path, precision='fp16')
