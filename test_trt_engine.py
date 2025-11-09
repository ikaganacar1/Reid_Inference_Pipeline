#!/usr/bin/env python3
"""
Quick test script to verify the TensorRT engine works correctly.
"""
import sys
import logging
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from reid_pipeline.models.reid_model import BatchReIDExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_trt_engine(engine_path: str):
    """Test TensorRT engine loading and inference."""
    logger.info("=" * 60)
    logger.info("Testing TensorRT Engine")
    logger.info("=" * 60)
    logger.info(f"Engine path: {engine_path}")

    # Check if file exists
    if not Path(engine_path).exists():
        logger.error(f"❌ Engine file not found: {engine_path}")
        return False

    logger.info(f"✅ Engine file exists (size: {Path(engine_path).stat().st_size / (1024*1024):.2f} MB)")

    try:
        # Initialize extractor with TensorRT engine
        logger.info("\nStep 1: Loading TensorRT engine...")
        extractor = BatchReIDExtractor(
            model_path=engine_path,
            device='cuda',
            embedding_dim=2048,
            batch_size=8,
            image_size=(384, 192),
            use_tensorrt=True
        )
        logger.info("✅ TensorRT engine loaded successfully!")
        logger.info(f"   Inference mode: {extractor.inference_mode}")

        # Test inference with dummy data
        logger.info("\nStep 2: Testing inference with dummy batch...")
        import torch

        # Create dummy batch (batch_size=4, channels=3, height=384, width=192)
        dummy_batch = torch.randn(4, 3, 384, 192).cuda()
        logger.info(f"   Dummy batch shape: {dummy_batch.shape}")

        # Extract features
        embeddings = extractor.extract_features_batch(dummy_batch, to_numpy=True)
        logger.info(f"✅ Inference successful!")
        logger.info(f"   Output embeddings shape: {embeddings.shape}")
        logger.info(f"   Expected shape: (4, 2048)")

        # Verify output shape
        if embeddings.shape == (4, 2048):
            logger.info("✅ Output shape matches expected!")
        else:
            logger.error(f"❌ Output shape mismatch! Expected (4, 2048), got {embeddings.shape}")
            return False

        # Verify embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        logger.info(f"   Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
        if np.allclose(norms, 1.0, atol=1e-3):
            logger.info("✅ Embeddings are L2-normalized!")
        else:
            logger.warning("⚠️  Embeddings may not be properly normalized")

        # Test different batch sizes (within optimization profile range)
        logger.info("\nStep 3: Testing different batch sizes...")
        for batch_size in [1, 2, 8, 16, 32]:
            try:
                test_batch = torch.randn(batch_size, 3, 384, 192).cuda()
                test_embeddings = extractor.extract_features_batch(test_batch, to_numpy=True)
                assert test_embeddings.shape == (batch_size, 2048), f"Shape mismatch for batch_size={batch_size}"
                logger.info(f"   ✅ Batch size {batch_size}: OK (output shape: {test_embeddings.shape})")
            except Exception as e:
                logger.error(f"   ❌ Batch size {batch_size}: Failed - {e}")
                return False

        # Get statistics
        logger.info("\nStep 4: Extractor statistics:")
        stats = extractor.get_statistics()
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")

        # Cleanup
        extractor.cleanup()
        logger.info("\n✅ Cleanup complete")

        logger.info("\n" + "=" * 60)
        logger.info("✅ All tests passed! TensorRT engine is working correctly.")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    engine_path = "/home/ika/yzlm/Reid_Inference_Pipeline/models/lttc_0.1.4.49.engine"
    success = test_trt_engine(engine_path)
    sys.exit(0 if success else 1)
