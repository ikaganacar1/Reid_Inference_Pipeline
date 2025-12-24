#!/usr/bin/env python3
"""
Local Evaluation Script for Market-1501 Dataset

Run evaluation locally without Docker worker.
Uses conda environment tensorrt_blackwell with ONNX Runtime.

Usage:
    conda activate tensorrt_blackwell
    python run_local_evaluation.py --dataset-path ./data --model models/lttc_0.1.4.49.onnx
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from reid_pipeline.evaluation.evaluation_pipeline import Market1501EvaluationPipeline


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    parser = argparse.ArgumentParser(description='Run Market-1501 Evaluation Locally')

    parser.add_argument(
        '--dataset-path', '-d',
        type=str,
        default='./data',
        help='Path to Market-1501 dataset (default: ./data)'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='models/lttc_0.1.4.49.onnx',
        help='Path to ReID model (.onnx, .pth, or .engine)'
    )

    parser.add_argument(
        '--subset-size', '-s',
        type=int,
        default=None,
        help='Limit number of images for quick testing (default: all)'
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for ReID inference (default: 32)'
    )

    parser.add_argument(
        '--gallery-max-size',
        type=int,
        default=1500,
        help='Maximum gallery size (default: 1500)'
    )

    parser.add_argument(
        '--threshold-match',
        type=float,
        default=0.70,
        help='Match threshold (default: 0.70)'
    )

    parser.add_argument(
        '--threshold-new',
        type=float,
        default=0.50,
        help='New identity threshold (default: 0.50)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference (default: cuda)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate paths
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        sys.exit(1)

    # Configuration
    config = {
        'reid_batch_size': args.batch_size,
        'gallery_max_size': args.gallery_max_size,
        'reid_threshold_match': args.threshold_match,
        'reid_threshold_new': args.threshold_new,
        'device': args.device,
        'embedding_dim': 256,  # ONNX model outputs 256D
    }

    logger.info("=" * 60)
    logger.info("LOCAL MARKET-1501 EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Subset size: {args.subset_size or 'Full dataset'}")
    logger.info(f"Config: {config}")
    logger.info("=" * 60)

    # Progress callback
    def progress_callback(current, total, message):
        if total > 0:
            pct = current / total * 100
            logger.info(f"[{pct:5.1f}%] {message}")

    try:
        # Initialize pipeline
        logger.info("Initializing evaluation pipeline...")
        pipeline = Market1501EvaluationPipeline(
            dataset_path=dataset_path,
            reid_model_path=str(model_path),
            config=config,
            progress_callback=progress_callback
        )

        # Run evaluation
        logger.info("Starting evaluation...")
        results = pipeline.run_evaluation(subset_size=args.subset_size)

        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)

        std_metrics = results.get('standard_metrics', {})
        logger.info(f"mAP: {std_metrics.get('mAP', 0):.2f}%")
        logger.info(f"Rank-1: {std_metrics.get('rank1', 0):.2f}%")
        logger.info(f"Rank-5: {std_metrics.get('rank5', 0):.2f}%")
        logger.info(f"Rank-10: {std_metrics.get('rank10', 0):.2f}%")

        gallery_metrics = results.get('gallery_metrics', {})
        logger.info(f"\nGallery Statistics:")
        logger.info(f"  MATCH: {gallery_metrics.get('total_match', 0)}")
        logger.info(f"  UNCERTAIN: {gallery_metrics.get('total_uncertain', 0)}")
        logger.info(f"  NEW: {gallery_metrics.get('total_new', 0)}")

        perf_metrics = results.get('performance_metrics', {})
        logger.info(f"\nPerformance:")
        logger.info(f"  FPS: {perf_metrics.get('fps', 0):.1f}")
        logger.info(f"  Total time: {perf_metrics.get('total_evaluation_time', 0):.1f}s")

        logger.info("=" * 60)
        logger.info("Evaluation complete!")

        return results

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
