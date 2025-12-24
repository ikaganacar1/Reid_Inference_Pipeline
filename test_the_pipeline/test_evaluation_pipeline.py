"""Test Market-1501 Evaluation Pipeline with real dataset"""
import sys
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reid_pipeline.evaluation.evaluation_pipeline import Market1501EvaluationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run evaluation on Market-1501 dataset"""

    # Dataset path
    dataset_path = PROJECT_ROOT / "data"

    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return 1

    # Model path (use TensorRT engine which is available)
    reid_model_path = "models/lttc_0.1.4.49.engine"  # TensorRT model file

    # Configuration
    config = {
        'embedding_dim': 2048,
        'reid_batch_size': 16,
        'device': 'cuda',
        'use_tensorrt': True,  # Use TensorRT engine
        'gallery_max_size': 1500,
        'reid_threshold_match': 0.70,
        'reid_threshold_new': 0.50
    }

    try:
        # Initialize pipeline
        logger.info("Initializing evaluation pipeline...")
        pipeline = Market1501EvaluationPipeline(
            dataset_path=dataset_path,
            reid_model_path=reid_model_path,
            config=config
        )

        # Run evaluation on subset for quick testing
        logger.info("Running evaluation on subset (50 queries, 200 gallery)...")
        results = pipeline.run_evaluation(subset_size=50)

        # Print results
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"\nStandard ReID Metrics:")
        print(f"  mAP:     {results['standard_metrics']['mAP']:.2f}%")
        print(f"  Rank-1:  {results['standard_metrics']['rank1']:.1f}%")
        print(f"  Rank-5:  {results['standard_metrics']['rank5']:.1f}%")
        print(f"  Rank-10: {results['standard_metrics']['rank10']:.1f}%")

        print(f"\nGallery Simulation Statistics:")
        gstats = results['gallery_metrics']
        print(f"  MATCH decisions:     {gstats['total_match']} ({gstats['match_rate']*100:.1f}%)")
        print(f"  UNCERTAIN decisions: {gstats['total_uncertain']} ({gstats['uncertain_rate']*100:.1f}%)")
        print(f"  NEW decisions:       {gstats['total_new']} ({gstats['new_rate']*100:.1f}%)")
        print(f"  Final gallery size:  {gstats['final_gallery_size']}")

        print(f"\nPerformance Metrics:")
        pstats = results['performance_metrics']
        print(f"  FPS:              {pstats['fps']:.1f}")
        print(f"  Avg ReID time:    {pstats['avg_reid_time']*1000:.1f} ms")
        print(f"  Avg Gallery time: {pstats['avg_gallery_time']*1000:.1f} ms")
        print(f"  Total time:       {pstats['total_evaluation_time']:.1f} s")

        print("\n" + "=" * 70)
        print("✓ Evaluation completed successfully!")
        print("=" * 70)

        return 0

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        logger.info("Please ensure reid_model.pth exists in models/ directory")
        logger.info("You can skip this test if models are not available yet")
        return 1
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
