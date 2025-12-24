"""Test Market-1501 Evaluation Pipeline Logic (without real model)

This test validates that the evaluation pipeline logic works correctly
by using a mock ReID extractor that returns consistent embeddings.
"""
import sys
from pathlib import Path
import logging
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reid_pipeline.evaluation.evaluation_pipeline import Market1501EvaluationPipeline
from reid_pipeline.models.reid_model import BatchReIDExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockReIDExtractor:
    """Mock ReID extractor that returns consistent embeddings for testing"""

    def __init__(self, embedding_dim=2048):
        self.embedding_dim = embedding_dim
        self.logger = logger
        # Cache for consistent embeddings per person
        self.person_embeddings = {}
        np.random.seed(42)  # For reproducibility

    def extract_features_from_frame(self, frame, bboxes):
        """Extract mock features that are consistent per image"""
        embeddings = []
        valid = []

        for bbox in bboxes:
            # Create a simple hash from bbox coordinates to get consistent embedding
            # In real scenario, same person in different cameras will have similar embeddings
            bbox_hash = hash(tuple(bbox.astype(int)))

            if bbox_hash not in self.person_embeddings:
                # Generate random embedding for this "person"
                emb = np.random.randn(self.embedding_dim).astype(np.float32)
                # L2 normalize
                emb = emb / (np.linalg.norm(emb) + 1e-12)
                self.person_embeddings[bbox_hash] = emb

            embeddings.append(self.person_embeddings[bbox_hash])
            valid.append(True)

        return np.array(embeddings), np.array(valid)


def main():
    """Run evaluation on Market-1501 dataset with mock extractor"""

    # Dataset path
    dataset_path = PROJECT_ROOT / "data"

    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return 1

    # Configuration (no real model needed)
    config = {
        'embedding_dim': 2048,
        'reid_batch_size': 16,
        'device': 'cuda',
        'use_tensorrt': False,
        'gallery_max_size': 1500,
        'reid_threshold_match': 0.70,
        'reid_threshold_new': 0.50
    }

    try:
        # Initialize pipeline
        logger.info("Initializing evaluation pipeline with mock extractor...")

        # Create pipeline but replace extractor with mock
        from reid_pipeline.evaluation.dataset_loader import Market1501Dataset
        from reid_pipeline.evaluation.metrics_calculator import ReIDMetricsCalculator
        from reid_pipeline.evaluation.gallery_tracker import GalleryDecisionTracker
        from reid_pipeline.evaluation.pipeline_simulator import EvaluationPipelineSimulator
        from reid_pipeline.gallery.gallery_manager import GalleryManager

        # Load dataset
        logger.info(f"Loading Market-1501 dataset from {dataset_path}")
        dataset = Market1501Dataset(dataset_path)

        # Create mock extractor
        reid_extractor = MockReIDExtractor(embedding_dim=config['embedding_dim'])

        # Create gallery manager
        gallery_manager = GalleryManager(
            max_gallery_size=config['gallery_max_size'],
            similarity_threshold_match=config['reid_threshold_match'],
            similarity_threshold_new=config['reid_threshold_new'],
            logger=logger
        )

        # Create simulator
        simulator = EvaluationPipelineSimulator(
            reid_extractor=reid_extractor,
            gallery_manager=gallery_manager
        )

        # Create trackers
        gallery_tracker = GalleryDecisionTracker()
        metrics_calculator = ReIDMetricsCalculator()

        # Run evaluation on small subset
        logger.info("Running evaluation on subset (20 queries, 100 gallery)...")

        # Build gallery
        gallery_images = dataset.get_gallery(with_images=True)[:100]
        logger.info(f"Building gallery from {len(gallery_images)} images...")

        stats = simulator.build_gallery_from_images(gallery_images)
        logger.info(f"Gallery built: {stats['gallery_size']} identities")

        # Process queries
        query_images = dataset.get_queries(with_images=True)[:20]
        logger.info(f"Processing {len(query_images)} queries...")

        query_results = simulator.evaluate_queries(query_images)

        # Track gallery decisions
        for result in query_results:
            gallery_tracker.record_decision(
                query_id=result['query_id'],
                decision=result['decision'],
                person_id=result['matched_person_id'],
                similarity=result['similarity'],
                gallery_size=len(gallery_manager.gallery),
                frame_id=0
            )

        # Get gallery statistics
        gallery_metrics = gallery_tracker.get_statistics()

        # Print results
        print("\n" + "=" * 70)
        print("EVALUATION LOGIC TEST RESULTS")
        print("=" * 70)
        print(f"\nGallery Simulation Statistics:")
        print(f"  MATCH decisions:     {gallery_metrics['total_match']} ({gallery_metrics['match_rate']*100:.1f}%)")
        print(f"  UNCERTAIN decisions: {gallery_metrics['total_uncertain']} ({gallery_metrics['uncertain_rate']*100:.1f}%)")
        print(f"  NEW decisions:       {gallery_metrics['total_new']} ({gallery_metrics['new_rate']*100:.1f}%)")
        print(f"  Final gallery size:  {gallery_metrics['final_gallery_size']}")

        print("\n" + "=" * 70)
        print("✓ Evaluation pipeline logic test completed successfully!")
        print("=" * 70)
        print("\nNOTE: This test uses mock embeddings, not real model.")
        print("To test with real model, ensure compatible TensorRT engine is available.")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
