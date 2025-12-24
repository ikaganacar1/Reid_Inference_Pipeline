"""
Market-1501 Evaluation Pipeline

Main orchestrator for dataset evaluation. Coordinates dataset loading, gallery building,
query processing, metrics computation, and results aggregation.
"""

import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, Callable
import logging

from .dataset_loader import Market1501Dataset
from .metrics_calculator import ReIDMetricsCalculator
from .gallery_tracker import GalleryDecisionTracker
from .pipeline_simulator import EvaluationPipelineSimulator
from ..gallery.gallery_manager import GalleryManager
from ..models.reid_model import BatchReIDExtractor

logger = logging.getLogger(__name__)


class Market1501EvaluationPipeline:
    """
    Complete evaluation pipeline for Market-1501 dataset.

    Runs full pipeline simulation and computes both standard metrics
    (mAP, CMC) and gallery simulation statistics.
    """

    def __init__(self,
                 dataset_path: Path,
                 reid_model_path: str,
                 config: Dict,
                 progress_callback: Optional[Callable] = None):
        """
        Initialize evaluation pipeline.

        Args:
            dataset_path: Path to Market-1501 dataset
            reid_model_path: Path to ReID model (.pth or .engine)
            config: Configuration dictionary
            progress_callback: Optional callback(current, total, message)
        """
        self.dataset_path = Path(dataset_path)
        self.reid_model_path = reid_model_path
        self.config = config
        self.progress_callback = progress_callback

        # Load dataset
        logger.info(f"Loading Market-1501 dataset from {dataset_path}")
        self.dataset = Market1501Dataset(dataset_path)

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize ReID extractor, gallery manager, and trackers"""
        # ReID extractor (initialize first to detect embedding dimension)
        logger.info(f"Initializing ReID extractor: {self.reid_model_path}")
        self.reid_extractor = BatchReIDExtractor(
            model_path=self.reid_model_path,
            embedding_dim=self.config.get('embedding_dim', 2048),
            batch_size=self.config.get('reid_batch_size', 16),
            device=self.config.get('device', 'cuda'),
            use_tensorrt=self.config.get('use_tensorrt', False)
        )

        # Get actual embedding dimension from extractor (may differ from config)
        actual_embedding_dim = self.reid_extractor.embedding_dim
        logger.info(f"Using embedding dimension: {actual_embedding_dim}")

        # Gallery manager (use actual embedding dimension from model)
        self.gallery_manager = GalleryManager(
            max_gallery_size=self.config.get('gallery_max_size', 1500),
            similarity_threshold_match=self.config.get('reid_threshold_match', 0.70),
            similarity_threshold_new=self.config.get('reid_threshold_new', 0.50),
            embedding_dim=actual_embedding_dim,
            logger=logger
        )

        # Pipeline simulator
        self.simulator = EvaluationPipelineSimulator(
            reid_extractor=self.reid_extractor,
            gallery_manager=self.gallery_manager
        )

        # Trackers
        self.gallery_tracker = GalleryDecisionTracker()
        self.metrics_calculator = ReIDMetricsCalculator()

    def run_evaluation(self, subset_size: Optional[int] = None) -> Dict:
        """
        Run complete evaluation pipeline.

        Args:
            subset_size: Optional limit on number of images (for testing)

        Returns:
            Complete results dictionary with metrics and statistics
        """
        logger.info("=" * 60)
        logger.info("Starting Market-1501 Evaluation")
        logger.info("=" * 60)

        start_time = time.time()

        # Phase 1: Build gallery
        logger.info("\n[Phase 1/3] Building gallery from gallery set...")
        gallery_results = self._build_gallery_phase(subset_size)

        # Phase 2: Process queries
        logger.info("\n[Phase 2/3] Processing query set...")
        query_results, query_embeddings, query_metadata = self._query_phase(subset_size)

        # Phase 3: Compute standard metrics
        logger.info("\n[Phase 3/3] Computing evaluation metrics...")
        metrics = self._compute_metrics_phase(query_embeddings, query_metadata, query_results)

        # Aggregate results
        total_time = time.time() - start_time
        results = self._aggregate_results(gallery_results, query_results, metrics, total_time)

        logger.info(f"\nEvaluation completed in {total_time:.1f}s")
        logger.info("=" * 60)

        return results

    def _build_gallery_phase(self, subset_size: Optional[int]) -> Dict:
        """Phase 1: Build gallery from gallery set images"""
        gallery_images = self.dataset.get_gallery(with_images=True)

        if subset_size:
            gallery_images = gallery_images[:subset_size]

        stats = self.simulator.build_gallery_from_images(
            gallery_images,
            progress_callback=self.progress_callback
        )

        logger.info(f"Gallery built: {stats['gallery_size']} identities")
        return stats

    def _query_phase(self, subset_size: Optional[int]):
        """Phase 2: Process queries and extract embeddings"""
        query_images = self.dataset.get_queries(with_images=True)

        if subset_size:
            query_images = query_images[:subset_size]

        # Process queries through pipeline
        query_results = self.simulator.evaluate_queries(
            query_images,
            progress_callback=self.progress_callback
        )

        # Extract embeddings and metadata for metrics computation
        query_embeddings = np.array([r['embedding'] for r in query_results])
        query_metadata = [
            {
                'person_id': r['query_person_id'],
                'camera_id': r['query_camera_id'],
                'query_id': r['query_id']
            }
            for r in query_results
        ]

        # Track gallery decisions
        for result in query_results:
            self.gallery_tracker.record_decision(
                query_id=result['query_id'],
                decision=result['decision'],
                person_id=result['matched_person_id'],
                similarity=result['similarity'],
                gallery_size=len(self.gallery_manager.gallery),
                frame_id=0
            )

        return query_results, query_embeddings, query_metadata

    def _compute_metrics_phase(self, query_embeddings, query_metadata, query_results):
        """Phase 3: Compute standard ReID metrics"""
        # Get gallery embeddings
        gallery_images = self.dataset.get_gallery(with_images=True)
        gallery_embeddings = []
        gallery_metadata = []

        logger.info("Extracting gallery embeddings for metric computation...")
        batch_size = 32
        for i in range(0, len(gallery_images), batch_size):
            batch = gallery_images[i:i+batch_size]
            for img_data in batch:
                h, w = img_data['image'].shape[:2]
                bbox = np.array([0, 0, w, h])
                emb, valid = self.reid_extractor.extract_features_from_frame(
                    img_data['image'], [bbox]
                )
                if valid[0]:
                    gallery_embeddings.append(emb[0])
                    gallery_metadata.append({
                        'person_id': img_data['person_id'],
                        'camera_id': img_data['camera_id']
                    })

        gallery_embeddings = np.array(gallery_embeddings)

        # Prepare arrays for metrics
        query_person_ids = np.array([m['person_id'] for m in query_metadata])
        query_camera_ids = np.array([m['camera_id'] for m in query_metadata])
        gallery_person_ids = np.array([m['person_id'] for m in gallery_metadata])
        gallery_camera_ids = np.array([m['camera_id'] for m in gallery_metadata])

        # Compute metrics
        metrics = self.metrics_calculator.evaluate(
            query_embeddings,
            query_person_ids,
            query_camera_ids,
            gallery_embeddings,
            gallery_person_ids,
            gallery_camera_ids,
            metric='cosine',
            top_k=10
        )

        return metrics

    def _aggregate_results(self, gallery_stats, query_results, metrics, total_time):
        """Aggregate all results into final dictionary"""
        # Gallery statistics
        gallery_metrics = self.gallery_tracker.get_statistics()

        # Performance statistics
        perf_stats = self.simulator.get_performance_stats()
        perf_stats['total_evaluation_time'] = total_time

        # Per-query results (limited for storage)
        per_query = self.gallery_tracker.get_per_query_results()

        results = {
            'standard_metrics': metrics,
            'gallery_metrics': gallery_metrics,
            'performance_metrics': perf_stats,
            'per_query_results': per_query[:100],  # Limit to first 100 for storage
            'dataset_info': {
                'num_queries': len(query_results),
                'num_gallery': len(self.dataset.gallery_metadata),
                'num_identities_query': len(set(r['query_person_id'] for r in query_results)),
                'final_gallery_size': len(self.gallery_manager.gallery)
            }
        }

        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Standard Metrics:")
        logger.info(f"  mAP: {metrics['mAP']:.2f}%")
        logger.info(f"  Rank-1: {metrics['rank1']:.1f}%")
        logger.info(f"  Rank-5: {metrics['rank5']:.1f}%")
        logger.info(f"  Rank-10: {metrics['rank10']:.1f}%")
        logger.info(f"\nGallery Statistics:")
        logger.info(f"  MATCH: {gallery_metrics['total_match']} ({gallery_metrics['match_rate']*100:.1f}%)")
        logger.info(f"  UNCERTAIN: {gallery_metrics['total_uncertain']} ({gallery_metrics['uncertain_rate']*100:.1f}%)")
        logger.info(f"  NEW: {gallery_metrics['total_new']} ({gallery_metrics['new_rate']*100:.1f}%)")
        logger.info(f"  Final gallery size: {gallery_metrics['final_gallery_size']}")
        logger.info(f"\nPerformance:")
        logger.info(f"  FPS: {perf_stats['fps']:.1f}")
        logger.info(f"  Avg ReID time: {perf_stats['avg_reid_time']*1000:.1f}ms")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info("=" * 60)

        return results
