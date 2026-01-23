"""
ReID Evaluator
Main class for evaluating ReID models on Market1501-format datasets using Triton Inference Server
"""

import json
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .dataset import ReIDDataset, load_query_gallery
from .metrics import evaluate_reid, compute_distance_matrix


class ReIDEvaluator:
    """
    Evaluator for ReID models served via Triton Inference Server.

    Uses the existing TritonReIDClient for embedding extraction and computes
    standard ReID metrics (CMC, mAP) following the Market1501 protocol.
    """

    def __init__(self, reid_config: Dict, eval_config: Dict):
        """
        Initialize evaluator.

        Args:
            reid_config: ReID/Triton configuration (same as pipeline uses)
            eval_config: Evaluation-specific configuration
        """
        self.reid_config = reid_config
        self.eval_config = eval_config

        # Import here to avoid circular imports
        from src.reid_client import TritonReIDClient

        print("Initializing Triton ReID client...")
        self.reid_client = TritonReIDClient(reid_config)

        # Evaluation settings
        self.batch_size = eval_config.get('evaluation', {}).get('batch_size', 16)

        # Validate batch size against Triton model config
        triton_max_batch = reid_config.get('tensorrt', {}).get('max_batch', 16)
        if self.batch_size > triton_max_batch:
            print(f"WARNING: Configured batch_size ({self.batch_size}) exceeds Triton max_batch ({triton_max_batch})")
            print(f"         Reducing batch_size to {triton_max_batch}")
            self.batch_size = triton_max_batch
        self.distance_metric = eval_config.get('evaluation', {}).get('distance_metric', 'cosine')
        self.cmc_ranks = eval_config.get('evaluation', {}).get('cmc_ranks', [1, 5, 10, 20])
        self.exclude_same_camera = eval_config.get('evaluation', {}).get('exclude_same_camera', True)

        # Output settings
        output_config = eval_config.get('output', {})
        self.results_dir = Path(output_config.get('results_dir', 'experiments/evaluation'))
        self.save_embeddings = output_config.get('save_embeddings', True)
        self.save_distance_matrix = output_config.get('save_distance_matrix', False)

    def extract_embeddings(self, dataset: ReIDDataset, desc: str = "Extracting") -> np.ndarray:
        """
        Extract embeddings for all images in a dataset.

        Args:
            dataset: ReID dataset
            desc: Progress bar description

        Returns:
            Embeddings array [N, 256]
        """
        num_images = len(dataset)
        all_embeddings = []

        # Process in batches
        num_batches = (num_images + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(num_batches), desc=desc):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_images)

            # Load batch images
            batch_images = dataset.get_batch_images(list(range(start_idx, end_idx)))

            # Extract embeddings via Triton
            embeddings = self.reid_client.infer(batch_images)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def evaluate(self,
                 query_dataset: ReIDDataset,
                 gallery_dataset: ReIDDataset,
                 experiment_name: Optional[str] = None) -> Dict:
        """
        Run full evaluation on query/gallery datasets.

        Args:
            query_dataset: Query dataset
            gallery_dataset: Gallery dataset
            experiment_name: Optional name for results directory

        Returns:
            Dictionary with evaluation results
        """
        # Create experiment directory
        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime("eval_%Y%m%d_%H%M%S")

        exp_dir = self.results_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("ReID Model Evaluation")
        print("=" * 70)
        print(f"\nQuery set: {len(query_dataset)} images, {len(query_dataset.get_unique_pids())} identities")
        print(f"Gallery set: {len(gallery_dataset)} images, {len(gallery_dataset.get_unique_pids())} identities")
        print(f"Results directory: {exp_dir}")

        start_time = time.time()

        # Extract embeddings
        print("\n[1/3] Extracting query embeddings...")
        query_features = self.extract_embeddings(query_dataset, desc="Query")

        print("\n[2/3] Extracting gallery embeddings...")
        gallery_features = self.extract_embeddings(gallery_dataset, desc="Gallery")

        extraction_time = time.time() - start_time

        # Save embeddings if requested
        if self.save_embeddings:
            print("\nSaving embeddings...")
            np.save(exp_dir / "query_embeddings.npy", query_features)
            np.save(exp_dir / "gallery_embeddings.npy", gallery_features)
            np.save(exp_dir / "query_pids.npy", query_dataset.pids)
            np.save(exp_dir / "query_camids.npy", query_dataset.camids)
            np.save(exp_dir / "gallery_pids.npy", gallery_dataset.pids)
            np.save(exp_dir / "gallery_camids.npy", gallery_dataset.camids)

        # Compute metrics
        print("\n[3/3] Computing evaluation metrics...")
        eval_start = time.time()

        results = evaluate_reid(
            query_features, gallery_features,
            query_dataset.pids, query_dataset.camids,
            gallery_dataset.pids, gallery_dataset.camids,
            metric=self.distance_metric,
            ranks=self.cmc_ranks,
            exclude_same_camera=self.exclude_same_camera
        )

        eval_time = time.time() - eval_start
        total_time = time.time() - start_time

        # Save distance matrix if requested (can be large!)
        if self.save_distance_matrix:
            print("Saving distance matrix...")
            np.save(exp_dir / "distance_matrix.npy", results['distance_matrix'])

        # Prepare results summary
        summary = {
            'experiment_name': experiment_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'dataset': {
                'query_images': len(query_dataset),
                'query_pids': len(query_dataset.get_unique_pids()),
                'gallery_images': len(gallery_dataset),
                'gallery_pids': len(gallery_dataset.get_unique_pids()),
            },
            'config': {
                'distance_metric': self.distance_metric,
                'exclude_same_camera': self.exclude_same_camera,
                'batch_size': self.batch_size,
            },
            'metrics': {
                'mAP': float(results['mAP']),
                'cmc': {str(k): float(v) for k, v in results['cmc'].items()}
            },
            'timing': {
                'extraction_time_sec': extraction_time,
                'evaluation_time_sec': eval_time,
                'total_time_sec': total_time,
                'images_per_second': (len(query_dataset) + len(gallery_dataset)) / extraction_time
            }
        }

        # Save results
        with open(exp_dir / "results.json", 'w') as f:
            json.dump(summary, f, indent=2)

        # Print results
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"\nmAP: {results['mAP'] * 100:.2f}%")
        print("\nCMC Curve:")
        for rank, acc in sorted(results['cmc'].items()):
            print(f"  Rank-{rank}: {acc * 100:.2f}%")

        print(f"\nTiming:")
        print(f"  Embedding extraction: {extraction_time:.1f}s ({summary['timing']['images_per_second']:.1f} img/s)")
        print(f"  Metric computation: {eval_time:.1f}s")
        print(f"  Total: {total_time:.1f}s")

        print(f"\nResults saved to: {exp_dir}")
        print("=" * 70)

        return summary

    def evaluate_from_embeddings(self,
                                 query_embeddings: np.ndarray,
                                 gallery_embeddings: np.ndarray,
                                 query_pids: np.ndarray,
                                 query_camids: np.ndarray,
                                 gallery_pids: np.ndarray,
                                 gallery_camids: np.ndarray,
                                 experiment_name: Optional[str] = None) -> Dict:
        """
        Evaluate using pre-extracted embeddings.

        Useful for quick re-evaluation with different settings.
        """
        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime("eval_%Y%m%d_%H%M%S")

        exp_dir = self.results_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("ReID Evaluation (from pre-extracted embeddings)")
        print("=" * 70)

        start_time = time.time()

        results = evaluate_reid(
            query_embeddings, gallery_embeddings,
            query_pids, query_camids,
            gallery_pids, gallery_camids,
            metric=self.distance_metric,
            ranks=self.cmc_ranks,
            exclude_same_camera=self.exclude_same_camera
        )

        total_time = time.time() - start_time

        summary = {
            'experiment_name': experiment_name,
            'timestamp': datetime.datetime.now().isoformat(),
            'metrics': {
                'mAP': float(results['mAP']),
                'cmc': {str(k): float(v) for k, v in results['cmc'].items()}
            },
            'timing': {
                'total_time_sec': total_time
            }
        }

        with open(exp_dir / "results.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nmAP: {results['mAP'] * 100:.2f}%")
        for rank, acc in sorted(results['cmc'].items()):
            print(f"  Rank-{rank}: {acc * 100:.2f}%")

        return summary


def run_evaluation(reid_config: Dict,
                   eval_config: Dict,
                   experiment_name: Optional[str] = None) -> Dict:
    """
    Convenience function to run full evaluation.

    Args:
        reid_config: ReID/Triton configuration
        eval_config: Evaluation configuration
        experiment_name: Optional experiment name

    Returns:
        Evaluation results dictionary
    """
    # Load datasets
    dataset_config = eval_config.get('dataset', {})
    root = Path(dataset_config.get('root', 'data'))
    dataset_format = dataset_config.get('format', 'ltcc')
    query_dir = dataset_config.get('query_dir', 'query')
    gallery_dir = dataset_config.get('gallery_dir', 'bounding_box_test')

    print("Loading datasets...")
    query, gallery = load_query_gallery(root, dataset_format, query_dir, gallery_dir)

    # Create evaluator and run
    evaluator = ReIDEvaluator(reid_config, eval_config)
    results = evaluator.evaluate(query, gallery, experiment_name)

    return results


if __name__ == "__main__":
    # Test evaluator
    import sys
    import yaml

    # Load configs
    reid_config_path = Path("configs/reid_config.yaml")
    eval_config_path = Path("configs/evaluation_config.yaml")

    if not reid_config_path.exists():
        print(f"ERROR: Config not found: {reid_config_path}")
        sys.exit(1)

    if not eval_config_path.exists():
        print(f"ERROR: Config not found: {eval_config_path}")
        sys.exit(1)

    with open(reid_config_path, 'r') as f:
        reid_config = yaml.safe_load(f)

    with open(eval_config_path, 'r') as f:
        eval_config = yaml.safe_load(f)

    # Run evaluation
    try:
        results = run_evaluation(reid_config, eval_config)
        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
