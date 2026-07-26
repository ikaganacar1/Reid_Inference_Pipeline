#!/usr/bin/env python3
"""
ReID Dataset Evaluation Script
Evaluates TAO ReID model on Market1501-format datasets using Triton Inference Server

Usage:
    python scripts/evaluate_dataset.py
    python scripts/evaluate_dataset.py --data-root data --experiment-name ltcc_eval
    python scripts/evaluate_dataset.py --from-embeddings experiments/evaluation/ltcc_eval
"""

import argparse
import importlib
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate ReID model on Market1501-format dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset options
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing dataset (query/, bounding_box_test/)"
    )
    parser.add_argument(
        "--format",
        choices=["market1501", "ltcc"],
        default="ltcc",
        help="Dataset format"
    )
    parser.add_argument(
        "--query-dir",
        default="query",
        help="Query directory name"
    )
    parser.add_argument(
        "--gallery-dir",
        default="bounding_box_test",
        help="Gallery directory name"
    )

    # Configuration
    parser.add_argument(
        "--reid-config",
        type=Path,
        default=Path("configs/reid_config.yaml"),
        help="Path to ReID/Triton config"
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=Path("configs/evaluation_config.yaml"),
        help="Path to evaluation config"
    )

    # Experiment options
    parser.add_argument(
        "--experiment-name",
        help="Custom experiment name (default: auto-generated timestamp)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/evaluation"),
        help="Base directory for evaluation results"
    )

    # Evaluation options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction (max: 16 for current model)"
    )
    parser.add_argument(
        "--distance-metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric for similarity computation"
    )
    parser.add_argument(
        "--include-same-camera",
        action="store_true",
        help="Include same-camera matches (not standard Market1501 protocol)"
    )

    # Re-evaluation from saved embeddings
    parser.add_argument(
        "--from-embeddings",
        type=Path,
        help="Re-evaluate using embeddings from a previous run"
    )

    # Output options
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        default=True,
        help="Save extracted embeddings for later analysis"
    )
    parser.add_argument(
        "--no-save-embeddings",
        action="store_true",
        help="Don't save embeddings"
    )
    parser.add_argument(
        "--save-distance-matrix",
        action="store_true",
        help="Save full distance matrix (can be large!)"
    )

    return parser.parse_args()


def load_configs(args):
    """Load and merge configuration files with command line arguments."""
    # Load ReID config
    if not args.reid_config.exists():
        print(f"ERROR: ReID config not found: {args.reid_config}")
        sys.exit(1)

    with open(args.reid_config, 'r') as f:
        reid_config = yaml.safe_load(f)

    # Load evaluation config
    if args.eval_config.exists():
        with open(args.eval_config, 'r') as f:
            eval_config = yaml.safe_load(f)
    else:
        print("WARNING: Evaluation config not found, using defaults")
        eval_config = {}

    # Override with command line arguments
    if 'dataset' not in eval_config:
        eval_config['dataset'] = {}
    eval_config['dataset']['root'] = str(args.data_root)
    eval_config['dataset']['format'] = args.format
    eval_config['dataset']['query_dir'] = args.query_dir
    eval_config['dataset']['gallery_dir'] = args.gallery_dir

    if 'evaluation' not in eval_config:
        eval_config['evaluation'] = {}
    eval_config['evaluation']['batch_size'] = args.batch_size
    eval_config['evaluation']['distance_metric'] = args.distance_metric
    eval_config['evaluation']['exclude_same_camera'] = not args.include_same_camera

    if 'output' not in eval_config:
        eval_config['output'] = {}
    eval_config['output']['results_dir'] = str(args.results_dir)
    eval_config['output']['save_embeddings'] = args.save_embeddings and not args.no_save_embeddings
    eval_config['output']['save_distance_matrix'] = args.save_distance_matrix

    return reid_config, eval_config


def validate_environment(reid_config):
    """Validate environment setup."""
    print("Validating environment...")

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("  WARNING: CUDA not available")
    except ImportError:
        print("  ERROR: PyTorch not installed")
        sys.exit(1)

    backend = str(reid_config.get("backend", "triton")).lower()
    if backend in {"triton", "triton_http", "onnxruntime_triton"}:
        try:
            importlib.import_module("tritonclient.http")
            print("  Triton client: OK")
        except ImportError:
            print("  ERROR: tritonclient not installed")
            print("  Install with: pip install tritonclient[http]")
            sys.exit(1)

    # Check tqdm
    try:
        importlib.import_module("tqdm")
        print("  tqdm: OK")
    except ImportError:
        print("  ERROR: tqdm not installed")
        print("  Install with: pip install tqdm")
        sys.exit(1)

    print("  All dependencies validated\n")


def run_from_embeddings(args, eval_config):
    """Re-evaluate using pre-extracted embeddings."""
    emb_dir = args.from_embeddings

    print(f"Loading embeddings from: {emb_dir}")

    # Load embeddings and metadata
    query_features = np.load(emb_dir / "query_embeddings.npy")
    gallery_features = np.load(emb_dir / "gallery_embeddings.npy")
    query_pids = np.load(emb_dir / "query_pids.npy")
    query_camids = np.load(emb_dir / "query_camids.npy")
    gallery_pids = np.load(emb_dir / "gallery_pids.npy")
    gallery_camids = np.load(emb_dir / "gallery_camids.npy")

    print(f"  Query: {len(query_features)} embeddings")
    print(f"  Gallery: {len(gallery_features)} embeddings")

    # Import evaluation function
    from src.evaluation.metrics import evaluate_reid

    # Run evaluation
    distance_metric = eval_config['evaluation']['distance_metric']
    exclude_same_camera = eval_config['evaluation']['exclude_same_camera']
    cmc_ranks = eval_config['evaluation'].get('cmc_ranks', [1, 5, 10, 20])

    print(f"\nEvaluating with {distance_metric} distance...")
    print(f"  Exclude same camera: {exclude_same_camera}")

    results = evaluate_reid(
        query_features, gallery_features,
        query_pids, query_camids,
        gallery_pids, gallery_camids,
        metric=distance_metric,
        ranks=cmc_ranks,
        exclude_same_camera=exclude_same_camera
    )

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nmAP: {results['mAP'] * 100:.2f}%")
    print("\nCMC Curve:")
    for rank, acc in sorted(results['cmc'].items()):
        print(f"  Rank-{rank}: {acc * 100:.2f}%")
    print("=" * 70)

    return results


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 70)
    print("ReID Dataset Evaluation")
    print("=" * 70)

    # Handle re-evaluation from embeddings
    if args.from_embeddings:
        if not args.from_embeddings.exists():
            print(f"ERROR: Embeddings directory not found: {args.from_embeddings}")
            sys.exit(1)

        _, eval_config = load_configs(args)
        run_from_embeddings(args, eval_config)
        return

    # Load configs
    reid_config, eval_config = load_configs(args)

    # Validate environment
    validate_environment(reid_config)

    # Validate data directory
    if not args.data_root.exists():
        print(f"ERROR: Data directory not found: {args.data_root}")
        sys.exit(1)

    query_dir = args.data_root / args.query_dir
    gallery_dir = args.data_root / args.gallery_dir

    if not query_dir.exists():
        print(f"ERROR: Query directory not found: {query_dir}")
        sys.exit(1)

    if not gallery_dir.exists():
        print(f"ERROR: Gallery directory not found: {gallery_dir}")
        sys.exit(1)

    print(f"Dataset root: {args.data_root}")
    print(f"  Query: {query_dir}")
    print(f"  Gallery: {gallery_dir}")
    print(f"  Format: {args.format}")

    # Run evaluation
    try:
        from src.evaluation.evaluator import run_evaluation

        run_evaluation(
            reid_config,
            eval_config,
            experiment_name=args.experiment_name
        )

        print("\nEvaluation completed successfully!")

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(1)

    except Exception as e:
        print("\n\nERROR: Evaluation failed")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
