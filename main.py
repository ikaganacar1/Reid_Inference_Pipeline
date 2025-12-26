#!/usr/bin/env python3
"""
YOLO + TAO ReID Inference Pipeline
Main CLI entry point for person detection, re-identification, and tracking
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.pipeline import ReIDPipeline
from src.logger import ExperimentLogger
from src.utils.config_loader import load_all_configs


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="YOLO + TAO ReID + BoxMOT Tracking Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output visualization video (optional)"
    )

    # Configuration
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing configuration files"
    )

    # Experiment tracking
    parser.add_argument(
        "--experiment-name",
        help="Custom experiment name (default: auto-generated timestamp)"
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments"),
        help="Base directory for experiments"
    )

    # Processing options
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to process (for testing)"
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable visualization output"
    )

    return parser.parse_args()


def validate_environment():
    """Validate environment setup"""
    print("Validating environment...")

    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
        else:
            print("  WARNING: CUDA not available, using CPU")
    except ImportError:
        print("  ERROR: PyTorch not installed")
        sys.exit(1)

    # Check TensorRT
    try:
        import tensorrt as trt
        print(f"  ✓ TensorRT {trt.__version__}")
    except ImportError:
        print("  ERROR: TensorRT not installed")
        sys.exit(1)

    # Check Triton client
    try:
        import tritonclient.http
        print(f"  ✓ Triton client installed")
    except ImportError:
        print("  ERROR: tritonclient not installed")
        print("  Install with: pip install tritonclient[all]")
        sys.exit(1)

    # Check YOLO
    try:
        from ultralytics import YOLO
        print(f"  ✓ Ultralytics YOLO installed")
    except ImportError:
        print("  ERROR: Ultralytics not installed")
        print("  Install with: pip install ultralytics")
        sys.exit(1)

    # Check BoxMOT
    try:
        import boxmot
        print(f"  ✓ BoxMOT installed")
    except ImportError:
        print("  ERROR: BoxMOT not installed")
        print("  Install with: pip install boxmot")
        sys.exit(1)

    print("  ✓ All dependencies validated")


def main():
    """Main entry point"""
    args = parse_args()

    print("="*70)
    print("YOLO + TAO ReID + BoxMOT Tracking Pipeline")
    print("="*70)

    # Validate environment
    validate_environment()

    # Validate input video
    if not args.video.exists():
        print(f"\nERROR: Video file not found: {args.video}")
        sys.exit(1)

    print(f"\nInput video: {args.video}")

    # Load configurations
    print(f"\nLoading configurations from: {args.config_dir}")
    try:
        configs = load_all_configs(args.config_dir)
        print(f"  ✓ Loaded {len(configs)} config files")

        # Override visualization setting
        if args.no_visualization:
            configs['pipeline']['io']['save_visualization'] = False

    except Exception as e:
        print(f"ERROR: Failed to load configurations: {e}")
        sys.exit(1)

    # Create experiment directory
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        exp_name = ExperimentLogger.create_experiment_id("reid_pipeline")

    exp_dir = args.experiment_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment: {exp_name}")
    print(f"  Results directory: {exp_dir}")

    # Determine output path
    output_path = args.output
    if output_path is None and not args.no_visualization:
        output_path = exp_dir / f"{args.video.stem}_output.mp4"

    if output_path:
        print(f"  Visualization output: {output_path}")

    # Run pipeline
    try:
        print("\n" + "="*70)
        pipeline = ReIDPipeline(configs, exp_dir)

        result_dir = pipeline.process_video(
            video_path=args.video,
            output_path=output_path,
            max_frames=args.max_frames
        )

        print("\n" + "="*70)
        print("Pipeline execution completed successfully!")
        print("="*70)
        print(f"\nExperiment results: {result_dir}")
        print(f"\nTo view logs:")
        print(f"  cat {result_dir}/detections.jsonl | jq '.'")
        print(f"  cat {result_dir}/tracks.jsonl | jq '.'")
        print(f"  cat {result_dir}/metrics.jsonl | jq '.'")

        if output_path and output_path.exists():
            print(f"\nVisualization video: {output_path}")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nERROR: Pipeline execution failed")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
