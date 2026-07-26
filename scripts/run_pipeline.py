#!/usr/bin/env python3
"""
Run ReID Pipeline on Video
Entry point script for the complete tracking pipeline
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.logger import ExperimentLogger
from src.pipeline import ReIDPipeline


def load_all_configs():
    """Load all configuration files"""
    configs = {}
    config_dir = Path("configs")

    for config_file in ["yolo_config.yaml", "reid_config.yaml", "tracker_config.yaml", "pipeline_config.yaml"]:
        config_path = config_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_name = config_file.split('_')[0]  # yolo, reid, tracker, pipeline
            configs[config_name] = yaml.safe_load(f)

    return configs


def main():
    parser = argparse.ArgumentParser(description="Run ReID tracking pipeline on video")
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output",
        help="Path to save output video (optional)"
    )
    parser.add_argument(
        "--experiment-dir",
        help="Directory to save experiment logs (optional)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to process (for testing)"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display video while processing"
    )

    args = parser.parse_args()

    # Validate input video
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    # Load configs
    print("Loading configurations...")
    configs = load_all_configs()

    # Create experiment directory
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
    else:
        exp_id = ExperimentLogger.create_experiment_id(video_path.stem)
        exp_dir = Path("experiments") / exp_id

    print(f"Experiment directory: {exp_dir}")

    # Create pipeline
    print("\nInitializing pipeline components...")
    pipeline = ReIDPipeline(configs, exp_dir)

    # Process video
    print(f"\nProcessing video: {video_path}")
    print(f"Output video: {args.output if args.output else 'None (logging only)'}")

    pipeline.process_video(
        video_path=str(video_path),
        output_path=args.output,
        max_frames=args.max_frames
    )

    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    print(f"Experiment logs: {exp_dir}")
    if args.output:
        print(f"Output video: {args.output}")


if __name__ == "__main__":
    main()
