#!/usr/bin/env python3
"""
Triton Model Repository Setup Script
Validates and organizes Triton model repository structure
"""

import argparse
import shutil
import sys
from pathlib import Path


def validate_model_repository(model_repo_path):
    """Validate Triton model repository structure"""
    model_repo = Path(model_repo_path)

    print(f"Validating Triton model repository: {model_repo}")

    if not model_repo.exists():
        print(f"ERROR: Model repository does not exist: {model_repo}")
        return False

    # Check for lttc_reid model
    lttc_reid_dir = model_repo / "lttc_reid"
    if not lttc_reid_dir.exists():
        print(f"ERROR: lttc_reid model directory not found: {lttc_reid_dir}")
        return False

    # Check for config.pbtxt
    config_file = lttc_reid_dir / "config.pbtxt"
    if not config_file.exists():
        print(f"ERROR: config.pbtxt not found: {config_file}")
        return False

    print(f"  ✓ Model directory: {lttc_reid_dir}")
    print(f"  ✓ Config file: {config_file}")

    # Check for version directories
    version_dirs = sorted([d for d in lttc_reid_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    if not version_dirs:
        print("ERROR: No version directories found (e.g., '1')")
        return False

    print(f"  ✓ Version directories: {[d.name for d in version_dirs]}")

    # Check for model.plan in version 1
    version_1_dir = lttc_reid_dir / "1"
    model_file = version_1_dir / "model.plan"

    if not model_file.exists():
        print(f"WARNING: model.plan not found in version 1: {model_file}")
        print("  Run export_to_tensorrt.py to generate the TensorRT engine")
        return False

    model_size_mb = model_file.stat().st_size / (1024 * 1024)
    print(f"  ✓ TensorRT engine: {model_file} ({model_size_mb:.2f} MB)")

    print("\nModel repository validation passed!")
    return True


def create_model_repository(model_repo_path, engine_path=None):
    """Create Triton model repository structure"""
    model_repo = Path(model_repo_path)
    lttc_reid_dir = model_repo / "lttc_reid"
    version_1_dir = lttc_reid_dir / "1"

    print(f"Creating Triton model repository: {model_repo}")

    # Create directories
    version_1_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created directory: {version_1_dir}")

    # Copy config.pbtxt if it doesn't exist
    config_file = lttc_reid_dir / "config.pbtxt"
    if not config_file.exists():
        # Check if config exists in project configs
        project_config = Path("triton_models/lttc_reid/config.pbtxt")
        if project_config.exists():
            shutil.copy(project_config, config_file)
            print(f"  ✓ Copied config.pbtxt")
        else:
            print(f"  WARNING: config.pbtxt not found, please create manually")

    # Copy engine if provided
    if engine_path:
        engine_path = Path(engine_path)
        if engine_path.exists():
            model_file = version_1_dir / "model.plan"
            shutil.copy(engine_path, model_file)
            print(f"  ✓ Copied TensorRT engine: {model_file}")
        else:
            print(f"  ERROR: Engine file not found: {engine_path}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Setup Triton model repository")
    parser.add_argument(
        "--model-repo",
        default="triton_models",
        help="Path to Triton model repository"
    )
    parser.add_argument(
        "--engine",
        help="Path to TensorRT engine file to copy"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing repository"
    )

    args = parser.parse_args()

    if args.validate_only:
        # Validate existing repository
        if not validate_model_repository(args.model_repo):
            sys.exit(1)
    else:
        # Create repository structure
        if not create_model_repository(args.model_repo, args.engine):
            sys.exit(1)

        # Validate after creation
        if not validate_model_repository(args.model_repo):
            sys.exit(1)

    print("\n" + "="*50)
    print("Triton model repository setup completed!")
    print("="*50)
    print(f"Model repository: {Path(args.model_repo).absolute()}")
    print("Ready to start Triton server with:")
    print("  bash scripts/start_triton_server.sh")


if __name__ == "__main__":
    main()
