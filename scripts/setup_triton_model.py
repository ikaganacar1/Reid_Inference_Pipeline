#!/usr/bin/env python3
"""
Triton Model Repository Setup Script
Validates and organizes Triton model repository structure
"""

import argparse
import shutil
import sys
from pathlib import Path


def validate_model_repository(model_repo_path, model_name):
    """Validate Triton model repository structure"""
    model_repo = Path(model_repo_path)

    print(f"Validating Triton model repository: {model_repo}")

    if not model_repo.exists():
        print(f"ERROR: Model repository does not exist: {model_repo}")
        return False

    model_dir = model_repo / model_name
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        return False

    # Check for config.pbtxt
    config_file = model_dir / "config.pbtxt"
    if not config_file.exists():
        print(f"ERROR: config.pbtxt not found: {config_file}")
        return False

    print(f"  ✓ Model directory: {model_dir}")
    print(f"  ✓ Config file: {config_file}")

    # Check for version directories
    version_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    if not version_dirs:
        print("ERROR: No version directories found (e.g., '1')")
        return False

    print(f"  ✓ Version directories: {[d.name for d in version_dirs]}")

    # Triton accepts different artifact names depending on the configured backend.
    version_1_dir = model_dir / "1"
    model_files = [version_1_dir / "model.onnx", version_1_dir / "model.plan"]
    model_file = next((path for path in model_files if path.exists()), None)

    if model_file is None:
        print(f"ERROR: No model.onnx or model.plan found in version 1: {version_1_dir}")
        return False

    model_size_mb = model_file.stat().st_size / (1024 * 1024)
    print(f"  ✓ Model artifact: {model_file} ({model_size_mb:.2f} MB)")

    print("\nModel repository validation passed!")
    return True


def create_model_repository(model_repo_path, model_name, model_path=None):
    """Create Triton model repository structure"""
    model_repo = Path(model_repo_path)
    model_dir = model_repo / model_name
    version_1_dir = model_dir / "1"

    print(f"Creating Triton model repository: {model_repo}")

    # Create directories
    version_1_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created directory: {version_1_dir}")

    # Copy config.pbtxt if it doesn't exist
    config_file = model_dir / "config.pbtxt"
    if not config_file.exists():
        # Check if config exists in project configs
        project_config = Path("triton_models") / model_name / "config.pbtxt"
        if project_config.exists():
            shutil.copy(project_config, config_file)
            print("  ✓ Copied config.pbtxt")
        else:
            print("  WARNING: config.pbtxt not found, please create manually")

    # Copy model artifact if provided
    if model_path:
        model_path = Path(model_path)
        if model_path.exists():
            artifact_name = "model.plan" if model_path.suffix == ".plan" else "model.onnx"
            model_file = version_1_dir / artifact_name
            shutil.copy(model_path, model_file)
            print(f"  ✓ Copied model artifact: {model_file}")
        else:
            print(f"  ERROR: Model file not found: {model_path}")
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
        "--model-name",
        default="swin_base_reid",
        help="Triton model name"
    )
    parser.add_argument(
        "--model",
        help="Path to ONNX model or TensorRT plan file to copy"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing repository"
    )

    args = parser.parse_args()

    if args.validate_only:
        # Validate existing repository
        if not validate_model_repository(args.model_repo, args.model_name):
            sys.exit(1)
    else:
        # Create repository structure
        if not create_model_repository(args.model_repo, args.model_name, args.model):
            sys.exit(1)

        # Validate after creation
        if not validate_model_repository(args.model_repo, args.model_name):
            sys.exit(1)

    print("\n" + "="*50)
    print("Triton model repository setup completed!")
    print("="*50)
    print(f"Model repository: {Path(args.model_repo).absolute()}")
    print("Ready to start Triton server with:")
    print("  bash scripts/start_triton_server.sh")


if __name__ == "__main__":
    main()
