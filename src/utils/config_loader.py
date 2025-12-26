"""
Configuration Loader
Utility to load and merge YAML configuration files
"""

from pathlib import Path
from typing import Dict, Any

import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load a single YAML configuration file

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_all_configs(config_dir: Path = None) -> Dict[str, Any]:
    """
    Load all configuration files and merge them

    Args:
        config_dir: Directory containing config files (default: configs/)

    Returns:
        Merged configuration dictionary
    """
    if config_dir is None:
        config_dir = Path("configs")

    configs = {}

    # Load individual configs
    config_files = {
        "yolo": config_dir / "yolo_config.yaml",
        "reid": config_dir / "reid_config.yaml",
        "tracker": config_dir / "tracker_config.yaml",
        "pipeline": config_dir / "pipeline_config.yaml"
    }

    for name, path in config_files.items():
        if path.exists():
            configs[name] = load_config(path)
        else:
            print(f"WARNING: Config file not found: {path}")

    return configs


def save_config(config: Dict[str, Any], output_path: Path):
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


if __name__ == "__main__":
    # Test config loader
    configs = load_all_configs()

    print("Loaded configurations:")
    for name, config in configs.items():
        print(f"  {name}: {len(config)} keys")

    print("\nConfig loader test passed!")
