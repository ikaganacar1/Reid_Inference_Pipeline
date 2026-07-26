#!/usr/bin/env python3
"""Run the worker Jetson control API."""

import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.realtime.worker_control import RealtimeWorkerControl, load_control_config


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime worker control API")
    parser.add_argument("--config", type=Path, default=Path("configs/realtime_config.yaml"))
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_control_config(args.config)
    server = RealtimeWorkerControl(config, args.repo_dir)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
