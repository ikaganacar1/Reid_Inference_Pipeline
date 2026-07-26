#!/usr/bin/env python3
"""Run the worker Jetson control API."""

import argparse
import asyncio
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.realtime.worker_control import RealtimeWorkerControl, load_control_config
from src.runtime_config import ROOT, load_runtime_environment, runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime worker control API")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--repo-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    load_runtime_environment(args.env_file)
    paths = runtime_paths()
    config = load_control_config(args.config or paths.realtime)
    server = RealtimeWorkerControl(config, args.repo_dir or ROOT)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
