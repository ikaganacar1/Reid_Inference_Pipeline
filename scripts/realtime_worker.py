#!/usr/bin/env python3
"""Run a Jetson camera worker for the realtime ReID pipeline."""

import argparse
import asyncio
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.realtime.worker import RealtimeWorker, load_worker_config, load_yolo_config
from src.runtime_config import ROOT, load_runtime_environment, runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime Jetson YOLO worker")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--yolo-config", type=Path, help="Explicit YOLO config path")
    parser.add_argument("--camera-id", help="Override worker camera ID")
    parser.add_argument("--source", help="Override camera source, for example 0 or rtsp://...")
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    load_runtime_environment(args.env_file)
    paths = runtime_paths(config_dir=args.config_dir)
    config = load_worker_config(
        args.config or paths.realtime,
        camera_id=args.camera_id,
        source=args.source,
    )
    yolo_config_path = args.yolo_config or paths.yolo
    yolo_config = load_yolo_config(yolo_config_path)
    worker = RealtimeWorker(config, yolo_config)
    asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main()
