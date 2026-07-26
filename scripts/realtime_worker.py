#!/usr/bin/env python3
"""Run a Jetson camera worker for the realtime ReID pipeline."""

import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.realtime.worker import RealtimeWorker, load_worker_config, load_yolo_config


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime Jetson YOLO worker")
    parser.add_argument("--config", type=Path, default=Path("configs/realtime_config.yaml"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--yolo-config", type=Path, help="Explicit YOLO config path")
    parser.add_argument("--camera-id", help="Override worker camera ID")
    parser.add_argument("--source", help="Override camera source, for example 0 or rtsp://...")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_worker_config(args.config, camera_id=args.camera_id, source=args.source)
    yolo_config_path = args.yolo_config or (args.config_dir / "yolo_config.yaml")
    yolo_config = load_yolo_config(yolo_config_path)
    worker = RealtimeWorker(config, yolo_config)
    asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main()
