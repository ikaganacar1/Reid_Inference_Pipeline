#!/usr/bin/env python3
"""Run the prime Jetson realtime ReID server."""

import argparse
import asyncio
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.realtime.prime_server import (
    RealtimePrimeServer,
    load_prime_pipeline_configs,
    load_realtime_config,
)
from src.runtime_config import ROOT, load_runtime_environment, runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime prime ReID server")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--config-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    load_runtime_environment(args.env_file)
    paths = runtime_paths("prime", args.config_dir)
    realtime_config = load_realtime_config(args.config or paths.realtime)
    pipeline_configs = load_prime_pipeline_configs(args.config_dir)
    server = RealtimePrimeServer(realtime_config, pipeline_configs)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
