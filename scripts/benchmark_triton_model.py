#!/usr/bin/env python3
"""
Benchmark Triton ReID Model
Test inference speed and latency for different batch sizes
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reid_client import TritonReIDClient


def benchmark_model(client: TritonReIDClient, batch_sizes=[1, 4, 8, 16], num_iterations=100):
    """Benchmark model at different batch sizes."""

    print("=" * 70)
    print("Triton ReID Model Benchmark")
    print("=" * 70)

    results = {}

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        print("-" * 40)

        # Create dummy crops
        dummy_crops = [
            np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]

        # Warmup
        for _ in range(10):
            client.infer(dummy_crops)

        # Benchmark
        times = []
        for i in range(num_iterations):
            start = time.time()
            embeddings = client.infer(dummy_crops)
            elapsed = time.time() - start
            times.append(elapsed * 1000)  # ms

        times = np.array(times)

        # Compute statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)

        throughput = (batch_size * 1000) / mean_time  # images/sec

        results[batch_size] = {
            'mean_ms': mean_time,
            'std_ms': std_time,
            'min_ms': min_time,
            'max_ms': max_time,
            'p50_ms': p50,
            'p95_ms': p95,
            'p99_ms': p99,
            'throughput_img_s': throughput,
            'latency_per_image_ms': mean_time / batch_size
        }

        print(f"  Mean latency:    {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"  Min/Max:         {min_time:.2f} / {max_time:.2f} ms")
        print(f"  Percentiles:     p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f} ms")
        print(f"  Throughput:      {throughput:.1f} images/sec")
        print(f"  Per-image cost:  {mean_time/batch_size:.2f} ms")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Batch':>8} {'Mean (ms)':>12} {'P95 (ms)':>12} {'Throughput':>15} {'$/img (ms)':>12}")
    print("-" * 70)

    for bs, stats in results.items():
        print(f"{bs:>8} {stats['mean_ms']:>12.2f} {stats['p95_ms']:>12.2f} "
              f"{stats['throughput_img_s']:>12.1f} img/s {stats['latency_per_image_ms']:>12.2f}")

    print("=" * 70)

    # Best configurations
    best_throughput_bs = max(results.keys(), key=lambda k: results[k]['throughput_img_s'])
    best_latency_bs = min(results.keys(), key=lambda k: results[k]['latency_per_image_ms'])

    print(f"\nRecommendations:")
    print(f"  Best throughput: batch_size={best_throughput_bs} "
          f"({results[best_throughput_bs]['throughput_img_s']:.1f} img/s)")
    print(f"  Best latency:    batch_size={best_latency_bs} "
          f"({results[best_latency_bs]['latency_per_image_ms']:.2f} ms/img)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton ReID model")
    parser.add_argument("--config", type=Path, default=Path("configs/reid_config.yaml"))
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per batch size")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    args = parser.parse_args()

    # Load config
    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Model: {config['triton']['model_name']}")
    print(f"Server: {config['triton']['server_url']}")
    print(f"Iterations per batch: {args.iterations}")

    # Create client
    try:
        client = TritonReIDClient(config)
    except Exception as e:
        print(f"\nERROR: Failed to connect to Triton server: {e}")
        print("\nMake sure Triton server is running:")
        print("  bash scripts/start_triton_server.sh")
        sys.exit(1)

    # Run benchmark
    results = benchmark_model(client, args.batch_sizes, args.iterations)


if __name__ == "__main__":
    main()
