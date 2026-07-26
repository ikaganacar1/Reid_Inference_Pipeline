#!/usr/bin/env python3
"""Analyze cosine-distance operating points from a saved ReID evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings_dir", type=Path)
    parser.add_argument("--thresholds", default="0.16,0.20,0.25,0.30,0.35,0.40")
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def normalize(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(~np.isfinite(rows)) or np.any(norms <= 0):
        raise ValueError("Embeddings must be finite and non-zero")
    return rows / norms


def percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        str(percentile): float(np.percentile(values, percentile))
        for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99)
    }


def analyze(directory: Path, thresholds: list[float], margin: float) -> dict:
    query = normalize(np.load(directory / "query_embeddings.npy"))
    gallery = normalize(np.load(directory / "gallery_embeddings.npy"))
    query_pids = np.load(directory / "query_pids.npy")
    query_camids = np.load(directory / "query_camids.npy")
    gallery_pids = np.load(directory / "gallery_pids.npy")
    gallery_camids = np.load(directory / "gallery_camids.npy")

    distances = 1.0 - query @ gallery.T
    same_identity = query_pids[:, None] == gallery_pids[None, :]
    same_camera = query_camids[:, None] == gallery_camids[None, :]
    positives = same_identity & ~same_camera
    negatives = ~same_identity

    valid_queries = positives.any(axis=1)
    if not np.all(valid_queries):
        distances = distances[valid_queries]
        positives = positives[valid_queries]
        negatives = negatives[valid_queries]
        query_pids = query_pids[valid_queries]

    positive_distances = distances[positives]
    negative_distances = distances[negatives]
    nearest_positive = np.min(np.where(positives, distances, np.inf), axis=1)
    nearest_negative = np.min(np.where(negatives, distances, np.inf), axis=1)

    pairwise_thresholds = {}
    for threshold in thresholds:
        pairwise_thresholds[str(threshold)] = {
            "positive_pair_recall": float(np.mean(positive_distances <= threshold)),
            "negative_pair_false_accept_rate": float(np.mean(negative_distances <= threshold)),
            "queries_with_positive_below_threshold": float(np.mean(nearest_positive <= threshold)),
            "queries_with_impostor_below_threshold": float(np.mean(nearest_negative <= threshold)),
        }

    unique_pids = np.unique(gallery_pids)
    centroids = normalize(
        np.stack([gallery[gallery_pids == pid].mean(axis=0) for pid in unique_pids])
    )
    centroid_distances = 1.0 - query @ centroids.T
    own_indices = np.searchsorted(unique_pids, query_pids)
    if np.any(unique_pids[own_indices] != query_pids):
        raise ValueError("A query identity is absent from the gallery centroids")

    own_distances = centroid_distances[np.arange(len(query)), own_indices]
    ranked = np.argsort(centroid_distances, axis=1)
    best_indices = ranked[:, 0]
    best_distances = centroid_distances[np.arange(len(query)), best_indices]
    second_distances = centroid_distances[np.arange(len(query)), ranked[:, 1]]
    best_is_correct = best_indices == own_indices

    centroid_thresholds = {}
    for threshold in thresholds:
        under_threshold = best_distances <= threshold
        separated = second_distances - best_distances >= margin
        accepted = under_threshold & separated
        centroid_thresholds[str(threshold)] = {
            "true_accept_rate": float(np.mean(accepted & best_is_correct)),
            "false_accept_rate": float(np.mean(accepted & ~best_is_correct)),
            "abstain_rate": float(np.mean(~accepted)),
            "accepted_precision": float(
                np.mean(best_is_correct[accepted]) if np.any(accepted) else 1.0
            ),
        }

    return {
        "dataset": {
            "queries": int(len(query)),
            "gallery": int(len(gallery)),
            "identities": int(len(unique_pids)),
        },
        "pairwise": {
            "positive_pairs": int(len(positive_distances)),
            "negative_pairs": int(len(negative_distances)),
            "positive_distance_percentiles": percentiles(positive_distances),
            "negative_distance_percentiles": percentiles(negative_distances),
            "nearest_positive_percentiles": percentiles(nearest_positive),
            "nearest_impostor_percentiles": percentiles(nearest_negative),
            "rank1_from_nearest_images": float(np.mean(nearest_positive < nearest_negative)),
            "thresholds": pairwise_thresholds,
        },
        "identity_centroids": {
            "rank1": float(np.mean(best_is_correct)),
            "own_distance_percentiles": percentiles(own_distances),
            "best_distance_percentiles": percentiles(best_distances),
            "margin": margin,
            "thresholds": centroid_thresholds,
        },
    }


def main() -> None:
    args = parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    results = analyze(args.embeddings_dir, thresholds, args.margin)
    serialized = json.dumps(results, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
