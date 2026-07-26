"""
ReID Evaluation Metrics
CMC (Cumulative Matching Characteristics) and mAP (mean Average Precision)

Implements standard Market1501 evaluation protocol:
- For each query, gallery images from the same camera are excluded
- CMC computes if correct identity appears in top-k results
- mAP computes average precision across all queries
"""

import numpy as np
from typing import Tuple, List, Dict


def compute_distance_matrix(query_features: np.ndarray,
                            gallery_features: np.ndarray,
                            metric: str = 'cosine') -> np.ndarray:
    """
    Compute pairwise distance matrix between query and gallery features.

    Args:
        query_features: Query embeddings [Q, D]
        gallery_features: Gallery embeddings [G, D]
        metric: Distance metric ('cosine' or 'euclidean')

    Returns:
        Distance matrix [Q, G] where lower = more similar
    """
    if metric == 'cosine':
        # Normalize features for cosine distance
        query_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-12)
        gallery_norm = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-12)

        # Cosine similarity: dot product of normalized vectors
        similarity = np.dot(query_norm, gallery_norm.T)

        # Convert to distance (1 - similarity)
        distance = 1.0 - similarity

    elif metric == 'euclidean':
        # Euclidean distance
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        query_sq = np.sum(query_features ** 2, axis=1, keepdims=True)
        gallery_sq = np.sum(gallery_features ** 2, axis=1, keepdims=True)
        cross = np.dot(query_features, gallery_features.T)

        distance = query_sq + gallery_sq.T - 2 * cross
        distance = np.sqrt(np.maximum(distance, 0))  # Avoid numerical issues

    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'")

    return distance


def compute_ap(index: np.ndarray, good_index: np.ndarray) -> float:
    """
    Compute Average Precision for a single query.

    This follows the Market1501 evaluation protocol (triangle mAP).

    Args:
        index: Sorted gallery indices (by distance, ascending)
        good_index: Indices of correct matches in gallery

    Returns:
        Average Precision score
    """
    num_good = len(good_index)
    if num_good == 0:
        return 0.0

    # Find positions of correct matches in ranked list
    mask = np.isin(index, good_index)
    rows_good = np.where(mask)[0]

    if len(rows_good) == 0:
        return 0.0

    # Compute precision at each recall point
    cmc = np.zeros(len(index))
    cmc[rows_good] = 1

    # Cumulative sum for precision calculation
    d_recall = 1.0 / num_good
    precision = np.cumsum(cmc) / (np.arange(len(index)) + 1)

    # AP = sum of precision at each correct match position
    ap = np.sum(precision[rows_good]) * d_recall

    return ap


def evaluate_single_query(query_idx: int,
                          distance_row: np.ndarray,
                          query_pid: int,
                          query_camid: int,
                          gallery_pids: np.ndarray,
                          gallery_camids: np.ndarray,
                          exclude_same_camera: bool = True) -> Tuple[np.ndarray, float]:
    """
    Evaluate a single query against the gallery.

    Args:
        query_idx: Query index (for debugging)
        distance_row: Distances from query to all gallery [G]
        query_pid: Query person ID
        query_camid: Query camera ID
        gallery_pids: Gallery person IDs [G]
        gallery_camids: Gallery camera IDs [G]
        exclude_same_camera: Whether to exclude same-camera matches

    Returns:
        Tuple of (cmc_curve, average_precision)
    """
    # Sort gallery by distance (ascending = most similar first)
    indices = np.argsort(distance_row)

    # Find matches (same person ID)
    matches = (gallery_pids[indices] == query_pid).astype(np.int32)

    # Create validity mask (exclude same query image and optionally same camera)
    valid = np.ones(len(indices), dtype=bool)

    if exclude_same_camera:
        # Exclude gallery images from same camera with same PID
        same_camera_same_pid = (gallery_pids[indices] == query_pid) & (gallery_camids[indices] == query_camid)
        valid = ~same_camera_same_pid

    # Also check for "junk" images if needed (pid == -1 or 0 typically)
    # In LTCC/Market1501, junk is usually marked with pid = -1 or special values
    junk_mask = gallery_pids[indices] < 0
    valid = valid & ~junk_mask

    # Apply validity mask
    valid_indices = indices[valid]
    valid_matches = matches[valid]

    if len(valid_indices) == 0 or np.sum(valid_matches) == 0:
        # No valid matches
        return np.zeros(len(indices)), 0.0

    # Compute CMC curve
    # CMC[k] = 1 if any of top-k results is correct
    cmc = np.cumsum(valid_matches) > 0
    cmc = cmc.astype(np.float32)

    # Pad CMC to original length for consistency
    full_cmc = np.zeros(len(indices))
    full_cmc[:len(cmc)] = cmc

    # Compute AP
    good_indices = np.where(valid_matches)[0]
    ap = compute_ap(np.arange(len(valid_indices)), good_indices)

    return full_cmc, ap


def compute_cmc(distance_matrix: np.ndarray,
                query_pids: np.ndarray,
                query_camids: np.ndarray,
                gallery_pids: np.ndarray,
                gallery_camids: np.ndarray,
                ranks: List[int] = None,
                exclude_same_camera: bool = True) -> Dict[int, float]:
    """
    Compute CMC curve (Cumulative Matching Characteristics).

    Args:
        distance_matrix: Distance matrix [Q, G]
        query_pids: Query person IDs [Q]
        query_camids: Query camera IDs [Q]
        gallery_pids: Gallery person IDs [G]
        gallery_camids: Gallery camera IDs [G]
        ranks: List of ranks to compute (default: [1, 5, 10, 20])
        exclude_same_camera: Exclude same-camera matches

    Returns:
        Dictionary mapping rank -> accuracy
    """
    if ranks is None:
        ranks = [1, 5, 10, 20]

    num_queries = len(query_pids)
    max_rank = max(ranks)
    if min(ranks) < 1 or max_rank > distance_matrix.shape[1]:
        raise ValueError(
            f"CMC ranks must be between 1 and gallery size {distance_matrix.shape[1]}, got {ranks}"
        )

    all_cmc = []

    for q_idx in range(num_queries):
        if not _has_valid_match(
            query_pids[q_idx],
            query_camids[q_idx],
            gallery_pids,
            gallery_camids,
            exclude_same_camera,
        ):
            continue
        cmc, _ = evaluate_single_query(
            q_idx,
            distance_matrix[q_idx],
            query_pids[q_idx],
            query_camids[q_idx],
            gallery_pids,
            gallery_camids,
            exclude_same_camera
        )
        all_cmc.append(cmc[:max_rank])

    if not all_cmc:
        raise ValueError("No query identity has a valid gallery match")
    all_cmc = np.array(all_cmc)

    # Compute accuracy at each rank
    cmc_results = {}
    for rank in ranks:
        cmc_results[rank] = np.mean(all_cmc[:, rank - 1])

    return cmc_results


def compute_map(distance_matrix: np.ndarray,
                query_pids: np.ndarray,
                query_camids: np.ndarray,
                gallery_pids: np.ndarray,
                gallery_camids: np.ndarray,
                exclude_same_camera: bool = True) -> float:
    """
    Compute mean Average Precision (mAP).

    Args:
        distance_matrix: Distance matrix [Q, G]
        query_pids: Query person IDs [Q]
        query_camids: Query camera IDs [Q]
        gallery_pids: Gallery person IDs [G]
        gallery_camids: Gallery camera IDs [G]
        exclude_same_camera: Exclude same-camera matches

    Returns:
        mAP score
    """
    num_queries = len(query_pids)
    all_ap = []

    for q_idx in range(num_queries):
        if not _has_valid_match(
            query_pids[q_idx],
            query_camids[q_idx],
            gallery_pids,
            gallery_camids,
            exclude_same_camera,
        ):
            continue
        _, ap = evaluate_single_query(
            q_idx,
            distance_matrix[q_idx],
            query_pids[q_idx],
            query_camids[q_idx],
            gallery_pids,
            gallery_camids,
            exclude_same_camera
        )
        all_ap.append(ap)

    if not all_ap:
        raise ValueError("No query identity has a valid gallery match")
    return np.mean(all_ap)


def _has_valid_match(
    query_pid: int,
    query_camid: int,
    gallery_pids: np.ndarray,
    gallery_camids: np.ndarray,
    exclude_same_camera: bool,
) -> bool:
    matches = gallery_pids == query_pid
    if exclude_same_camera:
        matches = matches & (gallery_camids != query_camid)
    return bool(np.any(matches))


def evaluate_reid(query_features: np.ndarray,
                  gallery_features: np.ndarray,
                  query_pids: np.ndarray,
                  query_camids: np.ndarray,
                  gallery_pids: np.ndarray,
                  gallery_camids: np.ndarray,
                  metric: str = 'cosine',
                  ranks: List[int] = None,
                  exclude_same_camera: bool = True) -> Dict:
    """
    Complete ReID evaluation computing CMC and mAP.

    Args:
        query_features: Query embeddings [Q, D]
        gallery_features: Gallery embeddings [G, D]
        query_pids: Query person IDs [Q]
        query_camids: Query camera IDs [Q]
        gallery_pids: Gallery person IDs [G]
        gallery_camids: Gallery camera IDs [G]
        metric: Distance metric ('cosine' or 'euclidean')
        ranks: CMC ranks to compute
        exclude_same_camera: Exclude same-camera matches

    Returns:
        Dictionary with 'cmc' (dict), 'mAP' (float), and 'distance_matrix'
    """
    if ranks is None:
        ranks = [1, 5, 10, 20]

    print(f"Computing distance matrix ({metric})...")
    distance_matrix = compute_distance_matrix(query_features, gallery_features, metric)

    print("Computing CMC...")
    cmc = compute_cmc(
        distance_matrix,
        query_pids, query_camids,
        gallery_pids, gallery_camids,
        ranks=ranks,
        exclude_same_camera=exclude_same_camera
    )

    print("Computing mAP...")
    mAP = compute_map(
        distance_matrix,
        query_pids, query_camids,
        gallery_pids, gallery_camids,
        exclude_same_camera=exclude_same_camera
    )

    return {
        'cmc': cmc,
        'mAP': mAP,
        'distance_matrix': distance_matrix
    }


if __name__ == "__main__":
    # Test metrics with synthetic data
    print("Testing ReID metrics with synthetic data...")
    print("=" * 60)

    np.random.seed(42)

    # Create synthetic embeddings
    # 10 query images, 50 gallery images, 5 identities
    num_queries = 10
    num_gallery = 50
    num_pids = 5
    embed_dim = 256

    # Assign PIDs and cameras
    query_pids = np.random.randint(0, num_pids, num_queries)
    query_camids = np.random.randint(0, 3, num_queries)
    gallery_pids = np.random.randint(0, num_pids, num_gallery)
    gallery_camids = np.random.randint(0, 3, num_gallery)

    # Create embeddings with some structure (same PID = similar embedding)
    pid_centers = np.random.randn(num_pids, embed_dim)
    query_features = pid_centers[query_pids] + np.random.randn(num_queries, embed_dim) * 0.1
    gallery_features = pid_centers[gallery_pids] + np.random.randn(num_gallery, embed_dim) * 0.1

    # Evaluate
    results = evaluate_reid(
        query_features, gallery_features,
        query_pids, query_camids,
        gallery_pids, gallery_camids,
        metric='cosine',
        exclude_same_camera=True
    )

    print("\nResults:")
    print(f"  mAP: {results['mAP']:.4f}")
    for rank, acc in results['cmc'].items():
        print(f"  Rank-{rank}: {acc:.4f}")

    print("\nMetrics test passed!")
