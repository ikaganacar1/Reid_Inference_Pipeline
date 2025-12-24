"""
ReID Metrics Calculator

Computes standard person re-identification evaluation metrics:
- mAP (mean Average Precision)
- CMC (Cumulative Matching Characteristics) curve
- Rank-1, Rank-5, Rank-10 accuracy

Implements same-camera match exclusion for Market-1501 evaluation protocol.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ReIDMetricsCalculator:
    """
    Person Re-Identification Metrics Calculator

    Computes standard ReID evaluation metrics following Market-1501 protocol:
    - Same-camera matches are excluded
    - Junk images (person_id = -1) are excluded
    - Distractors (person_id = 0) do not contribute to evaluation
    """

    def __init__(self):
        """Initialize metrics calculator"""
        pass

    def compute_distance_matrix(self,
                                query_embeddings: np.ndarray,
                                gallery_embeddings: np.ndarray,
                                metric: str = 'cosine') -> np.ndarray:
        """
        Compute pairwise distance matrix between query and gallery embeddings.

        Args:
            query_embeddings: Query embeddings (Q, D) - L2 normalized
            gallery_embeddings: Gallery embeddings (G, D) - L2 normalized
            metric: Distance metric ('cosine' or 'euclidean')

        Returns:
            Distance matrix (Q, G) where lower values indicate higher similarity

        Note:
            - For cosine distance: distance = 1 - similarity
            - For Euclidean distance: distance = ||q - g||_2
            - Embeddings should be L2-normalized before calling this function
        """
        if metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            # For L2-normalized vectors: cosine_sim = dot(q, g)
            similarity = np.dot(query_embeddings, gallery_embeddings.T)  # (Q, G)
            distance = 1.0 - similarity

        elif metric == 'euclidean':
            # Euclidean distance: ||q - g||_2
            # Efficient computation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b
            query_norm_sq = np.sum(query_embeddings ** 2, axis=1, keepdims=True)  # (Q, 1)
            gallery_norm_sq = np.sum(gallery_embeddings ** 2, axis=1, keepdims=True)  # (G, 1)

            distance = query_norm_sq + gallery_norm_sq.T
            distance -= 2.0 * np.dot(query_embeddings, gallery_embeddings.T)
            distance = np.sqrt(np.maximum(distance, 0.0))  # Avoid negative due to numerical errors

        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'")

        return distance

    def compute_cmc(self,
                   distance_matrix: np.ndarray,
                   query_person_ids: np.ndarray,
                   query_camera_ids: np.ndarray,
                   gallery_person_ids: np.ndarray,
                   gallery_camera_ids: np.ndarray,
                   top_k: int = 10) -> np.ndarray:
        """
        Compute CMC (Cumulative Matching Characteristics) curve.

        CMC measures the probability that a correct match appears in the top-K rankings.

        Args:
            distance_matrix: (Q, G) distance matrix
            query_person_ids: (Q,) person IDs for queries
            query_camera_ids: (Q,) camera IDs for queries
            gallery_person_ids: (G,) person IDs for gallery
            gallery_camera_ids: (G,) camera IDs for gallery
            top_k: Compute CMC up to rank K

        Returns:
            CMC curve (K,) with accuracy at each rank [1, 2, ..., K]

        Protocol:
            - Exclude same-camera matches (same person, same camera)
            - For each query, find rank of first correct match
            - CMC[k] = fraction of queries where first match appears at rank ≤ k
        """
        num_queries = distance_matrix.shape[0]
        num_gallery = distance_matrix.shape[1]

        # Initialize CMC accumulator
        cmc = np.zeros(top_k, dtype=np.float32)

        num_valid_queries = 0

        for q_idx in range(num_queries):
            # Get distances from this query to all gallery
            dists = distance_matrix[q_idx]  # (G,)

            # Build relevance mask
            # Relevance = 1 if same person AND different camera, 0 otherwise
            relevance = (gallery_person_ids == query_person_ids[q_idx])

            # Exclude same-camera matches
            same_camera = (gallery_camera_ids == query_camera_ids[q_idx])
            relevance = relevance & ~same_camera

            # Skip query if no valid matches in gallery
            if np.sum(relevance) == 0:
                continue

            num_valid_queries += 1

            # Sort gallery by distance (ascending - lower distance = higher similarity)
            sorted_indices = np.argsort(dists)
            sorted_relevance = relevance[sorted_indices]

            # Find rank of first correct match (0-indexed)
            first_match_ranks = np.where(sorted_relevance)[0]

            if len(first_match_ranks) > 0:
                first_match_rank = first_match_ranks[0]

                # Update CMC: All ranks >= first_match_rank should be incremented
                if first_match_rank < top_k:
                    cmc[first_match_rank:] += 1.0

        # Normalize by number of valid queries
        if num_valid_queries > 0:
            cmc = cmc / num_valid_queries * 100.0  # Convert to percentage

        logger.debug(f"CMC computed with {num_valid_queries}/{num_queries} valid queries")

        return cmc

    def compute_map(self,
                   distance_matrix: np.ndarray,
                   query_person_ids: np.ndarray,
                   query_camera_ids: np.ndarray,
                   gallery_person_ids: np.ndarray,
                   gallery_camera_ids: np.ndarray) -> float:
        """
        Compute mean Average Precision (mAP).

        mAP is the mean of Average Precision across all queries.
        Average Precision (AP) for each query is the mean precision at each recall point.

        Args:
            distance_matrix: (Q, G) distance matrix
            query_person_ids: (Q,) person IDs for queries
            query_camera_ids: (Q,) camera IDs for queries
            gallery_person_ids: (G,) person IDs for gallery
            gallery_camera_ids: (G,) camera IDs for gallery

        Returns:
            mAP score (0-100%)

        Protocol:
            - For each query, rank gallery by distance
            - Exclude same-camera matches
            - Compute AP: mean precision at each recall point
            - mAP: mean of AP across all queries
        """
        num_queries = distance_matrix.shape[0]

        ap_scores = []

        for q_idx in range(num_queries):
            # Get distances from this query to all gallery
            dists = distance_matrix[q_idx]  # (G,)

            # Build relevance mask
            relevance = (gallery_person_ids == query_person_ids[q_idx])

            # Exclude same-camera matches
            same_camera = (gallery_camera_ids == query_camera_ids[q_idx])
            relevance = relevance & ~same_camera

            # Skip query if no valid matches in gallery
            num_relevant = np.sum(relevance)
            if num_relevant == 0:
                continue

            # Sort gallery by distance (ascending)
            sorted_indices = np.argsort(dists)
            sorted_relevance = relevance[sorted_indices]

            # Compute Average Precision
            # AP = mean of precision values at each recall point
            match_positions = np.where(sorted_relevance)[0]  # Ranks of all matches (0-indexed)

            # Precision at k = (number of relevant items in top k) / k
            # For matches at positions [r1, r2, ..., rN], precision = [1/r1+1, 2/r2+1, ...]
            precisions = np.arange(1, len(match_positions) + 1) / (match_positions + 1)

            # Average Precision is the mean of these precisions
            ap = np.mean(precisions)
            ap_scores.append(ap)

        # Compute mean AP
        if len(ap_scores) > 0:
            mAP = np.mean(ap_scores) * 100.0  # Convert to percentage
        else:
            mAP = 0.0

        logger.debug(f"mAP computed with {len(ap_scores)}/{num_queries} valid queries")

        return mAP

    def compute_all_metrics(self,
                          distance_matrix: np.ndarray,
                          query_person_ids: np.ndarray,
                          query_camera_ids: np.ndarray,
                          gallery_person_ids: np.ndarray,
                          gallery_camera_ids: np.ndarray,
                          top_k: int = 10) -> Dict:
        """
        Compute all standard ReID metrics.

        Args:
            distance_matrix: (Q, G) distance matrix
            query_person_ids: (Q,) person IDs for queries
            query_camera_ids: (Q,) camera IDs for queries
            gallery_person_ids: (G,) person IDs for gallery
            gallery_camera_ids: (G,) camera IDs for gallery
            top_k: Compute CMC up to rank K

        Returns:
            Dictionary with metrics:
                - mAP: mean Average Precision (0-100%)
                - rank1: Rank-1 accuracy (0-100%)
                - rank5: Rank-5 accuracy (0-100%)
                - rank10: Rank-10 accuracy (0-100%)
                - cmc_curve: Full CMC curve up to rank K
        """
        # Compute mAP
        mAP = self.compute_map(
            distance_matrix,
            query_person_ids,
            query_camera_ids,
            gallery_person_ids,
            gallery_camera_ids
        )

        # Compute CMC curve
        cmc = self.compute_cmc(
            distance_matrix,
            query_person_ids,
            query_camera_ids,
            gallery_person_ids,
            gallery_camera_ids,
            top_k=top_k
        )

        # Extract specific ranks
        rank1 = cmc[0] if len(cmc) > 0 else 0.0
        rank5 = cmc[4] if len(cmc) > 4 else 0.0
        rank10 = cmc[9] if len(cmc) > 9 else 0.0

        metrics = {
            'mAP': float(mAP),
            'rank1': float(rank1),
            'rank5': float(rank5),
            'rank10': float(rank10),
            'cmc_curve': cmc.tolist()
        }

        logger.info(f"Metrics: mAP={mAP:.3f}%, Rank-1={rank1:.1f}%, "
                   f"Rank-5={rank5:.1f}%, Rank-10={rank10:.1f}%")

        return metrics

    def evaluate(self,
                query_embeddings: np.ndarray,
                query_person_ids: np.ndarray,
                query_camera_ids: np.ndarray,
                gallery_embeddings: np.ndarray,
                gallery_person_ids: np.ndarray,
                gallery_camera_ids: np.ndarray,
                metric: str = 'cosine',
                top_k: int = 10) -> Dict:
        """
        End-to-end evaluation: compute distance matrix and all metrics.

        Args:
            query_embeddings: (Q, D) L2-normalized query embeddings
            query_person_ids: (Q,) person IDs for queries
            query_camera_ids: (Q,) camera IDs for queries
            gallery_embeddings: (G, D) L2-normalized gallery embeddings
            gallery_person_ids: (G,) person IDs for gallery
            gallery_camera_ids: (G,) camera IDs for gallery
            metric: Distance metric ('cosine' or 'euclidean')
            top_k: Compute CMC up to rank K

        Returns:
            Dictionary with all metrics
        """
        logger.info(f"Evaluating with {len(query_embeddings)} queries and "
                   f"{len(gallery_embeddings)} gallery images")

        # Compute distance matrix
        distance_matrix = self.compute_distance_matrix(
            query_embeddings,
            gallery_embeddings,
            metric=metric
        )

        # Compute all metrics
        metrics = self.compute_all_metrics(
            distance_matrix,
            query_person_ids,
            query_camera_ids,
            gallery_person_ids,
            gallery_camera_ids,
            top_k=top_k
        )

        return metrics
