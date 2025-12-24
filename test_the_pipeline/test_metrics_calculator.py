"""
Unit Test for ReID Metrics Calculator

Tests distance matrix computation, CMC curve, and mAP calculation
with synthetic embeddings where ground truth is known.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reid_pipeline.evaluation.metrics_calculator import ReIDMetricsCalculator


def test_distance_matrix():
    """Test distance matrix computation"""
    print("\n=== Testing Distance Matrix Computation ===")

    calculator = ReIDMetricsCalculator()

    # Create simple embeddings (L2-normalized)
    query_embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    gallery_embeddings = np.array([
        [1.0, 0.0, 0.0],  # Same as query 0
        [0.0, 1.0, 0.0],  # Same as query 1
        [0.707, 0.707, 0.0],  # 45-degree angle from query 0
        [0.0, 0.0, 1.0]   # Same as query 2
    ], dtype=np.float32)

    # Test cosine distance
    dist = calculator.compute_distance_matrix(query_embeddings, gallery_embeddings, metric='cosine')

    # Expected: distance = 1 - similarity
    # Query 0 vs Gallery 0: similarity=1.0, distance=0.0
    # Query 0 vs Gallery 2: similarity=0.707, distance=0.293
    assert dist.shape == (3, 4)
    assert np.isclose(dist[0, 0], 0.0, atol=1e-5), f"Expected 0.0, got {dist[0, 0]}"
    assert np.isclose(dist[1, 1], 0.0, atol=1e-5), f"Expected 0.0, got {dist[1, 1]}"
    assert np.isclose(dist[2, 3], 0.0, atol=1e-5), f"Expected 0.0, got {dist[2, 3]}"
    assert np.isclose(dist[0, 2], 1.0 - 0.707, atol=1e-2), f"Expected ~0.293, got {dist[0, 2]}"

    print("✓ Cosine distance matrix computed correctly")

    # Test Euclidean distance
    dist_euclidean = calculator.compute_distance_matrix(
        query_embeddings, gallery_embeddings, metric='euclidean'
    )

    # For L2-normalized vectors:
    # Query 0 [1, 0, 0] vs Gallery 0 [1, 0, 0]: distance=0.0
    # Query 0 [1, 0, 0] vs Gallery 1 [0, 1, 0]: distance=sqrt(2)
    assert np.isclose(dist_euclidean[0, 0], 0.0, atol=1e-5)
    assert np.isclose(dist_euclidean[0, 1], np.sqrt(2.0), atol=1e-2)

    print("✓ Euclidean distance matrix computed correctly")


def test_cmc_perfect_ranking():
    """Test CMC with perfect ranking (all queries matched at rank 1)"""
    print("\n=== Testing CMC with Perfect Ranking ===")

    calculator = ReIDMetricsCalculator()

    # Create synthetic scenario:
    # 3 queries, 5 gallery images
    # Each query has exact match at rank 1

    distance_matrix = np.array([
        [0.0, 0.8, 0.9, 0.7, 0.6],  # Query 0: best match at position 0
        [0.9, 0.0, 0.8, 0.7, 0.6],  # Query 1: best match at position 1
        [0.8, 0.7, 0.0, 0.9, 0.6]   # Query 2: best match at position 2
    ], dtype=np.float32)

    query_person_ids = np.array([1, 2, 3])
    query_camera_ids = np.array([1, 1, 1])

    gallery_person_ids = np.array([1, 2, 3, 4, 5])  # Matches for persons 1, 2, 3
    gallery_camera_ids = np.array([2, 2, 2, 2, 2])  # All different cameras

    cmc = calculator.compute_cmc(
        distance_matrix,
        query_person_ids,
        query_camera_ids,
        gallery_person_ids,
        gallery_camera_ids,
        top_k=5
    )

    # All queries should have match at rank 1
    # CMC[0] (rank-1) should be 100%
    assert np.isclose(cmc[0], 100.0, atol=1e-3), f"Expected 100.0, got {cmc[0]}"
    assert np.isclose(cmc[1], 100.0, atol=1e-3), f"Expected 100.0, got {cmc[1]}"  # Rank-2

    print(f"✓ CMC (perfect ranking): Rank-1={cmc[0]:.1f}%, Rank-5={cmc[4]:.1f}%")


def test_cmc_with_failures():
    """Test CMC with some queries failing to find match"""
    print("\n=== Testing CMC with Partial Matches ===")

    calculator = ReIDMetricsCalculator()

    # 3 queries:
    # - Query 0: Match at rank 1
    # - Query 1: Match at rank 3
    # - Query 2: Match at rank 5

    # For Query 0 (person 1):
    # Gallery [1, 4, 5, 6, 7] with distances [0.1, 0.8, 0.9, 0.7, 0.6]
    # Sorted: person 1 (0.1) at rank 1 ✓

    # For Query 1 (person 2):
    # Gallery [3, 4, 2, 6, 7] with distances [0.1, 0.2, 0.5, 0.3, 0.4]
    # Sorted: [3 (0.1), 4 (0.2), 6 (0.3), 7 (0.4), 2 (0.5)]
    # Person 2 is at rank 5... but we want rank 3
    # Let me fix: distances [0.5, 0.1, 0.2, 0.7, 0.6]
    # Sorted: [4 (0.1), 2 (0.2), 3 (0.5), 7 (0.6), 6 (0.7)]
    # Person IDs: [4, 2, 3, 7, 6]
    # Person 2 is at rank 2... Let me try again
    # distances [0.5, 0.1, 0.6, 0.2, 0.7]
    # Sorted: [4 (0.1), 6 (0.2), 3 (0.5), 2 (0.6), 7 (0.7)]
    # Person IDs: [4, 6, 3, 2, 7]
    # Person 2 is at rank 4... Close
    # distances [0.5, 0.1, 0.6, 0.7, 0.2]
    # Sorted: [4 (0.1), 7 (0.2), 3 (0.5), 2 (0.6), 6 (0.7)]
    # Person IDs: [4, 7, 3, 2, 6]
    # Person 2 is at rank 4... hmm
    # Try: [0.5, 0.1, 0.2, 0.7, 0.6]
    # Sorted: [4 (0.1), 2 (0.2), 3 (0.5), 7 (0.6), 6 (0.7)]
    # Person 2 at rank 2...
    # I need distractors at ranks 1, 2, then match at rank 3
    # distances [0.1, 0.2, 0.5, 0.7, 0.6]
    # Sorted by distance: [3 (0.1), 4 (0.2), 2 (0.5), 7 (0.6), 6 (0.7)]
    # With gallery_person_ids = [3, 4, 2, 6, 7], person 2 is at rank 3 ✓

    distance_matrix = np.array([
        [0.1, 0.8, 0.9, 0.7, 0.6],  # Query 0 (person 1): sorted [1, 7, 6, 4, 5], match at rank 1
        [0.1, 0.2, 0.5, 0.7, 0.6],  # Query 1 (person 2): sorted [3, 4, 2, 6, 7], match at rank 3
        [0.1, 0.2, 0.3, 0.4, 0.5]   # Query 2 (person 3): sorted [6, 4, 7, 5, 3], match at rank 5
    ], dtype=np.float32)

    query_person_ids = np.array([1, 2, 3])
    query_camera_ids = np.array([1, 1, 1])

    gallery_person_ids = np.array([1, 4, 2, 6, 3])  # Matches at indices [0, 2, 4]
    gallery_camera_ids = np.array([2, 2, 2, 2, 2])

    cmc = calculator.compute_cmc(
        distance_matrix,
        query_person_ids,
        query_camera_ids,
        gallery_person_ids,
        gallery_camera_ids,
        top_k=5
    )

    # Expected CMC:
    # Rank-1: 1/3 = 33.3%
    # Rank-2: 1/3 = 33.3%
    # Rank-3: 2/3 = 66.7%
    # Rank-4: 2/3 = 66.7%
    # Rank-5: 3/3 = 100.0%

    assert np.isclose(cmc[0], 33.3, atol=1.0), f"Rank-1: Expected 33.3%, got {cmc[0]:.1f}%"
    assert np.isclose(cmc[2], 66.7, atol=1.0), f"Rank-3: Expected 66.7%, got {cmc[2]:.1f}%"
    assert np.isclose(cmc[4], 100.0, atol=1.0), f"Rank-5: Expected 100.0%, got {cmc[4]:.1f}%"

    print(f"✓ CMC (partial matches): Rank-1={cmc[0]:.1f}%, Rank-3={cmc[2]:.1f}%, Rank-5={cmc[4]:.1f}%")


def test_same_camera_exclusion():
    """Test that same-camera matches are properly excluded"""
    print("\n=== Testing Same-Camera Exclusion ===")

    calculator = ReIDMetricsCalculator()

    # Query 0 has perfect match at rank 1, but it's from same camera (should be excluded)
    # True match is at rank 2

    distance_matrix = np.array([
        [0.0, 0.5, 0.9],  # Best match at position 0, but same camera
    ], dtype=np.float32)

    query_person_ids = np.array([1])
    query_camera_ids = np.array([1])

    gallery_person_ids = np.array([1, 1, 2])  # Gallery[0] and Gallery[1] are same person
    gallery_camera_ids = np.array([1, 2, 2])  # Gallery[0] is same camera (excluded)

    cmc = calculator.compute_cmc(
        distance_matrix,
        query_person_ids,
        query_camera_ids,
        gallery_person_ids,
        gallery_camera_ids,
        top_k=3
    )

    # Gallery[0] should be excluded (same camera)
    # True first match is Gallery[1] at rank 2
    # So CMC[0] (rank-1) should be 0%, CMC[1] (rank-2) should be 100%

    assert np.isclose(cmc[0], 0.0, atol=1.0), f"Rank-1 should be 0% (same-camera excluded), got {cmc[0]:.1f}%"
    assert np.isclose(cmc[1], 100.0, atol=1.0), f"Rank-2 should be 100%, got {cmc[1]:.1f}%"

    print("✓ Same-camera exclusion works correctly")


def test_map_computation():
    """Test mAP computation"""
    print("\n=== Testing mAP Computation ===")

    calculator = ReIDMetricsCalculator()

    # Create scenario with 2 queries:
    # - Query 0: Has 2 matches in gallery at ranks 1 and 3
    # - Query 1: Has 1 match in gallery at rank 2

    distance_matrix = np.array([
        [0.1, 0.5, 0.3, 0.9],  # Query 0: matches at ranks 1 (0.1) and 3 (0.3)
        [0.9, 0.2, 0.8, 0.7]   # Query 1: match at rank 2 (0.2)
    ], dtype=np.float32)

    query_person_ids = np.array([1, 2])
    query_camera_ids = np.array([1, 1])

    gallery_person_ids = np.array([1, 2, 1, 3])  # Query 0 matches [0, 2], Query 1 matches [1]
    gallery_camera_ids = np.array([2, 2, 2, 2])

    mAP = calculator.compute_map(
        distance_matrix,
        query_person_ids,
        query_camera_ids,
        gallery_person_ids,
        gallery_camera_ids
    )

    # Expected AP for Query 0:
    # Sorted gallery by distance: [0 (person 1), 2 (person 1), 1 (person 2), 3 (person 3)]
    # Matches at ranks [1, 2] (1-indexed)
    # Precisions: [1/1, 2/2] = [1.0, 1.0]
    # AP = mean([1.0, 1.0]) = 1.0

    # Expected AP for Query 1:
    # Sorted gallery: [1 (person 2), 3 (person 3), 2 (person 1), 0 (person 1)]
    # Match at rank [1]
    # Precisions: [1/1] = [1.0]
    # AP = 1.0

    # mAP = mean([1.0, 1.0]) = 1.0 = 100%

    assert np.isclose(mAP, 100.0, atol=1.0), f"Expected mAP=100%, got {mAP:.2f}%"

    print(f"✓ mAP computation: mAP={mAP:.2f}%")


def test_end_to_end_evaluation():
    """Test full evaluation pipeline"""
    print("\n=== Testing End-to-End Evaluation ===")

    calculator = ReIDMetricsCalculator()

    # Create synthetic embeddings (L2-normalized)
    np.random.seed(42)

    # 5 queries with distinct embeddings
    query_embeddings = np.random.randn(5, 128).astype(np.float32)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    # 10 gallery images
    # - Gallery[0, 2, 4, 6, 8] match queries [0, 1, 2, 3, 4] respectively (same embeddings)
    # - Gallery[1, 3, 5, 7, 9] are random distractors
    gallery_embeddings = np.random.randn(10, 128).astype(np.float32)

    # Copy query embeddings to gallery at even indices (with small noise)
    for i in range(5):
        gallery_embeddings[i*2] = query_embeddings[i] + np.random.randn(128) * 0.01

    gallery_embeddings = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)

    # Metadata
    query_person_ids = np.array([1, 2, 3, 4, 5])
    query_camera_ids = np.array([1, 1, 1, 1, 1])

    gallery_person_ids = np.array([1, 6, 2, 7, 3, 8, 4, 9, 5, 10])  # Matches at even indices
    gallery_camera_ids = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    # Run evaluation
    metrics = calculator.evaluate(
        query_embeddings,
        query_person_ids,
        query_camera_ids,
        gallery_embeddings,
        gallery_person_ids,
        gallery_camera_ids,
        metric='cosine',
        top_k=10
    )

    print(f"\nEvaluation Results:")
    print(f"  mAP: {metrics['mAP']:.2f}%")
    print(f"  Rank-1: {metrics['rank1']:.1f}%")
    print(f"  Rank-5: {metrics['rank5']:.1f}%")
    print(f"  Rank-10: {metrics['rank10']:.1f}%")

    # With exact matches at even indices, we expect very high Rank-1
    # (Small noise might push some to rank 2, but should be >80%)
    assert metrics['rank1'] > 80.0, f"Expected Rank-1 > 80%, got {metrics['rank1']:.1f}%"
    assert metrics['mAP'] > 80.0, f"Expected mAP > 80%, got {metrics['mAP']:.2f}%"

    print("✓ End-to-end evaluation successful")


def main():
    """Run all tests"""
    print("=" * 60)
    print("ReID Metrics Calculator Unit Tests")
    print("=" * 60)

    try:
        test_distance_matrix()
        test_cmc_perfect_ranking()
        test_cmc_with_failures()
        test_same_camera_exclusion()
        test_map_computation()
        test_end_to_end_evaluation()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
