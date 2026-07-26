import numpy as np

from src.evaluation.metrics import compute_cmc, compute_map


def test_queries_without_valid_gallery_matches_are_excluded():
    distances = np.array([[0.1], [0.2]], dtype=np.float32)
    query_pids = np.array([1, 2])
    query_camids = np.array([1, 1])
    gallery_pids = np.array([1])
    gallery_camids = np.array([2])

    assert compute_cmc(
        distances, query_pids, query_camids, gallery_pids, gallery_camids, ranks=[1]
    ) == {1: 1.0}
    assert compute_map(distances, query_pids, query_camids, gallery_pids, gallery_camids) == 1.0
