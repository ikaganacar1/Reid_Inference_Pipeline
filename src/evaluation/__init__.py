"""
ReID Evaluation Module
Dataset evaluation with CMC and mAP metrics for Market1501-format datasets
"""

from .dataset import ReIDDataset
from .metrics import compute_cmc, compute_map, evaluate_reid
from .evaluator import ReIDEvaluator

__all__ = [
    'ReIDDataset',
    'compute_cmc',
    'compute_map',
    'evaluate_reid',
    'ReIDEvaluator'
]
