"""
Market-1501 Dataset Evaluation Module

This module provides comprehensive evaluation capabilities for person re-identification
datasets, with full pipeline simulation and standard metric computation.

Components:
- dataset_loader: Market-1501 dataset loading and ground truth generation
- metrics_calculator: Standard ReID metrics (mAP, CMC)
- gallery_tracker: Gallery decision tracking and statistics
- pipeline_simulator: Full pipeline simulation on static images
- evaluation_pipeline: Main evaluation orchestrator
"""

from .dataset_loader import Market1501Dataset

__all__ = [
    'Market1501Dataset',
]

# Import other modules when they are implemented
try:
    from .metrics_calculator import ReIDMetricsCalculator
    __all__.append('ReIDMetricsCalculator')
except ImportError:
    pass

try:
    from .gallery_tracker import GalleryDecisionTracker
    __all__.append('GalleryDecisionTracker')
except ImportError:
    pass
