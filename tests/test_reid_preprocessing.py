import numpy as np
import pytest

from src.reid_preprocessing import preprocess_reid_crops


def test_preprocessing_matches_rgb_half_range_training_normalization():
    crop = np.array([[[0, 127, 255]]], dtype=np.uint8)  # BGR

    batch = preprocess_reid_crops(crop[None, ...], [1, 1], [0.5] * 3, [0.5] * 3)

    assert batch.shape == (1, 3, 1, 1)
    np.testing.assert_allclose(
        batch[0, :, 0, 0],
        np.array([1.0, 127.0 / 127.5 - 1.0, -1.0], dtype=np.float32),
        atol=1e-6,
    )


def test_preprocessing_rejects_invalid_crop_and_zero_std():
    with pytest.raises(ValueError, match="Invalid ReID crop"):
        preprocess_reid_crops([np.empty((0, 0, 3), dtype=np.uint8)], [256, 128], [0.5] * 3, [0.5] * 3)

    with pytest.raises(ValueError, match="std values must be positive"):
        preprocess_reid_crops([], [256, 128], [0.5] * 3, [0.5, 0.0, 0.5])
