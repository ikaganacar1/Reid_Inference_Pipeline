"""Shared person-crop preprocessing for every ReID inference backend."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def preprocess_reid_crops(
    crops: Sequence[np.ndarray],
    input_shape: Sequence[int],
    mean: Sequence[float] | np.ndarray,
    std: Sequence[float] | np.ndarray,
    color_space: str = "RGB",
    channel_order: str = "CHW",
) -> np.ndarray:
    """Convert BGR uint8 crops to the model's contiguous float32 batch."""
    if len(input_shape) != 2:
        raise ValueError(f"ReID input_shape must be [height, width], got {input_shape}")
    height, width = (int(input_shape[0]), int(input_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"ReID input dimensions must be positive, got {input_shape}")

    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    if mean_array.shape != (3,) or std_array.shape != (3,):
        raise ValueError("ReID mean and std must each contain exactly three values")
    if not np.all(np.isfinite(mean_array)) or not np.all(np.isfinite(std_array)):
        raise ValueError("ReID mean and std must be finite")
    if np.any(std_array <= 0):
        raise ValueError("ReID std values must be positive")

    color_space = str(color_space).upper()
    channel_order = str(channel_order).upper()
    if color_space not in {"RGB", "BGR"}:
        raise ValueError(f"Unsupported ReID color_space: {color_space}")
    if channel_order != "CHW":
        raise ValueError(f"Unsupported ReID channel_order: {channel_order}")

    batch = []
    for index, crop in enumerate(crops):
        if not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.shape[2] != 3 or crop.size == 0:
            shape = getattr(crop, "shape", None)
            raise ValueError(f"Invalid ReID crop at index {index}: expected non-empty HxWx3, got {shape}")

        image = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
        if color_space == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = (image - mean_array) / std_array
        batch.append(np.transpose(image, (2, 0, 1)))

    if not batch:
        return np.empty((0, 3, height, width), dtype=np.float32)
    return np.ascontiguousarray(np.stack(batch).astype(np.float32, copy=False))
