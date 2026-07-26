import numpy as np

from src.detector import YOLOPersonDetector


def detector_for_nms(iou_threshold=0.7):
    detector = YOLOPersonDetector.__new__(YOLOPersonDetector)
    detector.conf_threshold = 0.5
    detector.iou_threshold = iou_threshold
    detector.max_det = 50
    return detector


def test_post_model_nms_removes_duplicate_end_to_end_boxes():
    detections = np.array(
        [
            [100, 100, 300, 500, 0.60, 0],
            [102, 101, 298, 498, 0.58, 0],
            [500, 100, 700, 500, 0.90, 0],
        ],
        dtype=np.float32,
    )

    filtered = detector_for_nms()._apply_class_aware_nms(detections)

    assert len(filtered) == 2
    np.testing.assert_allclose(filtered[:, 4], [0.90, 0.60])


def test_post_model_nms_is_class_aware():
    detections = np.array(
        [
            [100, 100, 300, 500, 0.90, 0],
            [100, 100, 300, 500, 0.80, 1],
        ],
        dtype=np.float32,
    )

    filtered = detector_for_nms()._apply_class_aware_nms(detections)

    assert len(filtered) == 2
