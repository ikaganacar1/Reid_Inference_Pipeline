"""
YOLO Person Detector
Wrapper for Ultralytics YOLO to detect persons and extract crops
"""

import hashlib
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO


class YOLOPersonDetector:
    """YOLO-based person detector with crop extraction"""

    def __init__(self, config):
        """
        Initialize YOLO detector

        Args:
            config: YOLO configuration dict with model and detection settings
        """
        self.config = config
        self.model_path = Path(config['model']['path'])
        self.device = config['model']['device']

        # Detection parameters
        self.conf_threshold = config['detection']['conf_threshold']
        self.iou_threshold = config['detection']['iou_threshold']
        self.classes = config['detection']['classes']  # [0] for person only
        self.imgsz = config['detection']['imgsz']

        # Performance settings
        self.half = config['inference'].get('half', True)
        self.max_det = config['inference'].get('max_det', 300)

        # Load YOLO model
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

        print(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)

        # Compute model hash for versioning
        self.model_hash = self._calculate_hash(self.model_path)
        print(f"  Model hash: {self.model_hash[:16]}...")
        print(f"  Device: {self.device}")
        print(f"  FP16: {self.half}")
        print(f"  Confidence threshold: {self.conf_threshold}")

    def _calculate_hash(self, model_path: Path) -> str:
        """Calculate SHA256 hash of model file"""
        sha256 = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Detect persons in frame and extract crops

        Args:
            frame: Input frame (H, W, 3) in BGR format

        Returns:
            detections: numpy array [N, 6] (x1, y1, x2, y2, conf, cls)
            crops: list of person crops (BGR images)
        """
        start_time = time.time()

        # Run YOLO inference
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.imgsz,
            half=self.half,
            max_det=self.max_det,
            verbose=False
        )[0]

        # Extract boxes
        if len(results.boxes) == 0:
            return np.array([]), []

        boxes_xyxy = results.boxes.xyxy.cpu().numpy()  # [N, 4]
        scores = results.boxes.conf.cpu().numpy()  # [N]
        classes = results.boxes.cls.cpu().numpy()  # [N]

        # Stack to [N, 6] format: x1, y1, x2, y2, conf, cls
        detections = np.hstack([
            boxes_xyxy,
            scores.reshape(-1, 1),
            classes.reshape(-1, 1)
        ])

        # Extract crops
        crops = []
        for x1, y1, x2, y2, _, _ in detections:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Validate coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Extract crop
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                crops.append(crop)
            else:
                # Invalid bbox, add empty crop
                crops.append(np.zeros((1, 1, 3), dtype=np.uint8))

        inference_time = time.time() - start_time

        return detections, crops

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_path": str(self.model_path),
            "model_hash": self.model_hash,
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "classes": self.classes,
            "fp16": self.half
        }


if __name__ == "__main__":
    # Test script
    import sys
    import yaml
    import cv2

    # Load config
    config_path = Path("configs/yolo_config.yaml")
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create detector
    try:
        detector = YOLOPersonDetector(config)

        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        print(f"\nTesting detection on dummy frame...")
        detections, crops = detector.detect(dummy_frame)

        print(f"  Detections: {len(detections)}")
        print(f"  Crops: {len(crops)}")

        if len(detections) > 0:
            print(f"  Detection shapes: {detections.shape}")
            print(f"  First detection: {detections[0]}")

        print(f"\nYOLO detector test completed!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
