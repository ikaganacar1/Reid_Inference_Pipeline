"""
YOLOE Person Detector
Open-vocabulary detection using YOLOE-26 with text prompts.
Drop-in replacement for YOLOPersonDetector — same detect() interface.

Why YOLOE over YOLO11n for intruder detection:
- Text-prompted: "person" is embedded via CLIP, not a fixed class index
- Open-vocabulary: prompt can be changed to "person wearing dark clothes",
  "person carrying bag", etc. without retraining
- Combined with ReID: YOLOE finds all people, ReID embeddings distinguish
  the specific intruder by appearance across cameras
"""

import hashlib
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from ultralytics import YOLOE


class YOLOEPersonDetector:
    """
    YOLOE-26 open-vocabulary person detector with text prompt support.
    Identical detect() interface to YOLOPersonDetector.
    """

    def __init__(self, config):
        """
        Args:
            config: yoloe_config dict (model, detection, inference sections)
        """
        self.config = config
        self.model_path = Path(config['model']['path'])
        self.device = config['model']['device']
        self.text_prompts = config['model'].get('text_prompts', ['person'])

        self.conf_threshold = config['detection']['conf_threshold']
        self.iou_threshold  = config['detection']['iou_threshold']
        self.imgsz          = config['detection']['imgsz']
        self.half           = config['inference'].get('half', True)
        self.max_det        = config['inference'].get('max_det', 300)

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLOE model not found: {self.model_path}")

        print(f"Loading YOLOE model: {self.model_path}")
        self.model = YOLOE(str(self.model_path))
        self.model.to(self.device)

        # Embed text prompts once — this runs CLIP, no cost at inference time
        print(f"  Setting text prompts: {self.text_prompts}")
        text_pe = self.model.get_text_pe(self.text_prompts)
        self.model.set_classes(self.text_prompts, text_pe)

        self.model_hash = self._calculate_hash(self.model_path)
        print(f"  Model hash: {self.model_hash[:16]}...")
        print(f"  Device: {self.device}")
        print(f"  FP16: {self.half}")
        print(f"  Confidence threshold: {self.conf_threshold}")
        print(f"  Text prompts: {self.text_prompts}")

    def _calculate_hash(self, model_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Detect persons in frame using text-prompted YOLOE.

        Returns:
            detections: [N, 6] (x1, y1, x2, y2, conf, cls=0)
            crops:      list of BGR person crops
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            half=self.half,
            max_det=self.max_det,
            verbose=False
        )[0]

        if len(results.boxes) == 0:
            return np.array([]), []

        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
        scores     = results.boxes.conf.cpu().numpy()
        classes    = results.boxes.cls.cpu().numpy()

        detections = np.hstack([
            boxes_xyxy,
            scores.reshape(-1, 1),
            classes.reshape(-1, 1)
        ])

        crops = []
        for x1, y1, x2, y2, _, _ in detections:
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(frame.shape[1], int(x2))
            y2 = min(frame.shape[0], int(y2))
            if x2 > x1 and y2 > y1:
                crops.append(frame[y1:y2, x1:x2])
            else:
                crops.append(np.zeros((1, 1, 3), dtype=np.uint8))

        return detections, crops

    def get_model_info(self) -> dict:
        return {
            "model_path":     str(self.model_path),
            "model_hash":     self.model_hash,
            "device":         self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold":  self.iou_threshold,
            "text_prompts":   self.text_prompts,
            "fp16":           self.half,
        }
