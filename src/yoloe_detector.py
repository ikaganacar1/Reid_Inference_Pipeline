"""
YOLOE-26x Open-Vocabulary Detector
Text-prompted detection with two classes: "person" and "intruder".
YOLOE uses CLIP to classify each detected box as one or the other.
ReID (Swin Base via Triton) then handles cross-camera re-identification.

Same detect() interface as YOLOPersonDetector, plus segmentation masks.
"""

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLOE


class YOLOEPersonDetector:
    """YOLOE-26x with text prompts. Drop-in for YOLOPersonDetector."""

    def __init__(self, config):
        self.config       = config
        self.model_path   = Path(config['model']['path'])
        self.device       = config['model']['device']
        self.text_prompts = config['model'].get('text_prompts', ['person', 'intruder'])

        self.conf_threshold = config['detection']['conf_threshold']
        self.iou_threshold  = config['detection']['iou_threshold']
        self.imgsz          = config['detection']['imgsz']
        self.half           = config['inference'].get('half', False)
        self.max_det        = config['inference'].get('max_det', 300)

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLOE model not found: {self.model_path}")

        print(f"Loading YOLOE model: {self.model_path}")
        self.model = YOLOE(str(self.model_path))
        self.model.to(self.device)

        # Embed text prompts once via CLIP — zero extra cost at inference time
        print(f"  Setting text prompts: {self.text_prompts}")
        text_pe = self.model.get_text_pe(self.text_prompts)
        self.model.set_classes(self.text_prompts, text_pe)

        self.model_hash = self._hash(self.model_path)
        print(f"  Model hash: {self.model_hash[:16]}...")
        print(f"  Device: {self.device}  |  FP16: {self.half}")
        print(f"  Conf: {self.conf_threshold}  |  Prompts: {self.text_prompts}")

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[Optional[np.ndarray]]]:
        """
        Returns:
            detections: [N, 6] (x1, y1, x2, y2, conf, cls)
                        cls=0 → self.text_prompts[0]  ("person")
                        cls=1 → self.text_prompts[1]  ("intruder")
            crops:      BGR person crops for ReID
            masks:      binary [H,W] segmentation mask per detection, or None
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            half=self.half,
            max_det=self.max_det,
            verbose=False,
        )[0]

        if len(results.boxes) == 0:
            return np.array([]), [], []

        boxes  = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        cls    = results.boxes.cls.cpu().numpy()
        dets   = np.hstack([boxes, scores.reshape(-1, 1), cls.reshape(-1, 1)])

        H, W = frame.shape[:2]

        # Segmentation masks upsampled to original frame size
        masks_out = []
        if results.masks is not None:
            raw = results.masks.data.cpu().numpy()   # [N, mh, mw]
            for m in raw:
                m_up = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
                masks_out.append((m_up > 0.5).astype(np.uint8))
        else:
            masks_out = [None] * len(dets)

        # Person crops for ReID
        crops = []
        for x1, y1, x2, y2, _, _ in dets:
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(W, int(x2)); y2 = min(H, int(y2))
            crops.append(frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1
                         else np.zeros((1, 1, 3), dtype=np.uint8))

        return dets, crops, masks_out

    @staticmethod
    def _hash(path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha.update(chunk)
        return sha.hexdigest()

    def get_model_info(self) -> dict:
        return {
            "model_path":     str(self.model_path),
            "model_hash":     self.model_hash,
            "device":         self.device,
            "conf_threshold": self.conf_threshold,
            "text_prompts":   self.text_prompts,
            "fp16":           self.half,
        }
