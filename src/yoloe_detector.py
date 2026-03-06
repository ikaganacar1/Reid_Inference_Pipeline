"""
YOLOE-26x Open-Vocabulary Detector
Supports two modes:

1. TEXT MODE (CAM1 / general detection)
   set_classes(['person']) — CLIP-embedded text prompt, finds all people.

2. VISUAL PROMPT MODE (CAM2 / targeted detection)
   Uses a reference crop of the specific burglar saved from CAM1.
   YOLOE's SAVPE encoder matches appearance to find ONLY that person,
   not just any person. This is the key differentiator from YOLO11n.

Both modes return bounding boxes + segmentation masks.
"""

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor


class YOLOEPersonDetector:
    """
    YOLOE-26x detector with text and visual prompt modes.
    Same detect() interface as YOLOPersonDetector, plus masks.
    """

    def __init__(self, config):
        self.config       = config
        self.model_path   = Path(config['model']['path'])
        self.device       = config['model']['device']
        self.text_prompts = config['model'].get('text_prompts', ['person'])

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

        # Start in text-prompt mode
        self._mode = "text"
        self._refer_frame: Optional[np.ndarray] = None
        self._refer_bboxes: Optional[np.ndarray] = None
        self._set_text_mode()

        self.model_hash = self._hash(self.model_path)
        print(f"  Model hash: {self.model_hash[:16]}...")
        print(f"  Device: {self.device}  |  FP16: {self.half}")
        print(f"  Conf: {self.conf_threshold}  |  Prompts: {self.text_prompts}")

    # ── Mode switching ─────────────────────────────────────────────────────

    def _set_text_mode(self):
        text_pe = self.model.get_text_pe(self.text_prompts)
        self.model.set_classes(self.text_prompts, text_pe)
        self._mode = "text"
        print(f"  [YOLOE] Mode: TEXT  prompts={self.text_prompts}")

    def set_visual_prompt(self, refer_frame: np.ndarray, refer_bboxes: np.ndarray):
        """
        Switch to visual-prompt mode.

        Args:
            refer_frame:  Full BGR frame from CAM1 containing the burglar
            refer_bboxes: [N, 4] float32 array of xyxy bboxes in refer_frame
        """
        self._refer_frame  = refer_frame
        self._refer_bboxes = refer_bboxes.astype(np.float32)
        self._mode = "visual"
        print(f"  [YOLOE] Mode: VISUAL PROMPT  "
              f"ref_shape={refer_frame.shape}  bboxes={refer_bboxes.tolist()}")

    # ── Inference ──────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[Optional[np.ndarray]]]:
        """
        Detect persons in frame.

        Returns:
            detections: [N, 6] (x1, y1, x2, y2, conf, cls)
            crops:      list of BGR person crops
            masks:      list of binary masks [H, W] or None per detection
        """
        if self._mode == "visual":
            return self._detect_visual(frame)
        return self._detect_text(frame)

    def _detect_text(self, frame):
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            half=self.half,
            max_det=self.max_det,
            verbose=False,
        )[0]
        return self._parse_results(results, frame)

    def _detect_visual(self, frame):
        visual_prompts = dict(
            bboxes=self._refer_bboxes,
            cls=np.zeros(len(self._refer_bboxes), dtype=np.int64),
        )
        results = self.model.predict(
            frame,
            refer_image=self._refer_frame,
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            max_det=self.max_det,
            verbose=False,
        )[0]
        return self._parse_results(results, frame)

    def _parse_results(self, results, frame) -> Tuple[np.ndarray, List, List]:
        if len(results.boxes) == 0:
            return np.array([]), [], []

        boxes  = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        cls    = results.boxes.cls.cpu().numpy()
        dets   = np.hstack([boxes, scores.reshape(-1, 1), cls.reshape(-1, 1)])

        # Extract per-detection binary masks (H, W) in original frame space
        H, W = frame.shape[:2]
        masks_out = []
        if results.masks is not None:
            raw_masks = results.masks.data.cpu().numpy()   # [N, mh, mw]
            for i in range(len(dets)):
                m = raw_masks[i]
                m_up = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
                masks_out.append((m_up > 0.5).astype(np.uint8))
        else:
            masks_out = [None] * len(dets)

        # Person crops
        crops = []
        for x1, y1, x2, y2, _, _ in dets:
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(W, int(x2)); y2 = min(H, int(y2))
            if x2 > x1 and y2 > y1:
                crops.append(frame[y1:y2, x1:x2])
            else:
                crops.append(np.zeros((1, 1, 3), dtype=np.uint8))

        return dets, crops, masks_out

    # ── Helpers ────────────────────────────────────────────────────────────

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
