"""
Enhanced Object Detector with TensorRT Support
Optimized for NVIDIA Jetson deployment with YOLOv11
"""
import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import torch
import logging
from pathlib import Path


@dataclass
class Detection:
    """Detection result with bounding box and metadata"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    reid_embedding: Optional[np.ndarray] = None


class EnhancedObjectDetector:
    """
    Enhanced YOLO detector with TensorRT optimization.
    
    Features:
    - TensorRT model loading for maximum performance
    - Batch inference support
    - Confidence stratification (high/medium/low)
    - Optimized NMS threshold for person tracking
    """
    
    def __init__(self,
                 model_path: str = 'yolo11n.pt',
                 device: str = 'auto',
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.55,
                 conf_high: float = 0.8,
                 conf_medium: float = 0.3,
                 use_tensorrt: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize enhanced object detector.
        
        Args:
            model_path: Path to YOLO model (.pt or .engine)
            device: Device for inference ('cuda', 'cpu', or 'auto')
            conf_threshold: Base confidence threshold (0.25-0.35 sweet spot)
            iou_threshold: NMS IoU threshold (0.5-0.6 for person tracking)
            conf_high: High confidence threshold (0.8+)
            conf_medium: Medium confidence threshold (0.3-0.8)
            use_tensorrt: Use TensorRT optimization
        """
        # Device selection
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Initializing Enhanced Object Detector on {self.device}...")
        
        # Load model
        self.model = YOLO(model_path)
        
        # Move to device
        if self.device == 'cuda':
            self.model.to('cuda')
        
        # Thresholds
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.conf_high = conf_high
        self.conf_medium = conf_medium
        
        # COCO class names
        self.class_names = self.model.names
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'high_conf_detections': 0,
            'medium_conf_detections': 0,
            'low_conf_detections': 0,
            'frames_processed': 0
        }
        
        self.logger.info(f"Detector initialized: conf_threshold={conf_threshold}, "
                        f"iou_threshold={iou_threshold}")
    
    def stratify_detections(self,
                           detections: List[Detection]) -> Tuple[List[Detection], List[Detection], List[Detection]]:
        """
        Stratify detections by confidence level.
        
        High confidence: Rely primarily on IoU matching
        Medium confidence: Full ReID matching required
        Low confidence: ReID only if strong track history
        
        Returns:
            (high_conf_dets, medium_conf_dets, low_conf_dets)
        """
        high_conf = []
        medium_conf = []
        low_conf = []
        
        for det in detections:
            if det.confidence >= self.conf_high:
                high_conf.append(det)
                self.stats['high_conf_detections'] += 1
            elif det.confidence >= self.conf_medium:
                medium_conf.append(det)
                self.stats['medium_conf_detections'] += 1
            else:
                low_conf.append(det)
                self.stats['low_conf_detections'] += 1
        
        return high_conf, medium_conf, low_conf
    
    def detect(self,
              frame: np.ndarray,
              target_classes: Optional[List[int]] = None,
              return_stratified: bool = False) -> Union[List[Detection], 
                                                        Tuple[List[Detection], List[Detection], List[Detection]]]:
        """
        Detect objects in frame.
        
        Args:
            frame: Input frame (H, W, 3)
            target_classes: Filter by class IDs (None = all classes)
            return_stratified: Return stratified detections
            
        Returns:
            Detections or (high_conf, medium_conf, low_conf) if stratified
        """
        # Run inference
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=target_classes,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes
                
                for i in range(len(boxes)):
                    # Extract detection info
                    bbox = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Create Detection object
                    detection = Detection(
                        bbox=bbox,
                        confidence=confidence,
                        class_id=class_id,
                        class_name=self.class_names[class_id]
                    )
                    
                    detections.append(detection)
        
        self.stats['total_detections'] += len(detections)
        self.stats['frames_processed'] += 1
        
        if return_stratified:
            return self.stratify_detections(detections)
        
        return detections
    
    def detect_persons(self,
                      frame: np.ndarray,
                      return_stratified: bool = False) -> Union[List[Detection], 
                                                                Tuple[List[Detection], List[Detection], List[Detection]]]:
        """
        Detect persons only (class 0 in COCO).
        
        Args:
            frame: Input frame
            return_stratified: Return confidence-stratified detections
            
        Returns:
            Person detections
        """
        return self.detect(frame, target_classes=[0], return_stratified=return_stratified)
    
    def detect_batch(self,
                    frames: List[np.ndarray],
                    target_classes: Optional[List[int]] = None) -> List[List[Detection]]:
        """
        Batch detection for multiple frames.
        
        Args:
            frames: List of input frames
            target_classes: Filter by class IDs
            
        Returns:
            List of detection lists (one per frame)
        """
        # YOLO ultralytics handles batching internally
        results = self.model(
            frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=target_classes,
            device=self.device,
            verbose=False,
            stream=False  # Process as batch
        )
        
        all_detections = []
        
        for r in results:
            frame_detections = []
            
            if r.boxes is not None:
                boxes = r.boxes
                
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    detection = Detection(
                        bbox=bbox,
                        confidence=confidence,
                        class_id=class_id,
                        class_name=self.class_names[class_id]
                    )
                    
                    frame_detections.append(detection)
            
            all_detections.append(frame_detections)
            self.stats['total_detections'] += len(frame_detections)
        
        self.stats['frames_processed'] += len(frames)
        
        return all_detections
    
    def filter_overlapping_detections(self,
                                     detections: List[Detection],
                                     iou_threshold: float = 0.6) -> Tuple[List[Detection], List[Detection]]:
        """
        Separate overlapping from non-overlapping detections.
        
        Process non-overlapping first for initial clustering,
        then assign overlapping boxes in final step.
        
        Args:
            detections: Input detections
            iou_threshold: IoU threshold for overlap (0.6)
            
        Returns:
            (non_overlapping, overlapping)
        """
        if len(detections) == 0:
            return [], []
        
        # Compute IoU matrix
        bboxes = np.array([det.bbox for det in detections])
        n = len(bboxes)
        iou_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                iou = self._compute_iou(bboxes[i], bboxes[j])
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
        
        # Find overlapping detections
        overlapping_mask = np.any(iou_matrix > iou_threshold, axis=1)
        
        non_overlapping = [det for i, det in enumerate(detections) if not overlapping_mask[i]]
        overlapping = [det for i, det in enumerate(detections) if overlapping_mask[i]]
        
        return non_overlapping, overlapping
    
    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return float(inter_area / union_area)
    
    def draw_detections(self,
                       frame: np.ndarray,
                       detections: List[Detection],
                       draw_labels: bool = True,
                       color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Draw detections on frame.
        
        Args:
            frame: Input frame
            detections: Detections to draw
            draw_labels: Draw labels
            color: Box color (None = auto by confidence)
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            
            # Color by confidence if not specified
            if color is None:
                if det.confidence >= self.conf_high:
                    box_color = (0, 255, 0)  # Green - high confidence
                elif det.confidence >= self.conf_medium:
                    box_color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    box_color = (0, 165, 255)  # Orange - low confidence
            else:
                box_color = color
            
            # Draw box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            
            if draw_labels:
                # Prepare label
                label = f"{det.class_name}: {det.confidence:.2f}"
                if det.track_id is not None:
                    label = f"ID:{det.track_id} " + label
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame,
                            (x1, y1 - label_size[1] - 4),
                            (x1 + label_size[0], y1),
                            box_color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label,
                          (x1, y1 - 2),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def get_statistics(self) -> dict:
        """Get detector statistics"""
        if self.stats['frames_processed'] == 0:
            avg_dets_per_frame = 0
        else:
            avg_dets_per_frame = self.stats['total_detections'] / self.stats['frames_processed']
        
        return {
            'frames_processed': self.stats['frames_processed'],
            'total_detections': self.stats['total_detections'],
            'avg_detections_per_frame': avg_dets_per_frame,
            'high_conf_detections': self.stats['high_conf_detections'],
            'medium_conf_detections': self.stats['medium_conf_detections'],
            'low_conf_detections': self.stats['low_conf_detections'],
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'device': self.device
        }
    
    def export_to_onnx(self, output_path: str):
        """Export YOLO model to ONNX for TensorRT conversion"""
        self.logger.info(f"Exporting YOLO model to ONNX: {output_path}")
        
        # Ultralytics YOLO has built-in ONNX export
        self.model.export(format='onnx', opset=11, simplify=True)
        
        self.logger.info(f"Model exported successfully")


if __name__ == "__main__":
    # Test the enhanced detector
    logging.basicConfig(level=logging.INFO)
    
    detector = EnhancedObjectDetector(
        model_path='yolo11n.pt',
        conf_threshold=0.3,
        iou_threshold=0.55
    )
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    detections = detector.detect_persons(dummy_image, return_stratified=False)
    print(f"Detected {len(detections)} persons")
    
    # Test stratification
    high, medium, low = detector.detect_persons(dummy_image, return_stratified=True)
    print(f"High conf: {len(high)}, Medium: {len(medium)}, Low: {len(low)}")
    
    print(f"Statistics: {detector.get_statistics()}")
