"""
YOLO Person Detector
Wrapper for Ultralytics YOLO to detect persons and extract crops
"""

import hashlib
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


class YOLOPersonDetector:
    """YOLO-based person detector with crop extraction"""

    def __init__(self, config):
        """
        Initialize YOLO detector

        Args:
            config: YOLO configuration dict with model and detection settings
        """
        self.config = config
        configured_model_path = config['model']['path']
        self.model_path = Path(os.environ.get('YOLO_MODEL_PATH', configured_model_path)).expanduser()
        self.device = config['model']['device']

        # Detection parameters
        self.conf_threshold = config['detection']['conf_threshold']
        self.iou_threshold = config['detection']['iou_threshold']
        self.classes = config['detection']['classes']  # [0] for person only
        self.imgsz = config['detection']['imgsz']

        # Performance settings
        self.half = config['inference'].get('half', True)
        self.max_det = config['inference'].get('max_det', 300)

        self.backend = "tensorrt" if self.model_path.suffix == ".engine" else "ultralytics"
        print(f"Loading YOLO model: {self.model_path}")
        if self.backend == "tensorrt":
            if not self.model_path.exists():
                raise FileNotFoundError(f"TensorRT YOLO engine not found: {self.model_path}")
            self.model = TensorRTYOLOBackend(self.model_path, self.imgsz, device_id=self._parse_device_id(self.device))
        else:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.model.to(self.device)
            try:
                from ultralytics.cfg import DEFAULT_CFG_DICT

                self._precision_argument = (
                    "quantize" if "quantize" in DEFAULT_CFG_DICT else "half"
                )
            except ImportError:
                self._precision_argument = "half"

        # Compute model hash for versioning
        loaded_model_path = self._resolve_loaded_model_path()
        self.model_hash = self._calculate_hash(loaded_model_path) if loaded_model_path is not None else "unavailable"
        print(f"  Model hash: {self.model_hash[:16]}...")
        print(f"  Device: {self.device}")
        print(f"  Backend: {self.backend}")
        print(f"  FP16: {self.half}")
        print(f"  Confidence threshold: {self.conf_threshold}")

    def _parse_device_id(self, device: str) -> int:
        if isinstance(device, str) and ":" in device:
            return int(device.rsplit(":", 1)[1])
        return 0

    def _calculate_hash(self, model_path: Path) -> str:
        """Calculate SHA256 hash of model file"""
        sha256 = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _resolve_loaded_model_path(self) -> Path | None:
        """Return the local model file path after any Ultralytics auto-download."""
        if self.model_path.exists():
            return self.model_path

        for attr_name in ("ckpt_path", "pt_path"):
            path_value = getattr(self.model, attr_name, None)
            if path_value:
                candidate = Path(path_value)
                if candidate.exists():
                    return candidate

        return None

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Detect persons in frame and extract crops

        Args:
            frame: Input frame (H, W, 3) in BGR format

        Returns:
            detections: numpy array [N, 6] (x1, y1, x2, y2, conf, cls)
            crops: list of person crops (BGR images)
        """
        if self.backend == "tensorrt":
            detections = self.model.detect(
                frame,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                classes=self.classes,
                max_det=self.max_det,
            )
            return self._extract_crops(frame, detections)

        predict_args = {
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "classes": self.classes,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "verbose": False,
        }
        if self._precision_argument == "quantize":
            predict_args["quantize"] = 16 if self.half else None
        else:
            predict_args["half"] = self.half
        results = self.model(frame, **predict_args)[0]

        # Extract boxes
        if len(results.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32), []

        boxes_xyxy = results.boxes.xyxy.cpu().numpy()  # [N, 4]
        scores = results.boxes.conf.cpu().numpy()  # [N]
        classes = results.boxes.cls.cpu().numpy()  # [N]

        # Stack to [N, 6] format: x1, y1, x2, y2, conf, cls
        detections = np.hstack([
            boxes_xyxy,
            scores.reshape(-1, 1),
            classes.reshape(-1, 1)
        ])

        # YOLO26 end-to-end checkpoints can bypass Ultralytics' conventional
        # NMS path. Apply a final class-aware NMS here so one person cannot
        # enter the tracker twice from nearly identical boxes.
        detections = self._apply_class_aware_nms(detections)
        detections, crops = self._extract_crops(frame, detections)
        return detections, crops

    def _apply_class_aware_nms(self, detections: np.ndarray) -> np.ndarray:
        if len(detections) <= 1:
            return np.asarray(detections, dtype=np.float32)

        keep: list[int] = []
        class_ids = detections[:, 5].astype(np.int64)
        for class_id in np.unique(class_ids):
            indices = np.flatnonzero(class_ids == class_id)
            boxes = [
                [
                    float(detections[index, 0]),
                    float(detections[index, 1]),
                    float(max(0.0, detections[index, 2] - detections[index, 0])),
                    float(max(0.0, detections[index, 3] - detections[index, 1])),
                ]
                for index in indices
            ]
            scores = detections[indices, 4].astype(float).tolist()
            selected = cv2.dnn.NMSBoxes(
                boxes,
                scores,
                float(self.conf_threshold),
                float(self.iou_threshold),
            )
            if len(selected):
                keep.extend(indices[np.asarray(selected).reshape(-1)].tolist())

        keep.sort(key=lambda index: (-float(detections[index, 4]), index))
        return np.asarray(detections[keep[: self.max_det]], dtype=np.float32)

    def _extract_crops(self, frame: np.ndarray, detections: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
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


class TensorRTYOLOBackend:
    """Minimal TensorRT runner for Ultralytics YOLO detect engines."""

    def __init__(self, engine_path: Path, imgsz: int, device_id: int = 0):
        try:
            import tensorrt as trt
            from src.tensorrt_reid_client import _CudaPythonBackend
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT YOLO backend requires TensorRT Python and cuda-python==12.5.0"
            ) from exc

        self.trt = trt
        self.cuda = _CudaPythonBackend()
        self.device_id = device_id
        self.imgsz = int(imgsz)
        self.logger = trt.Logger(trt.Logger.WARNING)

        self.cuda.set_device(self.device_id)
        with Path(engine_path).open("rb") as engine_file:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = self.cuda.create_stream()
        self.input_name, self.output_name = self._resolve_io_names()
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
        self.input_allocation = None
        self.input_nbytes = 0
        self.output_allocation = None
        self.output_nbytes = 0

        print(f"  TensorRT YOLO input='{self.input_name}' output='{self.output_name}'")

    def _resolve_io_names(self) -> tuple[str, str]:
        inputs = []
        outputs = []
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                inputs.append(name)
            elif mode == self.trt.TensorIOMode.OUTPUT:
                outputs.append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"Expected one YOLO input/output, got inputs={inputs}, outputs={outputs}")
        return inputs[0], outputs[0]

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        classes: list[int],
        max_det: int,
    ) -> np.ndarray:
        batch, ratio, pad = self._preprocess(frame)
        self.cuda.set_device(self.device_id)
        self._set_input_shape(batch.shape)
        output_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(self.output_name))
        output = np.empty(output_shape, dtype=self.output_dtype)

        d_input = self._ensure_input_allocation(batch.nbytes)
        d_output = self._ensure_output_allocation(output.nbytes)
        self.context.set_tensor_address(self.input_name, self.cuda.ptr(d_input))
        self.context.set_tensor_address(self.output_name, self.cuda.ptr(d_output))
        self.cuda.memcpy_htod_async(d_input, batch, self.stream)
        if not self.context.execute_async_v3(stream_handle=self.cuda.stream_handle(self.stream)):
            raise RuntimeError("TensorRT YOLO execute_async_v3 returned false")
        self.cuda.memcpy_dtoh_async(output, d_output, self.stream)
        self.cuda.synchronize(self.stream)

        return self._postprocess(output, frame.shape[:2], ratio, pad, conf_threshold, iou_threshold, classes, max_det)

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        h, w = frame.shape[:2]
        ratio = min(self.imgsz / h, self.imgsz / w)
        new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
        dw = (self.imgsz - new_w) / 2
        dh = (self.imgsz - new_h) / 2

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        canvas[top:top + new_h, left:left + new_w] = resized

        image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image[None], dtype=np.float32) / 255.0
        return image.astype(self.input_dtype, copy=False), ratio, (dw, dh)

    def _postprocess(
        self,
        output: np.ndarray,
        frame_shape: tuple[int, int],
        ratio: float,
        pad: tuple[float, float],
        conf_threshold: float,
        iou_threshold: float,
        classes: list[int],
        max_det: int,
    ) -> np.ndarray:
        pred = np.squeeze(output)
        if pred.ndim != 2:
            raise RuntimeError(f"Unexpected YOLO output shape after squeeze: {pred.shape}")
        if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 256:
            pred = pred.T

        boxes = pred[:, :4]
        scores_all = pred[:, 4:]
        class_ids = np.argmax(scores_all, axis=1)
        scores = scores_all[np.arange(scores_all.shape[0]), class_ids]

        mask = scores >= conf_threshold
        if classes:
            mask &= np.isin(class_ids, np.asarray(classes))
        if not np.any(mask):
            return np.empty((0, 6), dtype=np.float32)

        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        xyxy = np.empty_like(boxes, dtype=np.float32)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        dw, dh = pad
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dw) / ratio
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dh) / ratio

        frame_h, frame_w = frame_shape
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, frame_w - 1)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, frame_h - 1)

        nms_boxes = [
            [float(x1), float(y1), float(max(0, x2 - x1)), float(max(0, y2 - y1))]
            for x1, y1, x2, y2 in xyxy
        ]
        keep = cv2.dnn.NMSBoxes(nms_boxes, scores.astype(float).tolist(), conf_threshold, iou_threshold)
        if len(keep) == 0:
            return np.empty((0, 6), dtype=np.float32)

        keep = np.asarray(keep).reshape(-1)[:max_det]
        detections = np.column_stack([xyxy[keep], scores[keep], class_ids[keep]]).astype(np.float32)
        return detections

    def _set_input_shape(self, shape: tuple[int, ...]) -> None:
        if self.context.get_tensor_shape(self.input_name) != shape:
            self.context.set_input_shape(self.input_name, shape)

    def _ensure_input_allocation(self, nbytes: int):
        if self.input_allocation is None or nbytes > self.input_nbytes:
            self.cuda.mem_free(self.input_allocation)
            self.input_allocation = self.cuda.mem_alloc(nbytes)
            self.input_nbytes = nbytes
        return self.input_allocation

    def _ensure_output_allocation(self, nbytes: int):
        if self.output_allocation is None or nbytes > self.output_nbytes:
            self.cuda.mem_free(self.output_allocation)
            self.output_allocation = self.cuda.mem_alloc(nbytes)
            self.output_nbytes = nbytes
        return self.output_allocation


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

        print("\nTesting detection on dummy frame...")
        detections, crops = detector.detect(dummy_frame)

        print(f"  Detections: {len(detections)}")
        print(f"  Crops: {len(crops)}")

        if len(detections) > 0:
            print(f"  Detection shapes: {detections.shape}")
            print(f"  First detection: {detections[0]}")

        print("\nYOLO detector test completed!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
