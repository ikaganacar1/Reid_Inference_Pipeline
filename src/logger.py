"""
Experimental Logger
Comprehensive logging system for reproducibility and experiment tracking
"""

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class ExperimentLogger:
    """Comprehensive experimental logging for ReID pipeline"""

    def __init__(self, experiment_dir: Path, config: Dict[str, Any] = None):
        """
        Initialize experiment logger

        Args:
            experiment_dir: Directory to save experiment logs
            config: Pipeline configuration dict
        """
        self.exp_dir = Path(experiment_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Create log files
        self.detection_log = open(self.exp_dir / 'detections.jsonl', 'w')
        self.embedding_log = open(self.exp_dir / 'embeddings.jsonl', 'w')
        self.track_log = open(self.exp_dir / 'tracks.jsonl', 'w')
        self.metrics_log = open(self.exp_dir / 'metrics.jsonl', 'w')

        self.frame_count = 0

        print(f"Experiment logger initialized: {self.exp_dir}")

        # Log system info and config
        self._log_system_info()
        if config:
            self.log_config(config)

    def _log_system_info(self):
        """Log system information and environment"""
        system_info = {
            "timestamp": datetime.datetime.now().isoformat(),
            "hostname": os.uname().nodename,
            "system": {
                "os": os.uname().sysname,
                "release": os.uname().release,
                "machine": os.uname().machine,
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "python_version": os.sys.version
            }
        }

        # Add GPU info if available
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                system_info["gpus"] = [
                    {
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_total_mb": gpu.memoryTotal,
                        "driver": gpu.driver
                    }
                    for gpu in gpus
                ]
            except Exception:
                system_info["gpus"] = []

        with open(self.exp_dir / 'system_info.json', 'w') as f:
            json.dump(system_info, f, indent=2)

    def log_config(self, config: Dict[str, Any]):
        """Save configuration snapshot"""
        with open(self.exp_dir / 'config_snapshot.json', 'w') as f:
            json.dump(config, f, indent=2)

    def log_model_version(self, model_path: Path, model_type: str):
        """Compute and log model file hash for versioning"""
        sha256 = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        version_info = {
            "model_type": model_type,
            "model_path": str(model_path),
            "sha256": sha256.hexdigest(),
            "size_mb": model_path.stat().st_size / (1024**2),
            "timestamp": datetime.datetime.now().isoformat()
        }

        # Append to model versions file
        versions_file = self.exp_dir / 'model_versions.json'
        if versions_file.exists():
            with open(versions_file, 'r') as f:
                versions = json.load(f)
        else:
            versions = []

        versions.append(version_info)

        with open(versions_file, 'w') as f:
            json.dump(versions, f, indent=2)

    def log_video_metadata(self, video_path: Path):
        """Extract and log video metadata"""
        cap = cv2.VideoCapture(str(video_path))

        metadata = {
            "path": str(video_path),
            "filename": video_path.name,
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
            "duration_sec": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
            "size_mb": video_path.stat().st_size / (1024**2)
        }

        cap.release()

        with open(self.exp_dir / 'video_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def log_detections(self, frame_idx: int, detections: np.ndarray, inference_time_ms: float):
        """Log YOLO detections per frame"""
        log_entry = {
            "frame_idx": frame_idx,
            "timestamp": datetime.datetime.now().isoformat(),
            "num_detections": len(detections),
            "detections": detections.tolist() if len(detections) > 0 else [],
            "inference_time_ms": inference_time_ms
        }
        self.detection_log.write(json.dumps(log_entry) + '\n')
        self.detection_log.flush()

    def log_embeddings(self, frame_idx: int, bboxes: np.ndarray, embeddings: np.ndarray,
                       inference_time_ms: float, save_embeddings: bool = True):
        """Log ReID embeddings per frame"""
        log_entry = {
            "frame_idx": frame_idx,
            "timestamp": datetime.datetime.now().isoformat(),
            "num_embeddings": len(embeddings),
            "bboxes": bboxes.tolist() if len(bboxes) > 0 else [],
            "embeddings_saved": save_embeddings,
            "embeddings": embeddings.tolist() if save_embeddings and len(embeddings) > 0 else [],
            "inference_time_ms": inference_time_ms
        }
        self.embedding_log.write(json.dumps(log_entry) + '\n')
        self.embedding_log.flush()

    def log_tracks(self, frame_idx: int, tracks: np.ndarray):
        """Log tracking results per frame"""
        log_entry = {
            "frame_idx": frame_idx,
            "timestamp": datetime.datetime.now().isoformat(),
            "num_tracks": len(tracks),
            "tracks": tracks.tolist() if len(tracks) > 0 else []
        }
        self.track_log.write(json.dumps(log_entry) + '\n')
        self.track_log.flush()

    def log_performance(self, frame_idx: int, fps: float, gpu_memory_mb: float = None, latency_ms: float = None):
        """Log performance metrics"""
        log_entry = {
            "frame_idx": frame_idx,
            "timestamp": datetime.datetime.now().isoformat(),
            "fps": fps,
            "gpu_memory_mb": gpu_memory_mb,
            "latency_ms": latency_ms,
            "cpu_percent": psutil.cpu_percent(),
            "ram_used_gb": psutil.virtual_memory().used / (1024**3)
        }

        # Add GPU utilization if available
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    log_entry["gpu_util_percent"] = gpus[0].load * 100
                    log_entry["gpu_memory_used_mb"] = gpus[0].memoryUsed
            except Exception:
                pass

        self.metrics_log.write(json.dumps(log_entry) + '\n')
        self.metrics_log.flush()

    def close(self):
        """Close all log files"""
        self.detection_log.close()
        self.embedding_log.close()
        self.track_log.close()
        self.metrics_log.close()

        print(f"Experiment logs saved to: {self.exp_dir}")

    @staticmethod
    def create_experiment_id(prefix: str = "exp") -> str:
        """Create unique experiment ID with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"


if __name__ == "__main__":
    # Test logger

    exp_dir = Path("experiments") / ExperimentLogger.create_experiment_id("test")
    logger = ExperimentLogger(exp_dir)

    # Log some dummy data
    print("\nLogging test data...")

    # Model versions
    logger.log_model_version(Path("models/yolo11n.pt"), "yolo")

    # Detections
    dummy_dets = np.array([[100, 100, 200, 300, 0.9, 0]])
    logger.log_detections(0, dummy_dets, 15.5)

    # Embeddings
    dummy_embs = np.random.randn(1, 256).astype(np.float32)
    logger.log_embeddings(0, dummy_dets[:, :4], dummy_embs, 8.2)

    # Tracks
    dummy_tracks = np.array([[100, 100, 200, 300, 1, 0.9, 0]])
    logger.log_tracks(0, dummy_tracks)

    # Performance
    logger.log_performance(0, 25.5, 1024.0, 23.7)

    logger.close()

    print("\nLogger test completed!")
    print(f"Check logs in: {exp_dir}")
