"""Direct ONNX Runtime ReID client."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np

from .reid_preprocessing import preprocess_reid_crops


class ONNXRuntimeReIDClient:
    """Run a ReID ONNX model directly with ONNX Runtime."""

    def __init__(self, config):
        self.config = config
        self.onnx_path = self.resolve_model_path(config)

        self.mean = np.array(config["preprocessing"]["mean"], dtype=np.float32)
        self.std = np.array(config["preprocessing"]["std"], dtype=np.float32)
        self.color_space = config["preprocessing"].get("color_space", "RGB")
        self.channel_order = config["preprocessing"].get("channel_order", "CHW")
        self.input_shape = config["model"]["input_shape"]
        self.embedding_dim = int(config["model"]["embedding_dim"])
        inference_config = config.get("inference", {})
        self.max_batch_size = int(inference_config.get("max_batch_size", 1))

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Direct ONNX ReID requires onnxruntime.") from exc

        self.ort = ort
        ort_config = config.get("onnxruntime", {})
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.log_severity_level = int(ort_config.get("log_severity_level", 3))

        available = ort.get_available_providers()
        requested = ort_config.get(
            "providers",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        providers = [provider for provider in requested if provider in available]
        if not providers:
            providers = available

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=session_options,
            providers=providers,
        )
        active_providers = self.session.get_providers()
        if (
            "CUDAExecutionProvider" in requested
            and "CUDAExecutionProvider" not in active_providers
            and not ort_config.get("allow_cpu_fallback", False)
        ):
            raise RuntimeError(
                "CUDAExecutionProvider was requested for ReID, but ONNX Runtime "
                f"created a session with providers={active_providers}. This usually "
                "means the ONNX Runtime provider libraries are not visible to the "
                "dynamic linker. Start the app through scripts/start_reid_debug.sh "
                "or scripts/start_prime_dashboard.sh so LD_LIBRARY_PATH includes "
                "onnxruntime/capi and the CUDA/cuDNN library directories. To run "
                "intentionally on CPU, set onnxruntime.providers to "
                "['CPUExecutionProvider'] or set onnxruntime.allow_cpu_fallback: true."
            )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"✓ Loaded ONNX ReID model: {self.onnx_path}")
        print(f"✓ ONNX Runtime providers: {active_providers}")
        print(f"✓ ONNX input='{self.input_name}' output='{self.output_name}' max_batch={self.max_batch_size}")

    @staticmethod
    def resolve_model_path(config) -> Path:
        model_config = config["model"]
        configured = Path(str(model_config["onnx_path"])).expanduser()
        candidates = []
        environment_path = os.environ.get("REID_MODEL_PATH")
        if environment_path:
            candidates.append(Path(environment_path).expanduser())
        candidates.extend([configured, Path.home() / configured])
        for search_path in model_config.get("search_paths", []):
            candidate = Path(str(search_path)).expanduser()
            candidates.append(candidate / configured.name if candidate.is_dir() else candidate)

        checked = []
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in checked:
                continue
            checked.append(candidate)
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "ONNX ReID model not found. Checked: " + ", ".join(str(path) for path in checked)
        )

    def preprocess(self, crops: List[np.ndarray]) -> np.ndarray:
        return preprocess_reid_crops(
            crops,
            self.input_shape,
            self.mean,
            self.std,
            self.color_space,
            self.channel_order,
        )

    def infer(self, crops: List[np.ndarray], retry=None, max_batch_size=None) -> np.ndarray:
        if len(crops) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        request_batch_size = int(self.max_batch_size if max_batch_size is None else max_batch_size)
        if request_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {request_batch_size}")
        request_batch_size = min(request_batch_size, self.max_batch_size)

        chunks = [
            self._infer_batch(crops[start:start + request_batch_size])
            for start in range(0, len(crops), request_batch_size)
        ]
        return np.vstack(chunks)

    def _infer_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        batch = self.preprocess(crops)
        outputs = self.session.run([self.output_name], {self.input_name: batch})
        embeddings = np.asarray(outputs[0], dtype=np.float32)

        expected_shape = (len(crops), self.embedding_dim)
        if embeddings.shape != expected_shape:
            raise ValueError(f"Unexpected ONNX output shape: {embeddings.shape}, expected {expected_shape}")
        return embeddings

    def get_model_metadata(self):
        return {
            "backend": "onnxruntime_direct",
            "onnx_path": str(self.onnx_path),
            "input": self.input_name,
            "output": self.output_name,
            "embedding_dim": self.embedding_dim,
            "max_batch_size": self.max_batch_size,
            "providers": self.session.get_providers(),
        }

    def get_model_config(self):
        return self.config

    def close(self):
        self.session = None
