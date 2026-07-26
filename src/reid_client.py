"""
Triton ReID Client
HTTP client wrapper for TAO ReID model served by Triton Inference Server
"""

import time
from typing import List

import numpy as np

from .reid_preprocessing import preprocess_reid_crops


def _load_triton_http_client():
    """Import Triton HTTP support only when the Triton backend is used."""
    try:
        import tritonclient.http as httpclient
        from tritonclient.utils import InferenceServerException
    except Exception as exc:
        raise RuntimeError(
            "Triton HTTP client support is not available. Install "
            "'tritonclient[http]' or use the 'tensorrt_direct' backend."
        ) from exc
    return httpclient, InferenceServerException


class TritonReIDClient:
    """Triton Inference Server client for TAO ReID model"""

    def __init__(self, config):
        """
        Initialize Triton client

        Args:
            config: ReID configuration dict with triton and preprocessing settings
        """
        self.config = config
        self.triton_url = config['triton']['server_url']
        self.model_name = config['triton']['model_name']
        self.model_version = config['triton']['model_version']

        # Preprocessing parameters
        self.mean = np.array(config['preprocessing']['mean'], dtype=np.float32)
        self.std = np.array(config['preprocessing']['std'], dtype=np.float32)
        self.color_space = config['preprocessing'].get('color_space', 'RGB')
        self.channel_order = config['preprocessing'].get('channel_order', 'CHW')
        self.input_shape = config['model']['input_shape']  # [H, W]
        self.embedding_dim = config['model']['embedding_dim']  # Embedding dimension
        inference_config = config.get('inference', {})
        self.max_batch_size = inference_config.get('max_batch_size', 16)
        self.max_retry = inference_config.get('max_retry', 3)
        self.timeout_ms = inference_config.get('timeout_ms', 5000)
        self.httpclient, self.InferenceServerException = _load_triton_http_client()

        # Create Triton client
        try:
            self.client = self.httpclient.InferenceServerClient(
                url=self.triton_url,
                verbose=False,
                connection_timeout=self.timeout_ms / 1000,
                network_timeout=self.timeout_ms / 1000
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create Triton client: {e}")

        # Verify server health
        self._verify_server()

    def _verify_server(self):
        """Verify Triton server is live and model is loaded"""
        try:
            if not self.client.is_server_live():
                raise RuntimeError("Triton server is not live")

            if not self.client.is_server_ready():
                raise RuntimeError("Triton server is not ready")

            # Check if model is loaded
            if not self.client.is_model_ready(self.model_name, self.model_version):
                raise RuntimeError(f"Model {self.model_name} (version {self.model_version}) is not loaded")

            print(f"✓ Connected to Triton server: {self.triton_url}")
            print(f"✓ Model '{self.model_name}' (v{self.model_version}) is ready")

        except self.InferenceServerException as e:
            raise RuntimeError(f"Triton server error: {e}")

    def preprocess(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess person crops for ReID model

        Args:
            crops: List of BGR person images (H, W, 3)

        Returns:
            Batched tensor [N, 3, H, W] normalized and CHW format
        """
        return preprocess_reid_crops(
            crops,
            self.input_shape,
            self.mean,
            self.std,
            self.color_space,
            self.channel_order,
        )

    def infer(self, crops: List[np.ndarray], retry=None, max_batch_size=None) -> np.ndarray:
        """
        Extract ReID embeddings from person crops

        Args:
            crops: List of person crop images (BGR format)
            retry: Number of retries on failure
            max_batch_size: Maximum crops per Triton request

        Returns:
            Embeddings array [N, embedding_dim]
        """
        if len(crops) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        request_batch_size = self.max_batch_size if max_batch_size is None else max_batch_size
        if request_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {request_batch_size}")

        chunks = [
            self._infer_batch(crops[start:start + request_batch_size], retry=retry)
            for start in range(0, len(crops), request_batch_size)
        ]
        return np.vstack(chunks)

    def _infer_batch(self, crops: List[np.ndarray], retry=None) -> np.ndarray:
        """Extract embeddings for one bounded Triton request."""
        retry = self.max_retry if retry is None else retry
        batch = self.preprocess(crops)
        batch_size = len(crops)

        # Create Triton input
        inputs = [
            self.httpclient.InferInput("input", batch.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(batch)

        # Create output request
        outputs = [
            self.httpclient.InferRequestedOutput("fc_pred")
        ]

        # Inference with retry
        for attempt in range(retry):
            try:
                response = self.client.infer(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    inputs=inputs,
                    outputs=outputs
                )

                # Extract embeddings
                embeddings = response.as_numpy("fc_pred")

                # Validate output shape
                expected_shape = (batch_size, self.embedding_dim)
                if embeddings.shape != expected_shape:
                    raise ValueError(
                        f"Unexpected output shape: {embeddings.shape}, expected {expected_shape}"
                    )

                return embeddings

            except self.InferenceServerException as e:
                if attempt < retry - 1:
                    print(f"WARNING: Triton inference failed (attempt {attempt + 1}/{retry}): {e}")
                    time.sleep(0.1)
                else:
                    raise RuntimeError(f"Triton inference failed after {retry} attempts: {e}")

            except Exception as e:
                raise RuntimeError(f"Unexpected error during inference: {e}")

    def get_model_metadata(self):
        """Get model metadata from Triton server"""
        try:
            metadata = self.client.get_model_metadata(
                model_name=self.model_name,
                model_version=self.model_version
            )
            return metadata
        except self.InferenceServerException as e:
            print(f"WARNING: Failed to get model metadata: {e}")
            return None

    def get_model_config(self):
        """Get model configuration from Triton server"""
        try:
            config = self.client.get_model_config(
                model_name=self.model_name,
                model_version=self.model_version
            )
            return config
        except self.InferenceServerException as e:
            print(f"WARNING: Failed to get model config: {e}")
            return None

    def close(self):
        """Close Triton client connection"""
        # HTTP client doesn't need explicit closing
        pass


def create_reid_client(config):
    """Create the configured ReID inference client.

    Supported backends:
      - triton / triton_http / onnxruntime_triton: TritonReIDClient
      - tensorrt_direct / tensorrt: in-process TensorRTReIDClient
      - onnxruntime / onnxruntime_direct / onnx: in-process ONNXRuntimeReIDClient
    """
    backend = str(config.get("backend", "triton")).lower()
    if backend in {"triton", "triton_http", "onnxruntime_triton"}:
        return TritonReIDClient(config)
    if backend in {"tensorrt", "tensorrt_direct", "direct_tensorrt"}:
        from .tensorrt_reid_client import TensorRTReIDClient

        return TensorRTReIDClient(config)
    if backend in {"onnx", "onnxruntime", "onnxruntime_direct", "direct_onnxruntime"}:
        from .onnx_reid_client import ONNXRuntimeReIDClient

        return ONNXRuntimeReIDClient(config)
    raise ValueError(f"Unsupported ReID backend: {backend}")


if __name__ == "__main__":
    # Test script
    import sys
    import yaml
    from pathlib import Path

    # Load config
    config_path = Path("configs/reid_config.yaml")
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create the configured client.
    try:
        client = create_reid_client(config)

        # Get metadata
        metadata = client.get_model_metadata()
        if metadata:
            print("\nModel metadata:")
            print(metadata)

        # Create dummy crops for testing
        dummy_crops = [np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8) for _ in range(4)]

        print(f"\nTesting inference with {len(dummy_crops)} dummy crops...")
        embeddings = client.infer(dummy_crops)

        print(f"  Output shape: {embeddings.shape}")
        print(f"  Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
        print("\nConfigured ReID client test passed!")
        client.close()

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
