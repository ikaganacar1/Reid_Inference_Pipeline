"""
Triton ReID Client
HTTP client wrapper for TAO ReID model served by Triton Inference Server
"""

import time
from typing import List

import cv2
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


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
        self.input_shape = config['model']['input_shape']  # [H, W]
        self.embedding_dim = config['model']['embedding_dim']  # Embedding dimension

        # Create Triton client
        try:
            self.client = httpclient.InferenceServerClient(
                url=self.triton_url,
                verbose=False
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

        except InferenceServerException as e:
            raise RuntimeError(f"Triton server error: {e}")

    def preprocess(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess person crops for ReID model

        Args:
            crops: List of BGR person images (H, W, 3)

        Returns:
            Batched tensor [N, 3, H, W] normalized and CHW format
        """
        batch = []
        H, W = self.input_shape  # 384, 192

        for crop in crops:
            # Resize to expected input size (H x W)
            img = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0

            # Apply ImageNet normalization
            img = (img - self.mean) / self.std

            # HWC to CHW
            img = np.transpose(img, (2, 0, 1))

            batch.append(img)

        return np.array(batch, dtype=np.float32)

    def infer(self, crops: List[np.ndarray], retry=3) -> np.ndarray:
        """
        Extract ReID embeddings from person crops

        Args:
            crops: List of person crop images (BGR format)
            retry: Number of retries on failure

        Returns:
            Embeddings array [N, 256]
        """
        if len(crops) == 0:
            return np.array([])

        # Preprocess
        batch = self.preprocess(crops)
        batch_size = len(crops)

        # Create Triton input
        inputs = [
            httpclient.InferInput("input", batch.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(batch)

        # Create output request
        outputs = [
            httpclient.InferRequestedOutput("fc_pred")
        ]

        # Inference with retry
        for attempt in range(retry):
            try:
                start_time = time.time()

                response = self.client.infer(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    inputs=inputs,
                    outputs=outputs
                )

                inference_time = time.time() - start_time

                # Extract embeddings
                embeddings = response.as_numpy("fc_pred")

                # Validate output shape
                expected_shape = (batch_size, self.embedding_dim)
                if embeddings.shape != expected_shape:
                    raise ValueError(
                        f"Unexpected output shape: {embeddings.shape}, expected {expected_shape}"
                    )

                return embeddings

            except InferenceServerException as e:
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
        except InferenceServerException as e:
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
        except InferenceServerException as e:
            print(f"WARNING: Failed to get model config: {e}")
            return None

    def close(self):
        """Close Triton client connection"""
        # HTTP client doesn't need explicit closing
        pass


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

    # Create client
    try:
        client = TritonReIDClient(config)

        # Get metadata
        metadata = client.get_model_metadata()
        if metadata:
            print(f"\nModel metadata:")
            print(f"  Name: {metadata.get('name')}")
            print(f"  Version: {metadata.get('versions')}")

        # Create dummy crops for testing
        dummy_crops = [np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8) for _ in range(4)]

        print(f"\nTesting inference with {len(dummy_crops)} dummy crops...")
        embeddings = client.infer(dummy_crops)

        print(f"  Output shape: {embeddings.shape}")
        print(f"  Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
        print(f"\nTriton ReID client test passed!")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
