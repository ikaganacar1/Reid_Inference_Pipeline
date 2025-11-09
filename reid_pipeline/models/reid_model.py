"""
Person Re-Identification Model with Batch Processing and TensorRT Support
Optimized for NVIDIA Jetson deployment - FIXED VERSION with TensorRT inference
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from typing import List, Optional, Tuple, Union
import logging
from pathlib import Path
import cv2


class ReIDModel(nn.Module):
    """
    Person Re-Identification backbone model.
    
    Architecture: ResNet50 + Embedding Layer
    Output: L2-normalized embeddings
    """
    
    def __init__(self, 
                 embedding_dim: int = 2048,
                 pretrained: bool = True,
                 dropout: float = 0.0):
        """
        Initialize ReID model.
        
        Args:
            embedding_dim: Dimensionality of output embeddings (512/2048)
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout probability
        """
        super(ReIDModel, self).__init__()
        
        # ResNet50 backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Embedding head
        self.embedding = nn.Linear(2048, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            L2-normalized embeddings (B, D)
        """
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Generate embeddings
        embeddings = self.embedding(features)
        embeddings = self.bn(embeddings)
        
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)
        
        # L2 normalization
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class BatchReIDExtractor:
    """
    Batch-optimized ReID feature extractor with TensorRT support.
    
    Features:
    - Automatic batching for maximum throughput
    - TensorRT model loading
    - Dynamic batch size adjustment
    - Preprocessing pipeline
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'cuda',
                 embedding_dim: int = 2048,
                 batch_size: int = 16,
                 image_size: Tuple[int, int] = (384, 192),  # FIXED: Match TensorRT engine dimensions
                 use_tensorrt: bool = False,
                 tensorrt_precision: str = 'fp16',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize batch ReID extractor.
        
        Args:
            model_path: Path to trained model (.pth or .engine for TensorRT)
            device: Device for inference ('cuda' or 'cpu')
            embedding_dim: Embedding dimensionality
            batch_size: Target batch size for inference
            image_size: (height, width) for person crops - FIXED to (384, 192) for TensorRT
            use_tensorrt: Use TensorRT engine
            tensorrt_precision: 'fp32', 'fp16', or 'int8'
        """
        self.device = torch.device(device)
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.image_size = image_size
        self.use_tensorrt = use_tensorrt
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize model - auto-detect TensorRT engine files
        if model_path and model_path.endswith('.engine'):
            self.logger.info(f"Detected TensorRT engine file: {model_path}")
            self.model = self._load_tensorrt_model(model_path)
            self.inference_mode = 'tensorrt'
            self.use_tensorrt = True  # Auto-enable TensorRT for .engine files
        elif model_path and model_path.endswith('.pth'):
            self.logger.info(f"Detected PyTorch model file: {model_path}")
            self.model = ReIDModel(embedding_dim=embedding_dim).to(self.device)
            self.model.eval()
            self._load_pytorch_model(model_path)
            self.inference_mode = 'pytorch'
        elif model_path:
            # Unknown format, try PyTorch
            self.logger.warning(f"Unknown model format: {model_path}, attempting PyTorch loading")
            self.model = ReIDModel(embedding_dim=embedding_dim).to(self.device)
            self.model.eval()
            self._load_pytorch_model(model_path)
            self.inference_mode = 'pytorch'
        else:
            # No model path - use random initialization
            self.logger.warning("No model path provided, using randomly initialized model")
            self.model = ReIDModel(embedding_dim=embedding_dim).to(self.device)
            self.model.eval()
            self.inference_mode = 'pytorch'
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Statistics
        self.stats = {
            'total_inferences': 0,
            'total_crops_processed': 0,
            'batch_sizes': []
        }
        
        # TensorRT stream for async execution
        if self.inference_mode == 'tensorrt':
            try:
                import pycuda.driver as cuda
                self.cuda_stream = cuda.Stream()
            except:
                self.cuda_stream = None
        
        self.logger.info(f"BatchReIDExtractor initialized: mode={self.inference_mode}, "
                        f"batch_size={batch_size}, image_size={image_size}")
    
    def _load_pytorch_model(self, model_path: str):
        """Load PyTorch model weights"""
        if Path(model_path).suffix != '.pth':
            raise ValueError(f"Unsupported model format: {model_path}")

        # Try loading with weights_only=True first (safer, doesn't require external dependencies)
        checkpoint = None

        # First, try with omegaconf safe globals (if available)
        try:
            self.logger.info("Attempting to load model with weights_only=True (with omegaconf safe globals)...")
            try:
                from omegaconf import ListConfig, DictConfig
                torch.serialization.add_safe_globals([ListConfig, DictConfig])
                self.logger.info("Added omegaconf types to safe globals")
            except ImportError:
                self.logger.info("omegaconf not available, proceeding without it")

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.logger.info("✅ Successfully loaded with weights_only=True")
        except Exception as e:
            self.logger.warning(f"weights_only=True failed: {str(e)[:200]}...")

            # Fall back to weights_only=False (allows nvidia_tao_core, etc.)
            try:
                self.logger.info("Attempting to load model with weights_only=False...")
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                self.logger.info("✅ Successfully loaded with weights_only=False")
            except ModuleNotFoundError as e:
                if 'nvidia_tao_core' in str(e):
                    error_msg = (
                        "Model requires nvidia_tao_core but it's not installed. "
                        "Options: 1) Install via 'pip install nvidia-tao-core', or "
                        "2) Use a model checkpoint that doesn't require TAO dependencies, or "
                        "3) Use TensorRT .engine model instead"
                    )
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
                elif 'omegaconf' in str(e):
                    error_msg = (
                        "Model requires omegaconf but it's not installed. "
                        "Install via 'pip install omegaconf'"
                    )
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
                else:
                    raise
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                raise RuntimeError(f"Failed to load ReID model from {model_path}: {e}") from e

        # Handle different checkpoint formats
        try:
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                elif 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)

            self.logger.info(f"✅ Loaded PyTorch model from {model_path}")
        except Exception as e:
            error_msg = f"Failed to load state dict from checkpoint: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _load_tensorrt_model(self, engine_path: str):
        """Load TensorRT engine"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda

            cuda.init()
            self.cuda_ctx = cuda.Device(0).make_context()  # ADDED: Store context

            # Load engine
            with open(engine_path, 'rb') as f:
                engine_data = f.read()

            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(engine_data)

            # Check if engine loaded successfully
            if engine is None:
                error_msg = (
                    f"Failed to deserialize TensorRT engine from {engine_path}. "
                    f"This usually means:\n"
                    f"1. The engine was built with a different TensorRT version\n"
                    f"2. The engine is corrupted\n"
                    f"3. The engine was built for a different GPU architecture\n\n"
                    f"Current TensorRT version: {trt.__version__}\n"
                    f"Solution: Rebuild the engine with the correct TensorRT version using:\n"
                    f"  docker compose exec worker python3 /app/rebuild_engines.py"
                )
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            context = engine.create_execution_context()
            
            # --- START OF FIX ---
            # FIXED: Use the new TensorRT API (>= 8.0) to get I/O tensor names.
            # The 'num_bindings', 'get_binding_name', and 'binding_is_input'
            # attributes are deprecated and were removed in newer versions.
            
            num_io_tensors = engine.num_io_tensors # Replaced num_bindings
            input_name = None
            output_name = None
            
            tensor_names = [engine.get_tensor_name(i) for i in range(num_io_tensors)]
            self.logger.info(f"Found {num_io_tensors} I/O tensors: {tensor_names}")

            for name in tensor_names:
                # Replaced binding_is_input(i)
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT: 
                    if input_name is None:
                        input_name = name
                        self.logger.info(f"Using TensorRT input tensor: {name}")
                    else:
                        self.logger.warning(f"Found multiple input tensors. Using first one: {input_name}")
                else: # It's an output
                    if output_name is None:
                        output_name = name
                        self.logger.info(f"Using TensorRT output tensor: {name}")
                    else:
                        self.logger.warning(f"Found multiple output tensors. Using first one: {output_name}")
            
            # --- END OF FIX ---

            # If names not found, use defaults (from original logic)
            if input_name is None:
                input_name = 'input'
                self.logger.warning(f"No input tensor found. Defaulting to: {input_name}")
            if output_name is None:
                output_name = 'fc_pred'  # FIXED: Use correct output name from error messages
                self.logger.warning(f"No output tensor found. Defaulting to: {output_name}")
                
            # CRITICAL FIX: Detect embedding dimension from engine output shape
            output_shape = engine.get_tensor_shape(output_name)
            detected_embedding_dim = output_shape[-1]  # Last dimension is embedding size

            if detected_embedding_dim != self.embedding_dim:
                self.logger.warning(
                    f"Embedding dimension mismatch! "
                    f"Engine outputs {detected_embedding_dim}D embeddings, "
                    f"but config specifies {self.embedding_dim}D. "
                    f"Using engine's dimension: {detected_embedding_dim}"
                )
                self.embedding_dim = detected_embedding_dim

            self.logger.info(f"Loaded TensorRT engine from {engine_path}")
            self.logger.info(f"Input name: {input_name}, Output name: {output_name}")
            self.logger.info(f"Output shape: {output_shape}, Embedding dimension: {self.embedding_dim}")

            return {
                'engine': engine,
                'context': context,
                'input_name': input_name,
                'output_name': output_name
            }
            
        except ImportError:
            self.logger.error("TensorRT not available. Install pycuda and tensorrt.")
            raise
        except Exception as e:
            self.logger.error(f"Error loading TensorRT engine: {e}")
            raise
    
    def preprocess_crops(self, 
                        image: np.ndarray,
                        bboxes: List[np.ndarray]) -> Tuple[torch.Tensor, List[bool]]:
        """
        Preprocess person crops for batch inference.
        
        Args:
            image: Full frame (H, W, 3)
            bboxes: List of bounding boxes [x1, y1, x2, y2]
            
        Returns:
            (preprocessed_batch, valid_flags)
        """
        batch_tensors = []
        valid_flags = []
        
        for bbox in bboxes:
            try:
                x1, y1, x2, y2 = bbox.astype(int)
                
                # Clamp to image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                # Crop person
                person_crop = image[y1:y2, x1:x2]
                
                # Validate crop
                if person_crop.size == 0 or person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
                    valid_flags.append(False)
                    continue
                
                # Add padding (10% as per requirements)
                h, w = person_crop.shape[:2]
                pad_h = int(h * 0.1)
                pad_w = int(w * 0.1)
                
                person_crop = cv2.copyMakeBorder(
                    person_crop,
                    pad_h, pad_h, pad_w, pad_w,
                    cv2.BORDER_CONSTANT,
                    value=(128, 128, 128)
                )
                
                # Transform
                person_tensor = self.transform(person_crop)
                batch_tensors.append(person_tensor)
                valid_flags.append(True)
                
            except Exception as e:
                self.logger.debug(f"Error preprocessing crop: {e}")
                valid_flags.append(False)
        
        if len(batch_tensors) == 0:
            return torch.zeros((0, 3, self.image_size[0], self.image_size[1])), valid_flags
        
        # Stack into batch
        batch = torch.stack(batch_tensors)
        
        return batch, valid_flags
    
    def extract_features_batch(self, 
                               batch: torch.Tensor,
                               to_numpy: bool = True) -> Union[torch.Tensor, np.ndarray]:
        """
        Extract features from batch of person crops.
        
        Args:
            batch: Preprocessed batch (N, 3, H, W)
            to_numpy: Convert to numpy array
            
        Returns:
            Feature embeddings (N, D)
        """
        if batch.size(0) == 0:
            return np.zeros((0, self.embedding_dim)) if to_numpy else torch.zeros((0, self.embedding_dim))
        
        batch = batch.to(self.device)
        
        # Process in sub-batches if necessary
        embeddings_list = []
        num_crops = batch.size(0)
        
        for i in range(0, num_crops, self.batch_size):
            end_idx = min(i + self.batch_size, num_crops)
            sub_batch = batch[i:end_idx]
            
            with torch.no_grad():
                if self.inference_mode == 'pytorch':
                    sub_embeddings = self.model(sub_batch)
                elif self.inference_mode == 'tensorrt':
                    sub_embeddings = self._tensorrt_inference(sub_batch)
                else:
                    raise ValueError(f"Unknown inference mode: {self.inference_mode}")
            
            embeddings_list.append(sub_embeddings)
        
        # Concatenate all embeddings
        embeddings = torch.cat(embeddings_list, dim=0)
        
        # Update statistics
        self.stats['total_inferences'] += 1
        self.stats['total_crops_processed'] += num_crops
        self.stats['batch_sizes'].append(num_crops)
        
        if to_numpy:
            return embeddings.cpu().numpy()
        return embeddings
    
    def _tensorrt_inference(self, batch: torch.Tensor) -> torch.Tensor:
        """Run inference using TensorRT engine - FIXED VERSION"""
        import pycuda.driver as cuda
        
        engine = self.model['engine']
        context = self.model['context']
        input_name = self.model['input_name']
        output_name = self.model['output_name']
        
        # FIX: Push context for worker thread
        if hasattr(self, 'cuda_ctx'):
            self.cuda_ctx.push()
        
        try:
            input_data = batch.cpu().numpy().astype(np.float32)
            batch_size_actual = input_data.shape[0]
            output_size = batch_size_actual * self.embedding_dim
            
            d_input = cuda.mem_alloc(input_data.nbytes)
            d_output = cuda.mem_alloc(output_size * np.dtype(np.float32).itemsize)
            output = np.empty((batch_size_actual, self.embedding_dim), dtype=np.float32)
            
            try:
                cuda.memcpy_htod_async(d_input, input_data, self.cuda_stream)
                
                # FIXED: Use correct tensor names
                context.set_input_shape(input_name, input_data.shape)
                context.set_tensor_address(input_name, int(d_input))
                context.set_tensor_address(output_name, int(d_output))  # FIXED: Use correct output name
                
                context.execute_async_v3(stream_handle=self.cuda_stream.handle)
                self.cuda_stream.synchronize()
                cuda.memcpy_dtoh(output, d_output)
            finally:
                d_input.free()
                d_output.free()
            
            embeddings = torch.from_numpy(output).to(self.device)
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings
            
        finally:
            # FIX: Pop context
            if hasattr(self, 'cuda_ctx'):
                self.cuda_ctx.pop()
                
    def cleanup(self):
        """Explicitly clean up CUDA context."""
        if hasattr(self, 'cuda_ctx'):
            try:
                self.logger.info("Detaching CUDA context...")
                self.cuda_ctx.detach()
                self.logger.info("CUDA context detached.")
            except Exception as e:
                self.logger.error(f"Error detaching CUDA context: {e}")
            
    def extract_features_from_frame(self,
                                   image: np.ndarray,
                                   bboxes: List[np.ndarray]) -> Tuple[np.ndarray, List[bool]]:
        """
        Extract features from person crops in a frame.
        
        Args:
            image: Full frame (H, W, 3)
            bboxes: List of bounding boxes
            
        Returns:
            (embeddings, valid_flags)
        """
        if len(bboxes) == 0:
            return np.zeros((0, self.embedding_dim)), []
        
        # Preprocess
        batch, valid_flags = self.preprocess_crops(image, bboxes)
        
        # Extract features
        if batch.size(0) > 0:
            embeddings = self.extract_features_batch(batch, to_numpy=True)
        else:
            embeddings = np.zeros((0, self.embedding_dim))
        
        return embeddings, valid_flags
    
    def get_statistics(self) -> dict:
        """Get extractor statistics"""
        if len(self.stats['batch_sizes']) == 0:
            avg_batch_size = 0
        else:
            avg_batch_size = np.mean(self.stats['batch_sizes'])
        
        return {
            'total_inferences': self.stats['total_inferences'],
            'total_crops_processed': self.stats['total_crops_processed'],
            'avg_batch_size': float(avg_batch_size),
            'inference_mode': self.inference_mode,
            'device': str(self.device)
        }
    
    def warmup(self, num_iterations: int = 10):
        """
        Warm up the model with dummy inputs.
        
        Important for TensorRT to allocate resources and optimize kernels.
        
        Args:
            num_iterations: Number of warm-up iterations
        """
        self.logger.info(f"Warming up model ({num_iterations} iterations)...")
        
        dummy_input = torch.randn(
            self.batch_size, 3, self.image_size[0], self.image_size[1]
        ).to(self.device)
        
        for i in range(num_iterations):
            with torch.no_grad():
                if self.inference_mode == 'pytorch':
                    _ = self.model(dummy_input)
                elif self.inference_mode == 'tensorrt':
                    _ = self._tensorrt_inference(dummy_input)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        self.logger.info("Warm-up complete")
    
    def export_to_onnx(self, output_path: str):
        """
        Export model to ONNX format for TensorRT conversion.
        
        Args:
            output_path: Path to save ONNX model
        """
        if self.inference_mode != 'pytorch':
            self.logger.error("Can only export PyTorch models to ONNX")
            return
        
        self.logger.info(f"Exporting model to ONNX: {output_path}")
        
        dummy_input = torch.randn(
            1, 3, self.image_size[0], self.image_size[1]
        ).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['fc_pred'],  # FIXED: Use correct output name
            dynamic_axes={
                'input': {0: 'batch_size'},
                'fc_pred': {0: 'batch_size'}  # FIXED: Use correct output name
            }
        )
        
        self.logger.info(f"Model exported to {output_path}")


if __name__ == "__main__":
    # Test the batch extractor
    logging.basicConfig(level=logging.INFO)
    
    # FIXED: Use correct dimensions
    extractor = BatchReIDExtractor(
        embedding_dim=2048,
        batch_size=16,
        image_size=(384, 192),  # FIXED: Match TensorRT engine dimensions
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Warm up
    extractor.warmup(num_iterations=5)
    
    # Test with dummy data
    dummy_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    dummy_bboxes = [
        np.array([100, 100, 300, 500]),
        np.array([400, 150, 600, 550]),
        np.array([700, 200, 900, 600])
    ]
    
    embeddings, valid_flags = extractor.extract_features_from_frame(
        dummy_image, dummy_bboxes
    )
    
    print(f"Extracted {embeddings.shape[0]} embeddings")
    print(f"Valid flags: {valid_flags}")
    print(f"Statistics: {extractor.get_statistics()}")