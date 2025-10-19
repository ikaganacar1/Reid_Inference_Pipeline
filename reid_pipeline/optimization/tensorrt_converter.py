"""
TensorRT Optimization Utilities
Convert PyTorch/ONNX models to TensorRT for maximum performance on Jetson
"""
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging
import torch


class TensorRTConverter:
    """
    Convert models to TensorRT format with INT8 calibration support.
    
    Performance expectations on Xavier NX:
    - YOLOv8m: 65ms (FP32) → 30ms (FP16) → 8ms (INT8)
    - ReID: 50-100ms (FP32) → 10-15ms (FP16+batching+INT8)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Check TensorRT availability
        try:
            import tensorrt as trt
            self.trt = trt
            self.trt_available = True
            self.logger.info(f"TensorRT version: {trt.__version__}")
        except ImportError:
            self.logger.warning("TensorRT not available. Install python3-libnvinfer")
            self.trt_available = False
            self.trt = None
    
    def pytorch_to_onnx(self,
                       model: torch.nn.Module,
                       input_shape: Tuple[int, ...],
                       output_path: Path,
                       opset_version: int = 11,
                       dynamic_batch: bool = True) -> bool:
        """
        Convert PyTorch model to ONNX.
        
        Args:
            model: PyTorch model
            input_shape: Input shape (C, H, W) or (B, C, H, W)
            output_path: Output ONNX file path
            opset_version: ONNX opset version
            dynamic_batch: Support dynamic batch size
            
        Returns:
            True if successful
        """
        self.logger.info("Converting PyTorch model to ONNX...")
        
        try:
            # Ensure model is in eval mode
            model.eval()
            
            # Create dummy input
            if len(input_shape) == 3:
                dummy_input = torch.randn(1, *input_shape).cuda()
            else:
                dummy_input = torch.randn(*input_shape).cuda()
            
            # Set dynamic axes if requested
            dynamic_axes = None
            if dynamic_batch:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            
            # Export
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            self.logger.info(f"ONNX model saved to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error converting to ONNX: {e}", exc_info=True)
            return False
    
    def onnx_to_tensorrt(self,
                        onnx_path: Path,
                        engine_path: Path,
                        precision: str = 'fp16',
                        max_batch_size: int = 32,
                        workspace_size: int = 1 << 30,
                        calibration_cache: Optional[Path] = None,
                        calibration_data: Optional[List[np.ndarray]] = None) -> bool:
        """
        Convert ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Input ONNX file
            engine_path: Output TensorRT engine file
            precision: 'fp32', 'fp16', or 'int8'
            max_batch_size: Maximum batch size
            workspace_size: Maximum workspace size (bytes)
            calibration_cache: Path to INT8 calibration cache
            calibration_data: Calibration images for INT8 (500-1000 images)
            
        Returns:
            True if successful
        """
        if not self.trt_available:
            self.logger.error("TensorRT not available")
            return False
        
        self.logger.info(f"Converting ONNX to TensorRT ({precision})...")
        
        try:
            import tensorrt as trt
            
            # Create builder
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        self.logger.error(parser.get_error(error))
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = workspace_size
            
            # Set precision
            if precision == 'fp16':
                config.set_flag(trt.BuilderFlag.FP16)
                self.logger.info("FP16 mode enabled")
            elif precision == 'int8':
                config.set_flag(trt.BuilderFlag.INT8)
                self.logger.info("INT8 mode enabled")
                
                # INT8 requires calibration
                if calibration_data is not None:
                    calibrator = INT8Calibrator(
                        calibration_data,
                        cache_file=calibration_cache,
                        batch_size=1
                    )
                    config.int8_calibrator = calibrator
                elif calibration_cache and calibration_cache.exists():
                    self.logger.info(f"Using calibration cache: {calibration_cache}")
                else:
                    self.logger.error("INT8 requires calibration data or cache")
                    return False
            
            # Build engine
            self.logger.info("Building TensorRT engine (this may take several minutes)...")
            engine = builder.build_engine(network, config)
            
            if engine is None:
                self.logger.error("Failed to build engine")
                return False
            
            # Save engine
            engine_path.parent.mkdir(parents=True, exist_ok=True)
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            self.logger.info(f"TensorRT engine saved to: {engine_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error converting to TensorRT: {e}", exc_info=True)
            return False
    
    def convert_yolo_to_tensorrt(self,
                                 yolo_model_path: Path,
                                 output_dir: Path,
                                 precision: str = 'fp16') -> Optional[Path]:
        """
        Complete conversion pipeline for YOLO models.
        
        Args:
            yolo_model_path: Path to YOLO .pt file
            output_dir: Output directory
            precision: Target precision
            
        Returns:
            Path to TensorRT engine or None if failed
        """
        from ultralytics import YOLO
        
        self.logger.info("Converting YOLO model to TensorRT...")
        
        try:
            # Load YOLO model
            model = YOLO(yolo_model_path)
            
            # Export to ONNX first
            onnx_path = output_dir / 'yolo_model.onnx'
            model.export(format='onnx', simplify=True)
            
            # Move ONNX to output directory
            default_onnx = Path(str(yolo_model_path).replace('.pt', '.onnx'))
            if default_onnx.exists():
                import shutil
                shutil.move(str(default_onnx), str(onnx_path))
            
            # Convert to TensorRT
            engine_path = output_dir / f'yolo_model_{precision}.engine'
            success = self.onnx_to_tensorrt(
                onnx_path,
                engine_path,
                precision=precision,
                max_batch_size=8
            )
            
            if success:
                return engine_path
            return None
            
        except Exception as e:
            self.logger.error(f"Error in YOLO conversion: {e}", exc_info=True)
            return None
    
    def convert_reid_to_tensorrt(self,
                                 pytorch_model: torch.nn.Module,
                                 output_dir: Path,
                                 input_shape: Tuple[int, int, int] = (3, 256, 128),
                                 precision: str = 'fp16',
                                 calibration_images: Optional[List[np.ndarray]] = None) -> Optional[Path]:
        """
        Complete conversion pipeline for ReID models.
        
        Args:
            pytorch_model: PyTorch ReID model
            output_dir: Output directory
            input_shape: Input shape (C, H, W)
            precision: Target precision
            calibration_images: Images for INT8 calibration (if needed)
            
        Returns:
            Path to TensorRT engine or None if failed
        """
        self.logger.info("Converting ReID model to TensorRT...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: PyTorch to ONNX
            onnx_path = output_dir / 'reid_model.onnx'
            success = self.pytorch_to_onnx(
                pytorch_model,
                input_shape,
                onnx_path,
                dynamic_batch=True
            )
            
            if not success:
                return None
            
            # Step 2: ONNX to TensorRT
            engine_path = output_dir / f'reid_model_{precision}.engine'
            
            # Prepare calibration data if INT8
            calib_data = None
            calib_cache = None
            if precision == 'int8' and calibration_images:
                calib_data = calibration_images
                calib_cache = output_dir / 'reid_calibration.cache'
            
            success = self.onnx_to_tensorrt(
                onnx_path,
                engine_path,
                precision=precision,
                max_batch_size=32,
                calibration_cache=calib_cache,
                calibration_data=calib_data
            )
            
            if success:
                return engine_path
            return None
            
        except Exception as e:
            self.logger.error(f"Error in ReID conversion: {e}", exc_info=True)
            return None


class INT8Calibrator:
    """
    INT8 calibration for TensorRT.
    
    Requires 500-1000 representative images from deployment conditions.
    """
    
    def __init__(self,
                 calibration_data: List[np.ndarray],
                 cache_file: Optional[Path] = None,
                 batch_size: int = 1):
        """
        Initialize INT8 calibrator.
        
        Args:
            calibration_data: List of calibration images (numpy arrays)
            cache_file: Path to save/load calibration cache
            batch_size: Calibration batch size
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            self.calibration_data = calibration_data
            self.cache_file = cache_file
            self.batch_size = batch_size
            self.current_index = 0
            
            # Allocate device memory
            self.device_input = None
            
        except ImportError:
            raise ImportError("pycuda required for INT8 calibration")
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        """Get next calibration batch"""
        import pycuda.driver as cuda
        
        if self.current_index >= len(self.calibration_data):
            return None
        
        batch = self.calibration_data[self.current_index]
        self.current_index += 1
        
        # Copy to device
        if self.device_input is None:
            self.device_input = cuda.mem_alloc(batch.nbytes)
        
        cuda.memcpy_htod(self.device_input, batch)
        
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        """Read calibration cache"""
        if self.cache_file and self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """Write calibration cache"""
        if self.cache_file:
            with open(self.cache_file, 'wb') as f:
                f.write(cache)


def prepare_calibration_dataset(image_dir: Path,
                                num_images: int = 500,
                                input_shape: Tuple[int, int, int] = (3, 256, 128)) -> List[np.ndarray]:
    """
    Prepare calibration dataset from image directory.
    
    Args:
        image_dir: Directory containing calibration images
        num_images: Number of images to use (500-1000 recommended)
        input_shape: Target input shape (C, H, W)
        
    Returns:
        List of preprocessed calibration images
    """
    import cv2
    from torchvision import transforms
    
    calibration_data = []
    
    # Get image files
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    image_files = image_files[:num_images]
    
    # Preprocessing transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_shape[1], input_shape[2])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img)
            calibration_data.append(img_tensor.numpy())
    
    return calibration_data


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    converter = TensorRTConverter()
    
    # Example: Convert YOLO model
    # engine_path = converter.convert_yolo_to_tensorrt(
    #     Path('yolo11n.pt'),
    #     Path('models/tensorrt'),
    #     precision='fp16'
    # )
    
    print("TensorRT converter ready")
    print("Example usage:")
    print("  converter.convert_yolo_to_tensorrt('yolo11n.pt', 'output/', 'fp16')")
    print("  converter.convert_reid_to_tensorrt(model, 'output/', precision='int8')")
