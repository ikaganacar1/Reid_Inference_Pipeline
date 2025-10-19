"""
Configuration Management for ReID Pipeline
Hardware-specific presets and YAML-based configuration
"""
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import logging


@dataclass
class DetectionConfig:
    """Object detection configuration"""
    model_path: str = 'yolo11n.pt'
    conf_threshold: float = 0.3
    iou_threshold: float = 0.55
    conf_high: float = 0.8
    conf_medium: float = 0.3
    use_tensorrt: bool = False


@dataclass
class ReIDConfig:
    """ReID model configuration"""
    model_path: Optional[str] = None
    embedding_dim: int = 2048
    batch_size: int = 16
    image_size_h: int = 256
    image_size_w: int = 128
    use_tensorrt: bool = False
    tensorrt_precision: str = 'fp16'


@dataclass
class GalleryConfig:
    """Gallery management configuration"""
    max_size: int = 500
    buffer_capacity: int = 10
    similarity_threshold_match: float = 0.70
    similarity_threshold_new: float = 0.50
    quality_admission_threshold: float = 0.91
    ttl_frames_crowded: int = 30
    ttl_frames_sparse: int = 150
    ema_alpha_high_conf: float = 0.3
    ema_alpha_low_conf: float = 0.1


@dataclass
class QueueConfig:
    """Queue sizes for async processing"""
    input_queue_size: int = 10
    processing_queue_size: int = 50
    output_queue_size: int = 20


@dataclass
class MonitoringConfig:
    """Monitoring and health check configuration"""
    enable_monitoring: bool = True
    metrics_update_interval: float = 1.0
    log_level: str = 'INFO'
    save_stats: bool = True
    stats_save_interval: int = 100


@dataclass
class DegradationConfig:
    """Graceful degradation configuration"""
    enable_degradation: bool = True
    gpu_threshold_tier1: float = 0.85
    gpu_threshold_tier2: float = 0.90
    memory_threshold_tier1: float = 0.85
    memory_threshold_tier2: float = 0.90
    queue_depth_threshold: int = 45


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    device: str = 'cuda'
    enable_display: bool = True
    detection: DetectionConfig = None
    reid: ReIDConfig = None
    gallery: GalleryConfig = None
    queues: QueueConfig = None
    monitoring: MonitoringConfig = None
    degradation: DegradationConfig = None
    
    def __post_init__(self):
        """Initialize nested configs if not provided"""
        if self.detection is None:
            self.detection = DetectionConfig()
        if self.reid is None:
            self.reid = ReIDConfig()
        if self.gallery is None:
            self.gallery = GalleryConfig()
        if self.queues is None:
            self.queues = QueueConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.degradation is None:
            self.degradation = DegradationConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'device': self.device,
            'enable_display': self.enable_display,
            'detection': asdict(self.detection),
            'reid': asdict(self.reid),
            'gallery': asdict(self.gallery),
            'queues': asdict(self.queues),
            'monitoring': asdict(self.monitoring),
            'degradation': asdict(self.degradation)
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create from dictionary"""
        detection = DetectionConfig(**config_dict.get('detection', {}))
        reid = ReIDConfig(**config_dict.get('reid', {}))
        gallery = GalleryConfig(**config_dict.get('gallery', {}))
        queues = QueueConfig(**config_dict.get('queues', {}))
        monitoring = MonitoringConfig(**config_dict.get('monitoring', {}))
        degradation = DegradationConfig(**config_dict.get('degradation', {}))
        
        return cls(
            device=config_dict.get('device', 'cuda'),
            enable_display=config_dict.get('enable_display', True),
            detection=detection,
            reid=reid,
            gallery=gallery,
            queues=queues,
            monitoring=monitoring,
            degradation=degradation
        )
    
    def save_yaml(self, filepath: Path):
        """Save configuration to YAML file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load_yaml(cls, filepath: Path) -> 'PipelineConfig':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)


class HardwarePresets:
    """Hardware-specific configuration presets for Jetson devices"""
    
    @staticmethod
    def xavier_nx() -> PipelineConfig:
        """
        Configuration for NVIDIA Jetson Xavier NX.
        
        Target: 4 streams @ 1080p, 20-35 FPS
        Optimization: FP16, moderate batching
        """
        config = PipelineConfig()
        
        # Detection
        config.detection.conf_threshold = 0.3
        config.detection.use_tensorrt = True
        
        # ReID
        config.reid.batch_size = 8
        config.reid.use_tensorrt = True
        config.reid.tensorrt_precision = 'fp16'
        
        # Gallery
        config.gallery.max_size = 300
        config.gallery.buffer_capacity = 5
        config.gallery.ttl_frames_crowded = 30
        config.gallery.ttl_frames_sparse = 100
        
        # Queues (conservative for memory)
        config.queues.input_queue_size = 5
        config.queues.processing_queue_size = 30
        config.queues.output_queue_size = 15
        
        # Degradation (aggressive for limited resources)
        config.degradation.gpu_threshold_tier1 = 0.80
        config.degradation.gpu_threshold_tier2 = 0.90
        
        return config
    
    @staticmethod
    def orin_nx() -> PipelineConfig:
        """
        Configuration for NVIDIA Jetson Orin NX.
        
        Target: 8-12 streams @ 1080p, 30-50 FPS
        Optimization: INT8, aggressive batching
        """
        config = PipelineConfig()
        
        # Detection
        config.detection.conf_threshold = 0.3
        config.detection.use_tensorrt = True
        
        # ReID
        config.reid.batch_size = 16
        config.reid.use_tensorrt = True
        config.reid.tensorrt_precision = 'int8'
        
        # Gallery
        config.gallery.max_size = 500
        config.gallery.buffer_capacity = 10
        config.gallery.ttl_frames_crowded = 30
        config.gallery.ttl_frames_sparse = 150
        
        # Queues (larger for better throughput)
        config.queues.input_queue_size = 10
        config.queues.processing_queue_size = 50
        config.queues.output_queue_size = 20
        
        # Degradation (standard)
        config.degradation.gpu_threshold_tier1 = 0.85
        config.degradation.gpu_threshold_tier2 = 0.90
        
        return config
    
    @staticmethod
    def agx_orin() -> PipelineConfig:
        """
        Configuration for NVIDIA Jetson AGX Orin.
        
        Target: 32+ streams @ 4K, 50+ FPS
        Optimization: INT8, maximum batching
        """
        config = PipelineConfig()
        
        # Detection
        config.detection.conf_threshold = 0.25
        config.detection.use_tensorrt = True
        
        # ReID
        config.reid.batch_size = 32
        config.reid.use_tensorrt = True
        config.reid.tensorrt_precision = 'int8'
        
        # Gallery
        config.gallery.max_size = 1000
        config.gallery.buffer_capacity = 10
        config.gallery.ttl_frames_crowded = 50
        config.gallery.ttl_frames_sparse = 200
        
        # Queues (large for maximum throughput)
        config.queues.input_queue_size = 20
        config.queues.processing_queue_size = 100
        config.queues.output_queue_size = 30
        
        # Degradation (relaxed for powerful hardware)
        config.degradation.gpu_threshold_tier1 = 0.90
        config.degradation.gpu_threshold_tier2 = 0.95
        
        return config
    
    @staticmethod
    def development() -> PipelineConfig:
        """
        Configuration for development/testing on desktop GPU.
        
        No TensorRT optimization, full logging
        """
        config = PipelineConfig()
        
        # Detection
        config.detection.conf_threshold = 0.3
        config.detection.use_tensorrt = False
        
        # ReID
        config.reid.batch_size = 16
        config.reid.use_tensorrt = False
        
        # Gallery
        config.gallery.max_size = 500
        config.gallery.buffer_capacity = 10
        
        # Queues
        config.queues.input_queue_size = 10
        config.queues.processing_queue_size = 50
        config.queues.output_queue_size = 20
        
        # Monitoring (verbose for debugging)
        config.monitoring.log_level = 'DEBUG'
        config.monitoring.save_stats = True
        
        return config


class ConfigManager:
    """Configuration manager with validation and presets"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.current_config: Optional[PipelineConfig] = None
    
    def load_config(self, source: str) -> PipelineConfig:
        """
        Load configuration from file or preset.
        
        Args:
            source: Either a file path or preset name 
                   ('xavier_nx', 'orin_nx', 'agx_orin', 'development')
        
        Returns:
            PipelineConfig object
        """
        # Check if it's a preset
        presets = {
            'xavier_nx': HardwarePresets.xavier_nx,
            'orin_nx': HardwarePresets.orin_nx,
            'agx_orin': HardwarePresets.agx_orin,
            'development': HardwarePresets.development
        }
        
        if source in presets:
            self.logger.info(f"Loading preset configuration: {source}")
            config = presets[source]()
        else:
            # Try to load from file
            config_path = Path(source)
            if config_path.exists():
                self.logger.info(f"Loading configuration from: {config_path}")
                config = PipelineConfig.load_yaml(config_path)
            else:
                raise ValueError(f"Configuration source not found: {source}")
        
        self.current_config = config
        self.validate_config(config)
        
        return config
    
    def validate_config(self, config: PipelineConfig) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        # Detection validation
        if not 0 < config.detection.conf_threshold < 1:
            errors.append("Detection conf_threshold must be between 0 and 1")
        
        # ReID validation
        if config.reid.batch_size < 1:
            errors.append("ReID batch_size must be >= 1")
        if config.reid.embedding_dim not in [512, 1024, 2048]:
            errors.append("ReID embedding_dim should be 512, 1024, or 2048")
        
        # Gallery validation
        if config.gallery.max_size < 1:
            errors.append("Gallery max_size must be >= 1")
        if not 0 < config.gallery.similarity_threshold_match < 1:
            errors.append("Gallery similarity_threshold_match must be between 0 and 1")
        
        # Queue validation
        if config.queues.input_queue_size < 1:
            errors.append("Queue sizes must be >= 1")
        
        if errors:
            for error in errors:
                self.logger.error(f"Configuration error: {error}")
            return False
        
        self.logger.info("Configuration validation passed")
        return True
    
    def save_config(self, config: PipelineConfig, filepath: Path):
        """Save configuration to YAML file"""
        config.save_yaml(filepath)
        self.logger.info(f"Configuration saved to: {filepath}")
    
    def get_config_summary(self, config: Optional[PipelineConfig] = None) -> str:
        """Get human-readable configuration summary"""
        if config is None:
            config = self.current_config
        
        if config is None:
            return "No configuration loaded"
        
        summary = [
            "Pipeline Configuration Summary:",
            "=" * 50,
            f"Device: {config.device}",
            f"Display: {config.enable_display}",
            "",
            "Detection:",
            f"  Model: {config.detection.model_path}",
            f"  Confidence: {config.detection.conf_threshold}",
            f"  TensorRT: {config.detection.use_tensorrt}",
            "",
            "ReID:",
            f"  Batch Size: {config.reid.batch_size}",
            f"  Embedding Dim: {config.reid.embedding_dim}",
            f"  TensorRT: {config.reid.use_tensorrt} ({config.reid.tensorrt_precision})",
            "",
            "Gallery:",
            f"  Max Size: {config.gallery.max_size}",
            f"  Match Threshold: {config.gallery.similarity_threshold_match}",
            f"  TTL: {config.gallery.ttl_frames_sparse} frames",
            "",
            "Queues:",
            f"  Input: {config.queues.input_queue_size}",
            f"  Processing: {config.queues.processing_queue_size}",
            f"  Output: {config.queues.output_queue_size}",
            "=" * 50
        ]
        
        return "\n".join(summary)


if __name__ == "__main__":
    # Test configuration management
    logging.basicConfig(level=logging.INFO)
    
    manager = ConfigManager()
    
    # Test presets
    print("\n" + "="*60)
    print("Testing Hardware Presets")
    print("="*60)
    
    for preset in ['xavier_nx', 'orin_nx', 'agx_orin', 'development']:
        config = manager.load_config(preset)
        print(f"\n{preset.upper()}:")
        print(manager.get_config_summary(config))
    
    # Save example config
    config = manager.load_config('xavier_nx')
    config.save_yaml(Path('config_xavier_nx.yaml'))
    print("\nExample configuration saved to: config_xavier_nx.yaml")
