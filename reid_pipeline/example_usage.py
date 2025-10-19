"""
Example Usage Script for Production ReID Pipeline
Demonstrates basic usage patterns
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.production_pipeline import ProductionReIDPipeline
from utils.config import ConfigManager


def example_basic_usage():
    """Example 1: Basic usage with default settings"""
    print("="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    pipeline = ProductionReIDPipeline(
        yolo_model_path='yolo11n.pt',
        reid_model_path=None,  # Will use randomly initialized model
        detection_conf=0.3,
        reid_threshold_match=0.70,
        enable_display=True
    )
    
    # Run on video
    pipeline.run(
        video_source='test_video.mp4',
        output_path='output_basic.mp4'
    )


def example_with_preset():
    """Example 2: Using hardware preset"""
    print("="*60)
    print("Example 2: Using Hardware Preset")
    print("="*60)
    
    # Load configuration preset
    config_manager = ConfigManager()
    config = config_manager.load_config('xavier_nx')
    
    # Print configuration
    print(config_manager.get_config_summary(config))
    
    # Create pipeline from config
    pipeline = ProductionReIDPipeline(
        yolo_model_path=config.detection.model_path,
        reid_model_path=config.reid.model_path,
        device=config.device,
        detection_conf=config.detection.conf_threshold,
        reid_threshold_match=config.gallery.similarity_threshold_match,
        gallery_max_size=config.gallery.max_size,
        reid_batch_size=config.reid.batch_size,
        enable_display=config.enable_display
    )
    
    # Run on webcam
    pipeline.run(
        video_source=0,  # Webcam
        output_path='output_webcam.mp4'
    )


def example_with_state_persistence():
    """Example 3: Gallery state persistence"""
    print("="*60)
    print("Example 3: State Persistence")
    print("="*60)
    
    # Create pipeline
    pipeline = ProductionReIDPipeline(
        yolo_model_path='yolo11n.pt',
        reid_model_path='reid_model.pth',
        gallery_max_size=500
    )
    
    # Load existing gallery if available
    state_file = Path('gallery_state.pkl')
    if state_file.exists():
        print("Loading existing gallery state...")
        pipeline.load_state(state_file)
    
    # Process video
    try:
        pipeline.run(
            video_source='video.mp4',
            output_path='output.mp4'
        )
    finally:
        # Save gallery state
        print("Saving gallery state...")
        pipeline.save_state(state_file)


def example_custom_configuration():
    """Example 4: Custom configuration"""
    print("="*60)
    print("Example 4: Custom Configuration")
    print("="*60)
    
    # Load and modify configuration
    config_manager = ConfigManager()
    config = config_manager.load_config('development')
    
    # Customize settings
    config.detection.conf_threshold = 0.35
    config.reid.batch_size = 32
    config.gallery.max_size = 300
    config.gallery.similarity_threshold_match = 0.75
    
    # Save custom configuration
    config.save_yaml(Path('my_custom_config.yaml'))
    
    # Create pipeline from custom config
    pipeline = ProductionReIDPipeline(
        yolo_model_path=config.detection.model_path,
        reid_model_path=config.reid.model_path,
        detection_conf=config.detection.conf_threshold,
        reid_threshold_match=config.gallery.similarity_threshold_match,
        gallery_max_size=config.gallery.max_size,
        reid_batch_size=config.reid.batch_size
    )
    
    pipeline.run(
        video_source='video.mp4',
        output_path='output_custom.mp4'
    )


def example_tensorrt_models():
    """Example 5: Using TensorRT models"""
    print("="*60)
    print("Example 5: TensorRT Accelerated Pipeline")
    print("="*60)
    
    # Assumes you've already converted models to TensorRT
    pipeline = ProductionReIDPipeline(
        yolo_model_path='tensorrt_models/yolo_fp16.engine',
        reid_model_path='tensorrt_models/reid_fp16.engine',
        detection_conf=0.3,
        reid_threshold_match=0.70,
        reid_batch_size=16  # Larger batch for better GPU utilization
    )
    
    pipeline.run(
        video_source='video.mp4',
        output_path='output_tensorrt.mp4'
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ReID Pipeline Examples')
    parser.add_argument('--example', type=int, default=1,
                       help='Example number to run (1-5)')
    
    args = parser.parse_args()
    
    examples = {
        1: example_basic_usage,
        2: example_with_preset,
        3: example_with_state_persistence,
        4: example_custom_configuration,
        5: example_tensorrt_models
    }
    
    if args.example in examples:
        examples[args.example]()
    else:
        print("Available examples:")
        print("1. Basic usage")
        print("2. Using hardware preset")
        print("3. State persistence")
        print("4. Custom configuration")
        print("5. TensorRT models")
