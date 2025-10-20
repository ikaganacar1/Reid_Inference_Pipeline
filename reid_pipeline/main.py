"""
Main Entry Point for Production ReID Pipeline
Command-line interface for running the pipeline with various configurations
"""
import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from reid_pipeline.pipeline.production_pipeline import ProductionReIDPipeline
from reid_pipeline.utils.config import ConfigManager, PipelineConfig
from reid_pipeline.optimization.tensorrt_converter import TensorRTConverter


def setup_logging(log_level: str = 'INFO', log_file: Optional[Path] = None):
    """Setup logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure logging
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger('ReIDPipeline')


def run_pipeline(args):
    """Run the ReID pipeline"""
    logger = setup_logging(args.log_level, args.log_file)
    
    logger.info("="*70)
    logger.info("Production Person Re-Identification Pipeline")
    logger.info("="*70)
    
    # Load configuration
    config_manager = ConfigManager(logger=logger)
    
    if args.config:
        config = config_manager.load_config(args.config)
    else:
        # Use default or preset
        preset = args.preset or 'development'
        config = config_manager.load_config(preset)
    
    # Override config with command line arguments
    if args.device:
        config.device = args.device
    if args.no_display:
        config.enable_display = False
    if args.yolo_model:
        config.detection.model_path = args.yolo_model
    if args.reid_model:
        config.reid.model_path = args.reid_model
    
    # Print configuration
    logger.info("\n" + config_manager.get_config_summary(config))
    
    # Create pipeline
    pipeline = ProductionReIDPipeline(
        yolo_model_path=config.detection.model_path,
        reid_model_path=config.reid.model_path,
        device=config.device,
        detection_conf=config.detection.conf_threshold,
        reid_threshold_match=config.gallery.similarity_threshold_match,
        reid_threshold_new=config.gallery.similarity_threshold_new,
        gallery_max_size=config.gallery.max_size,
        reid_batch_size=config.reid.batch_size,
        use_tensorrt=config.reid.use_tensorrt,
        tensorrt_precision=config.reid.tensorrt_precision,
        queue_size_input=config.queues.input_queue_size,
        queue_size_processing=config.queues.processing_queue_size,
        queue_size_output=config.queues.output_queue_size,
        enable_display=config.enable_display,
        logger=logger
    )
    
    # Load gallery state if specified
    if args.load_state:
        state_path = Path(args.load_state)
        if state_path.exists():
            pipeline.load_state(state_path)
        else:
            logger.warning(f"State file not found: {state_path}")
    
    # Run pipeline
    try:
        pipeline.run(
            video_source=args.input,
            output_path=args.output
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return 1
    finally:
        # Save state if specified
        if args.save_state:
            pipeline.save_state(Path(args.save_state))
    
    return 0


def convert_models(args):
    """Convert models to TensorRT"""
    logger = setup_logging(args.log_level)
    
    logger.info("="*70)
    logger.info("TensorRT Model Conversion")
    logger.info("="*70)
    
    converter = TensorRTConverter(logger=logger)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert YOLO model
    if args.yolo_model:
        logger.info(f"Converting YOLO model: {args.yolo_model}")
        engine_path = converter.convert_yolo_to_tensorrt(
            Path(args.yolo_model),
            output_dir,
            precision=args.precision
        )
        if engine_path:
            logger.info(f"YOLO TensorRT engine: {engine_path}")
        else:
            logger.error("YOLO conversion failed")
            return 1
    
    # Convert ReID model
    if args.reid_model:
        logger.info(f"Converting ReID model: {args.reid_model}")
        
        # Load PyTorch model
        import torch
        from models.reid_model import ReIDModel
        
        model = ReIDModel(embedding_dim=args.embedding_dim).cuda()
        
        try:
            checkpoint = torch.load(args.reid_model)
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            logger.error(f"Error loading ReID model: {e}")
            return 1
        
        model.eval()
        
        # Convert
        engine_path = converter.convert_reid_to_tensorrt(
            model,
            output_dir,
            input_shape=(3, 256, 128),
            precision=args.precision
        )
        
        if engine_path:
            logger.info(f"ReID TensorRT engine: {engine_path}")
        else:
            logger.error("ReID conversion failed")
            return 1
    
    logger.info("Conversion complete!")
    return 0


def generate_config(args):
    """Generate configuration file"""
    logger = setup_logging('INFO')
    
    config_manager = ConfigManager(logger=logger)
    
    # Load preset
    config = config_manager.load_config(args.preset)
    
    # Save to file
    output_path = Path(args.output)
    config_manager.save_config(config, output_path)
    
    logger.info(f"Configuration saved to: {output_path}")
    logger.info("\n" + config_manager.get_config_summary(config))
    
    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Production Person Re-Identification Pipeline for NVIDIA Jetson',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with preset configuration
  python main.py run --preset xavier_nx --input video.mp4 --output result.mp4
  
  # Run with custom configuration
  python main.py run --config config.yaml --input 0 --output webcam.mp4
  
  # Convert models to TensorRT
  python main.py convert --yolo-model yolo11n.pt --precision fp16
  
  # Generate configuration file
  python main.py config --preset orin_nx --output my_config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run pipeline command
    run_parser = subparsers.add_parser('run', help='Run ReID pipeline')
    run_parser.add_argument('--input', '-i', required=True,
                           help='Input video file or camera index (e.g., 0 for webcam)')
    run_parser.add_argument('--output', '-o',
                           help='Output video file path (optional)')
    run_parser.add_argument('--config', '-c',
                           help='Configuration file path')
    run_parser.add_argument('--preset', '-p',
                           choices=['xavier_nx', 'orin_nx', 'agx_orin', 'development'],
                           help='Hardware preset configuration')
    run_parser.add_argument('--device',
                           help='Device for inference (cuda, cpu, or auto)')
    run_parser.add_argument('--yolo-model',
                           help='Path to YOLO model')
    run_parser.add_argument('--reid-model',
                           help='Path to ReID model')
    run_parser.add_argument('--load-state',
                           help='Load gallery state from file')
    run_parser.add_argument('--save-state',
                           help='Save gallery state to file')
    run_parser.add_argument('--no-display', action='store_true',
                           help='Disable display window')
    run_parser.add_argument('--log-level',
                           default='INFO',
                           choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                           help='Logging level')
    run_parser.add_argument('--log-file',
                           type=Path,
                           help='Log file path')
    
    # Convert models command
    convert_parser = subparsers.add_parser('convert', help='Convert models to TensorRT')
    convert_parser.add_argument('--yolo-model',
                               help='Path to YOLO .pt model')
    convert_parser.add_argument('--reid-model',
                               help='Path to ReID .pth model')
    convert_parser.add_argument('--output-dir', '-o',
                               default='./tensorrt_models',
                               help='Output directory for TensorRT engines')
    convert_parser.add_argument('--precision',
                               choices=['fp32', 'fp16', 'int8'],
                               default='fp16',
                               help='TensorRT precision mode')
    convert_parser.add_argument('--embedding-dim',
                               type=int,
                               default=2048,
                               help='ReID embedding dimension')
    convert_parser.add_argument('--log-level',
                               default='INFO',
                               choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                               help='Logging level')
    
    # Generate config command
    config_parser = subparsers.add_parser('config', help='Generate configuration file')
    config_parser.add_argument('--preset', '-p',
                              required=True,
                              choices=['xavier_nx', 'orin_nx', 'agx_orin', 'development'],
                              help='Hardware preset')
    config_parser.add_argument('--output', '-o',
                              required=True,
                              help='Output configuration file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'run':
        return run_pipeline(args)
    elif args.command == 'convert':
        return convert_models(args)
    elif args.command == 'config':
        return generate_config(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
