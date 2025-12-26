"""
ReID Pipeline Orchestration
Main pipeline integrating YOLO detection, TAO ReID (Triton), and BoxMOT tracking
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from .detector import YOLOPersonDetector
from .reid_client import TritonReIDClient
from .tracker import ReIDTracker
from .logger import ExperimentLogger
from .utils.visualization import Visualizer
from .utils.metrics import PerformanceMetrics


class ReIDPipeline:
    """End-to-end ReID pipeline"""

    def __init__(self, configs: dict, experiment_dir: Path):
        """
        Initialize pipeline with all components

        Args:
            configs: Configuration dictionaries
            experiment_dir: Directory for experiment logs
        """
        print("="*60)
        print("Initializing ReID Pipeline")
        print("="*60)

        self.configs = configs
        self.exp_dir = Path(experiment_dir)

        # Initialize components
        print("\n[1/5] Initializing YOLO detector...")
        self.detector = YOLOPersonDetector(configs['yolo'])

        print("\n[2/5] Initializing Triton ReID client...")
        self.reid_client = TritonReIDClient(configs['reid'])

        print("\n[3/5] Initializing BoxMOT tracker...")
        self.tracker = ReIDTracker(configs['tracker'])

        print("\n[4/5] Initializing logger...")
        self.logger = ExperimentLogger(experiment_dir, configs)

        # Log model versions
        yolo_path = Path(configs['yolo']['model']['path'])
        if yolo_path.exists():
            self.logger.log_model_version(yolo_path, "yolo")

        print("\n[5/5] Initializing visualizer and metrics...")
        self.visualizer = Visualizer()
        self.metrics = PerformanceMetrics(window_size=30)

        # Pipeline settings
        self.save_visualization = configs['pipeline'].get('io', {}).get('save_visualization', True)
        self.log_every_n_frames = configs['pipeline'].get('logging', {}).get('log_every_n_frames', 30)

        print("\n" + "="*60)
        print("Pipeline initialization completed!")
        print("="*60)

    def process_video(self, video_path: Path, output_path: Optional[Path] = None, max_frames: Optional[int] = None):
        """
        Process video through complete pipeline

        Args:
            video_path: Path to input video
            output_path: Optional path to save visualization
            max_frames: Optional limit on frames to process
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"\nProcessing video: {video_path}")

        # Log video metadata
        self.logger.log_video_metadata(video_path)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")

        # Setup output video writer
        out = None
        if output_path and self.save_visualization:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"  Output video: {output_path}")

        frame_idx = 0
        start_time = time.time()

        print(f"\nStarting processing...")
        print("-" * 60)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and frame_idx >= max_frames:
                    print(f"\nReached max frames limit: {max_frames}")
                    break

                # Stage 1: YOLO Detection
                det_start = time.time()
                detections, crops = self.detector.detect(frame)
                det_time = time.time() - det_start

                # Log detections
                if frame_idx % self.log_every_n_frames == 0 or len(detections) > 0:
                    self.logger.log_detections(frame_idx, detections, det_time * 1000)

                # Stage 2: ReID Embeddings
                reid_start = time.time()
                if len(crops) > 0:
                    embeddings = self.reid_client.infer(crops)
                    self.logger.log_embeddings(frame_idx, detections[:, :4], embeddings, (time.time() - reid_start) * 1000)
                else:
                    embeddings = np.array([])
                reid_time = time.time() - reid_start

                # Stage 3: Tracking
                track_start = time.time()
                tracks = self.tracker.update(detections, frame, embeddings)
                track_time = time.time() - track_start

                # Log tracks
                if frame_idx % self.log_every_n_frames == 0 or len(tracks) > 0:
                    self.logger.log_tracks(frame_idx, tracks)

                # Update metrics
                self.metrics.update(det_time, reid_time, track_time, len(detections), len(tracks))

                # Log performance
                if frame_idx % self.log_every_n_frames == 0:
                    gpu_mem = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else None
                    self.logger.log_performance(
                        frame_idx,
                        self.metrics.get_average_fps(),
                        gpu_mem,
                        (det_time + reid_time + track_time) * 1000
                    )

                # Visualization
                if out is not None:
                    # Draw tracks
                    self.visualizer.draw_tracks(frame, tracks)

                    # Draw stats
                    stats = {
                        "Frame": frame_idx,
                        "FPS": self.metrics.get_average_fps(),
                        "Detections": len(detections),
                        "Tracks": len(tracks)
                    }
                    self.visualizer.draw_stats(frame, stats, position="top-left")

                    out.write(frame)

                # Progress update
                if (frame_idx + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = self.metrics.get_average_fps()
                    print(f"Frame {frame_idx + 1}/{total_frames if not max_frames else max_frames} | "
                          f"FPS: {avg_fps:.2f} | "
                          f"Detections: {len(detections)} | "
                          f"Tracks: {len(tracks)} | "
                          f"Elapsed: {elapsed:.1f}s")

                frame_idx += 1

        except KeyboardInterrupt:
            print("\n\nProcessing interrupted by user")

        except Exception as e:
            print(f"\n\nERROR during processing: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup
            cap.release()
            if out is not None:
                out.release()

            total_time = time.time() - start_time

            print("\n" + "="*60)
            print("Processing completed!")
            print("="*60)

            # Print summary
            summary = self.metrics.get_summary()
            print(f"\nSummary:")
            print(f"  Total frames processed: {summary['total_frames']}")
            print(f"  Total detections: {summary['total_detections']}")
            print(f"  Total tracks: {summary['total_tracks']}")
            print(f"  Processing time: {total_time:.2f}s")
            print(f"  Overall FPS: {summary['overall_fps']:.2f}")
            print(f"  Average detection time: {summary['average_detection_time_ms']:.2f} ms")
            print(f"  Average ReID time: {summary['average_reid_time_ms']:.2f} ms")
            print(f"  Average tracking time: {summary['average_tracking_time_ms']:.2f} ms")

            # Close logger
            self.logger.close()

            print(f"\nExperiment logs saved to: {self.exp_dir}")
            if output_path:
                print(f"Visualization saved to: {output_path}")

            return self.exp_dir


if __name__ == "__main__":
    # Test pipeline
    import sys
    from utils.config_loader import load_all_configs

    # Load configs
    configs = load_all_configs()

    # Create experiment directory
    exp_dir = Path("experiments") / ExperimentLogger.create_experiment_id("test_pipeline")

    # Create pipeline
    try:
        pipeline = ReIDPipeline(configs, exp_dir)
        print("\nPipeline initialization test passed!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
