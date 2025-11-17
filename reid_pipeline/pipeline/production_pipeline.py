"""
Production ReID Pipeline with Asynchronous Processing
Optimized for NVIDIA Jetson deployment with multi-threading and queue management
"""
import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from datetime import datetime
import logging
import threading
import queue
from dataclasses import dataclass
from collections import deque

from reid_pipeline.models import EnhancedObjectDetector, Detection, BatchReIDExtractor
from reid_pipeline.gallery import GalleryManager, MatchDecision


@dataclass
class FramePacket:
    """Data packet for frame processing"""
    frame_id: int
    frame: np.ndarray
    timestamp: float


@dataclass
class DetectionPacket:
    """Data packet for detection results"""
    frame_id: int
    frame: np.ndarray
    detections: List[Detection]
    timestamp: float


class ProductionReIDPipeline:
    """
    Production-grade Person Re-Identification Pipeline.
    
    Architecture:
    Thread 1: Video capture → Input Queue
    Thread 2: Detection → Detection Queue
    Thread 3: ReID + Tracking → Output Queue
    Thread 4: Display/Save
    
    Features:
    - Asynchronous multi-threaded processing
    - Queue-based buffering with backpressure
    - Graceful degradation under load
    - Comprehensive monitoring
    - State persistence
    """
    
    def __init__(self,
                 yolo_model_path: str = 'yolo11n.pt',
                 reid_model_path: Optional[str] = None,
                 device: str = 'auto',
                 detection_conf: float = 0.3,
                 reid_threshold_match: float = 0.70,
                 reid_threshold_new: float = 0.50,
                 gallery_max_size: int = 500,
                 reid_batch_size: int = 16,
                 queue_size_input: int = 30,
                 queue_size_processing: int = 100,
                 queue_size_output: int = 50,
                 enable_display: bool = True,
                 use_tensorrt: bool = False,
                 tensorrt_precision: str = 'fp16',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize production pipeline.
        
        Args:
            yolo_model_path: Path to YOLO model
            reid_model_path: Path to ReID model
            device: Device for inference
            detection_conf: Detection confidence threshold
            reid_threshold_match: ReID match threshold
            reid_threshold_new: ReID new person threshold
            gallery_max_size: Maximum gallery size
            reid_batch_size: Batch size for ReID
            queue_size_input: Input queue size
            queue_size_processing: Processing queue size
            queue_size_output: Output queue size
            enable_display: Enable display window
        """
        self.logger = logger or self._setup_logger()
        self.enable_display = enable_display
        
        self.logger.info("="*60)
        self.logger.info("Initializing Production ReID Pipeline")
        self.logger.info("="*60)
        
        # Initialize detector
        self.detector = EnhancedObjectDetector(
            model_path=yolo_model_path,
            device=device,
            conf_threshold=detection_conf,
            iou_threshold=0.55,
            logger=self.logger
        )
        
        # Initialize ReID extractor
        self.reid_extractor = BatchReIDExtractor(
            model_path=reid_model_path,
            device=device,
            batch_size=reid_batch_size,
            use_tensorrt=use_tensorrt,
            tensorrt_precision=tensorrt_precision,
            logger=self.logger
        )

        # Initialize gallery manager with correct embedding dimension from ReID model
        self.gallery_manager = GalleryManager(
            max_gallery_size=gallery_max_size,
            embedding_dim=self.reid_extractor.embedding_dim,  # Use actual embedding dimension
            similarity_threshold_match=reid_threshold_match,
            similarity_threshold_new=reid_threshold_new,
            logger=self.logger
        )
        
        # Queues for async processing
        self.input_queue = queue.Queue(maxsize=queue_size_input)
        self.detection_queue = queue.Queue(maxsize=queue_size_processing)
        self.output_queue = queue.Queue(maxsize=queue_size_output)
        
        # Threading control
        self.running = False
        self.threads = []
        
        # Pipeline statistics
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'frames_displayed': 0,
            'total_detections': 0,
            'total_persons_tracked': 0,
            'processing_times': deque(maxlen=100),
            'fps_history': deque(maxlen=30),
            'queue_sizes': {
                'input': deque(maxlen=100),
                'detection': deque(maxlen=100),
                'output': deque(maxlen=100)
            }
        }
        
        # Timing
        self.start_time = None
        self.frame_count = 0
        
        self.logger.info("Pipeline initialization complete")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logger"""
        logger = logging.getLogger('ReIDPipeline')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        return logger
    
    def _capture_thread(self, video_source):
        """Thread 1: Video capture"""
        self.logger.info("Capture thread started")
        
        # Try to convert video_source to int for camera, else it's a file
        is_camera = False
        try:
            video_source_int = int(video_source)
            cap = cv2.VideoCapture(video_source_int)
            is_camera = True
            self.logger.info(f"Opening camera source: {video_source_int}")
        except ValueError:
            cap = cv2.VideoCapture(str(video_source)) # Ensure it's a string
            self.logger.info(f"Opening video file: {video_source}")

        if not cap.isOpened():
            self.logger.error(f"Failed to open video source: {video_source}")
            self.running = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or is_camera:
            # Use 30 for cameras or if FPS is not available/unreliable
            fps = 30.0
            if not is_camera:
                 self.logger.warning(f"Video FPS not found, defaulting to {fps} FPS.")
        
        delay_between_frames = 1.0 / fps
        self.logger.info(f"Limiting capture to {fps:.2f} FPS (Delay: {delay_between_frames*1000:.2f}ms)")

        frame_id = 0
        
        try:
            while self.running:
                frame_read_start = time.time()

                ret, frame = cap.read()
                
                if not ret:
                    self.logger.info("End of video stream")
                    break
                
                # Create frame packet
                packet = FramePacket(
                    frame_id=frame_id,
                    frame=frame.copy(),
                    timestamp=time.time()
                )
                
                try:
                    # Try non-blocking first
                    self.input_queue.put(packet, block=False)
                    self.stats['frames_captured'] += 1
                    frame_id += 1
                except queue.Full:
                    # Log warning only every 30 frames to reduce spam
                    if frame_id % 30 == 0:
                        self.logger.warning(f"Input queue full at frame {frame_id}, pipeline is lagging")

                    # Block briefly (up to 0.5s) instead of dropping immediately
                    try:
                        self.input_queue.put(packet, block=True, timeout=0.5)
                        self.stats['frames_captured'] += 1
                        frame_id += 1
                    except queue.Full:
                        # Only drop if still full after waiting
                        if frame_id % 30 == 0:
                            self.logger.warning(f"Dropped frame {frame_id} after timeout")
                
                # Track queue size
                self.stats['queue_sizes']['input'].append(self.input_queue.qsize())

                if not is_camera: 
                    elapsed = time.time() - frame_read_start
                    sleep_time = delay_between_frames - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        
        finally:
            cap.release()
            self.logger.info("Capture thread stopped")
            self.logger.info("Capture thread sending shutdown signal...")
            self.input_queue.put(None) # Signal for next thread
    
    def _detection_thread(self):
        """Thread 2: Object detection"""
        self.logger.info("Detection thread started")
        
        try:
            while self.running:
                try:
                    # Get frame from input queue
                    packet = self.input_queue.get(timeout=1.0)
                    
                    if packet is None: # Check for sentinel
                        self.logger.info("Detection thread received shutdown signal.")
                        self.detection_queue.put(None) # Pass sentinel to next thread
                        self.running = False # Help signal other threads
                        break # Exit loop
                    
                except queue.Empty:
                    continue
                
                # Detect persons
                start_time = time.time()
                detections = self.detector.detect_persons(packet.frame)
                detection_time = time.time() - start_time
                
                # Create detection packet
                det_packet = DetectionPacket(
                    frame_id=packet.frame_id,
                    frame=packet.frame,
                    detections=detections,
                    timestamp=time.time()
                )
                
                # Put in detection queue
                try:
                    self.detection_queue.put(det_packet, timeout=0.1)
                    self.stats['total_detections'] += len(detections)
                except queue.Full:
                    self.logger.warning("Detection queue full, dropping packet")
                
                # Track queue size
                self.stats['queue_sizes']['detection'].append(self.detection_queue.qsize())
        
        except Exception as e:
            self.logger.error(f"Error in detection thread: {e}", exc_info=True)
        finally:
            self.logger.info("Detection thread stopped")
    
    def _reid_tracking_thread(self):
        """Thread 3: ReID and tracking"""
        self.logger.info("ReID tracking thread started")
        
        try:
            while self.running:
                try:
                    # Get detection packet
                    packet = self.detection_queue.get(timeout=1.0)
                    
                    if packet is None: # Check for sentinel
                        self.logger.info("ReID tracking thread received shutdown signal.")
                        self.output_queue.put(None) # Pass sentinel to next thread
                        self.running = False # Help signal other threads
                        break # Exit loop
                    
                except queue.Empty:
                    continue
                
                start_time = time.time()
                
                # Extract ReID features
                if len(packet.detections) > 0:
                    bboxes = [det.bbox for det in packet.detections]
                    embeddings, valid_flags = self.reid_extractor.extract_features_from_frame(
                        packet.frame, bboxes
                    )
                    
                    # Filter valid detections
                    valid_detections = [det for det, valid in zip(packet.detections, valid_flags) if valid]
                    valid_embeddings = embeddings
                    
                    if len(valid_detections) > 0:
                        # Match to gallery
                        confidences = np.array([det.confidence for det in valid_detections])
                        matches = self.gallery_manager.match_queries_to_gallery(
                            valid_embeddings, confidences
                        )
                        
                        # Assign IDs
                        for i, (det, (person_id, decision, similarity)) in enumerate(zip(valid_detections, matches)):
                            if decision == MatchDecision.MATCH:
                                # Existing person
                                det.track_id = person_id
                                self.gallery_manager.update_gallery_entry(
                                    person_id,
                                    valid_embeddings[i],
                                    det.confidence,
                                    packet.frame_id,
                                    det.bbox
                                )
                            elif decision == MatchDecision.UNCERTAIN:
                                # Tentative match (could implement temporal analysis here)
                                det.track_id = person_id
                                self.gallery_manager.update_gallery_entry(
                                    person_id,
                                    valid_embeddings[i],
                                    det.confidence,
                                    packet.frame_id,
                                    det.bbox
                                )
                            else:  # NEW
                                # New person
                                new_id = self.gallery_manager.next_person_id
                                det.track_id = new_id
                                
                                success = self.gallery_manager.add_to_gallery(
                                    new_id,
                                    valid_embeddings[i],
                                    det.confidence,
                                    packet.frame_id,
                                    det.bbox
                                )
                                
                                if success:
                                    self.stats['total_persons_tracked'] += 1
                    
                    packet.detections = valid_detections
                
                # Update gallery
                self.gallery_manager.update_frame()
                
                processing_time = time.time() - start_time
                self.stats['processing_times'].append(processing_time)
                
                # Put in output queue with backpressure
                try:
                    self.output_queue.put(packet, block=False)
                    self.stats['frames_processed'] += 1
                except queue.Full:
                    # Block briefly (up to 0.5s) instead of dropping
                    try:
                        self.output_queue.put(packet, block=True, timeout=0.5)
                        self.stats['frames_processed'] += 1
                    except queue.Full:
                        # Only log occasionally to reduce spam
                        if self.stats['frames_processed'] % 30 == 0:
                            self.logger.warning(f"Output queue full, dropped packet at frame {self.stats['frames_processed']}")
                
                # Track queue size
                self.stats['queue_sizes']['output'].append(self.output_queue.qsize())
        
        except Exception as e:
            self.logger.error(f"Error in ReID tracking thread: {e}", exc_info=True)
        finally:
            self.logger.info("ReID tracking thread stopped")
    
    def _display_thread(self, output_path: Optional[str] = None):
        """Thread 4: Display and save"""
        self.logger.info("Display thread started")
        
        # Setup video writer
        out = None
        if output_path:
            # Will be initialized on first frame
            # Use mp4v codec (OpenCV default that always works)
            # We'll convert to H.264 using FFmpeg after pipeline completes
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        try:
            while self.running:
                try:
                    # Get processed packet
                    packet = self.output_queue.get(timeout=1.0)
                    
                    if packet is None: # Check for sentinel
                        self.logger.info("Display thread received shutdown signal.")
                        self.running = False # Signal other threads
                        break # Exit loop
                    
                except queue.Empty:
                    continue
                
                # Draw detections
                annotated_frame = self._draw_results(packet.frame, packet.detections)
                
                # Add statistics overlay
                self._draw_stats_overlay(annotated_frame)
                
                # Save frame
                if output_path:
                    if out is None:
                        h, w = annotated_frame.shape[:2]
                        out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
                    
                    if out:
                        out.write(annotated_frame)
                
                # Display
                if self.enable_display:
                    cv2.imshow('Production ReID Pipeline', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        self.logger.info("User requested stop")
                        self.running = False
                    elif key == ord('s'):
                        # Save gallery
                        self.save_state()
                        self.logger.info("State saved")
                
                self.stats['frames_displayed'] += 1
                
                # Calculate FPS
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    if elapsed > 0:
                        fps = self.stats['frames_displayed'] / elapsed
                        self.stats['fps_history'].append(fps)
        
        except Exception as e:
            self.logger.error(f"Error in display thread: {e}", exc_info=True)
        finally:
            if out:
                out.release()
            if self.enable_display:
                cv2.destroyAllWindows()
            self.logger.info("Display thread stopped")
    
    def _draw_results(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw tracking results on frame"""
        annotated = frame.copy()
        
        for det in detections:
            if det.track_id is not None:
                x1, y1, x2, y2 = det.bbox.astype(int)
                
                # Color by ID
                color = self._get_color_for_id(det.track_id)
                
                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"Person {det.track_id}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                cv2.rectangle(annotated,
                            (x1, y1 - label_size[1] - 4),
                            (x1 + label_size[0], y1),
                            color, -1)
                
                cv2.putText(annotated, label,
                          (x1, y1 - 2),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6, (255, 255, 255), 2)
        
        return annotated
    
    def _get_color_for_id(self, track_id: int) -> tuple:
        """Get consistent color for track ID"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        return colors[track_id % len(colors)]
    
    def _draw_stats_overlay(self, frame: np.ndarray):
        """Draw statistics overlay"""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Get statistics
        gallery_stats = self.gallery_manager.get_statistics()
        avg_fps = np.mean(self.stats['fps_history']) if self.stats['fps_history'] else 0
        avg_proc_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        # Draw text
        y_offset = 35
        texts = [
            f"FPS: {avg_fps:.1f}",
            f"Processing Time: {avg_proc_time*1000:.1f}ms",
            f"Detections: {len(self.detector.detect_persons(frame))}",
            f"Gallery Size: {gallery_stats['gallery_size']}/{gallery_stats['max_size']}",
            f"Total Persons: {self.stats['total_persons_tracked']}",
            f"Queue: I:{self.input_queue.qsize()} D:{self.detection_queue.qsize()} O:{self.output_queue.qsize()}",
            f"Frames: {self.stats['frames_processed']}"
        ]
        
        for text in texts:
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 20
    
    def _convert_to_h264(self, input_path: str) -> bool:
        """
        Convert video to H.264 codec using FFmpeg for browser compatibility.

        Args:
            input_path: Path to input video with mp4v codec

        Returns:
            True if conversion successful, False otherwise
        """
        import subprocess
        from pathlib import Path

        input_file = Path(input_path)
        if not input_file.exists():
            self.logger.error(f"Input file not found: {input_path}")
            return False

        # Create temporary output path
        temp_output = input_file.parent / f"{input_file.stem}_h264.mp4"

        try:
            self.logger.info(f"Converting video to H.264 for browser compatibility...")

            # FFmpeg command to re-encode with H.264
            cmd = [
                'ffmpeg',
                '-i', str(input_file),
                '-c:v', 'libx264',  # H.264 video codec
                '-preset', 'fast',  # Encoding speed
                '-crf', '23',  # Quality (18-28, lower = better quality)
                '-c:a', 'copy',  # Copy audio stream if exists
                '-movflags', '+faststart',  # Enable streaming
                '-y',  # Overwrite output file
                str(temp_output)
            ]

            # Run FFmpeg
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0 and temp_output.exists():
                # Replace original with H.264 version
                temp_output.replace(input_file)
                self.logger.info(f"✅ Video converted to H.264 successfully: {input_file}")
                return True
            else:
                self.logger.error(f"FFmpeg conversion failed: {result.stderr.decode()}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error("FFmpeg conversion timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error during video conversion: {e}", exc_info=True)
            return False
        finally:
            # Clean up temp file if it exists
            if temp_output.exists() and temp_output != input_file:
                try:
                    temp_output.unlink()
                except:
                    pass

    def run(self, video_source, output_path: Optional[str] = None):
        """
        Run the pipeline.

        Args:
            video_source: Video file path or camera index
            output_path: Output video path (optional)
        """
        self.logger.info(f"Starting pipeline: source={video_source}")

        self.running = True
        self.start_time = time.time()

        # Warm up models
        self.logger.info("Warming up models...")
        self.reid_extractor.warmup(num_iterations=5)

        # Start threads
        self.threads = [
            threading.Thread(target=self._capture_thread, args=(video_source,), daemon=True),
            threading.Thread(target=self._detection_thread, daemon=True),
            threading.Thread(target=self._reid_tracking_thread, daemon=True),
            threading.Thread(target=self._display_thread, args=(output_path,), daemon=True)
        ]

        for thread in self.threads:
            thread.start()

        try:
            # Wait for threads to finish (or for interrupt)
            for thread in self.threads:
                thread.join()
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received. Stopping threads...")
            self.running = False
            # Wait for threads to acknowledge the stop
            for thread in self.threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)
        finally:
            # Cleanup
            self.logger.info("Shutting down pipeline and cleaning up resources...")

            # Explicitly clean up CUDA context from ReID extractor
            if self.reid_extractor.inference_mode == 'tensorrt':
                self.reid_extractor.cleanup()

            self.logger.info("Pipeline stopped")
            self._print_summary()

            # Convert video to H.264 for browser compatibility
            if output_path:
                self._convert_to_h264(output_path)
    
    def stop(self):
        """Stop the pipeline"""
        self.running = False
    
    def save_state(self, filepath: Optional[Path] = None):
        """Save pipeline state"""
        if filepath is None:
            filepath = Path('reid_pipeline_state.pkl')
        
        self.gallery_manager.save_gallery(filepath)
        self.logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: Path):
        """Load pipeline state"""
        success = self.gallery_manager.load_gallery(filepath)
        if success:
            self.logger.info(f"State loaded from {filepath}")
        return success
    
    def _print_summary(self):
        """Print pipeline summary"""
        total_time = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.stats['frames_processed'] / total_time if total_time > 0 else 0
        
        gallery_stats = self.gallery_manager.get_statistics()
        
        self.logger.info("\n" + "="*60)
        self.logger.info("Pipeline Summary")
        self.logger.info("="*60)
        self.logger.info(f"Total Time: {total_time:.2f}s")
        self.logger.info(f"Frames Captured: {self.stats['frames_captured']}")
        self.logger.info(f"Frames Processed: {self.stats['frames_processed']}")
        self.logger.info(f"Average FPS: {avg_fps:.2f}")
        self.logger.info(f"Total Detections: {self.stats['total_detections']}")
        self.logger.info(f"Total Persons Tracked: {self.stats['total_persons_tracked']}")
        self.logger.info(f"Gallery Size: {gallery_stats['gallery_size']}")
        self.logger.info("="*60)


if __name__ == "__main__":
    # Test the pipeline
    logging.basicConfig(level=logging.INFO)
    
    pipeline = ProductionReIDPipeline(
        yolo_model_path='yolo11n.pt',
        reid_model_path=None,  # Uses randomly initialized model
        detection_conf=0.3,
        reid_threshold_match=0.70,
        enable_display=True
    )
    
    # Run on video
    pipeline.run(
        video_source='test_video.mp4',
        output_path='output.mp4'
    )
