"""
Multi-Camera ReID Pipeline with Shared Gallery - FIXED VERSION
Single detection thread to avoid CUDA conflicts
"""
import cv2
import time
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging
import threading
import queue
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent / 'reid_pipeline'))

from reid_pipeline.models import EnhancedObjectDetector, Detection, BatchReIDExtractor
from reid_pipeline.gallery import GalleryManager, MatchDecision


@dataclass
class FramePacket:
    camera_id: int
    frame_id: int
    frame: np.ndarray
    timestamp: float


@dataclass
class DetectionPacket:
    camera_id: int
    frame_id: int
    frame: np.ndarray
    detections: List[Detection]
    timestamp: float


class MultiCameraReIDPipeline:
    def __init__(self,
                 yolo_model_path: str,
                 reid_model_path: str,
                 device: str = 'cuda',
                 detection_conf: float = 0.3,
                 reid_threshold_match: float = 0.70,
                 reid_threshold_new: float = 0.50,
                 gallery_max_size: int = 1000,
                 reid_batch_size: int = 16,
                 use_tensorrt: bool = False,
                 tensorrt_precision: str = 'fp16',
                 logger: Optional[logging.Logger] = None):
        
        self.logger = logger or self._setup_logger()
        
        # Shared detector (used in single thread, no lock needed)
        self.detector = EnhancedObjectDetector(
            model_path=yolo_model_path,
            device=device,
            conf_threshold=detection_conf,
            logger=self.logger
        )
        
        # Shared ReID extractor (used in single thread, no lock needed)
        self.reid_extractor = BatchReIDExtractor(
            model_path=reid_model_path,
            device=device,
            batch_size=reid_batch_size,
            use_tensorrt=use_tensorrt,
            tensorrt_precision=tensorrt_precision,
            logger=self.logger
        )
        
        # SHARED gallery for all cameras (with lock for thread safety)
        self.gallery_manager = GalleryManager(
            max_gallery_size=gallery_max_size,
            similarity_threshold_match=reid_threshold_match,
            similarity_threshold_new=reid_threshold_new,
            logger=self.logger
        )
        self.gallery_lock = threading.Lock()
        
        self.reid_threshold_match = reid_threshold_match
        self.reid_threshold_new = reid_threshold_new
        
        # Queues: frame -> processing -> output
        self.frame_queues = [queue.Queue(maxsize=10) for _ in range(4)]
        self.output_queues = [queue.Queue(maxsize=20) for _ in range(4)]
        
        self.running = False
        self.threads = []
        
        self.stats = {
            'frames_processed': [0, 0, 0, 0],
            'total_detections': 0,
            'total_persons_tracked': 0
        }
        
        self.logger.info("Multi-camera pipeline initialized with shared gallery")
    
    def _setup_logger(self):
        logger = logging.getLogger('MultiCameraReID')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger
    
    def _capture_thread(self, camera_id: int, video_path: str):
        """Capture frames from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Camera {camera_id}: Failed to open {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        delay = 1.0 / fps
        
        frame_id = 0
        self.logger.info(f"Camera {camera_id}: Started ({fps:.1f} FPS)")
        
        while self.running:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            packet = FramePacket(
                camera_id=camera_id,
                frame_id=frame_id,
                frame=frame,
                timestamp=time.time()
            )
            
            try:
                self.frame_queues[camera_id].put(packet, block=False)
            except queue.Full:
                pass
            
            frame_id += 1
            
            elapsed = time.time() - start
            sleep_time = delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        cap.release()
        self.frame_queues[camera_id].put(None)
        self.logger.info(f"Camera {camera_id}: Stopped")
    
    def _processing_thread(self):
        """Single thread for detection + ReID for ALL cameras (avoids CUDA conflicts)"""
        self.logger.info("Processing thread: Started")
        
        active_cameras = [True] * 4
        
        while self.running and any(active_cameras):
            for cam_id in range(4):
                if not active_cameras[cam_id]:
                    continue  # Skip inactive cameras
                
                try:
                    packet = self.frame_queues[cam_id].get(timeout=0.01)  # Increased timeout
                    
                    if packet is None:
                        active_cameras[cam_id] = False
                        self.output_queues[cam_id].put(None)
                        self.logger.info(f"Camera {cam_id}: Finished processing")
                        continue
                    
                    # Detect (no lock needed - single thread)
                    detections = self.detector.detect_persons(packet.frame)
                    self.stats['total_detections'] += len(detections)
                    
                    # ReID (no lock needed - single thread)
                    if len(detections) > 0:
                        bboxes = [det.bbox for det in detections]
                        embeddings, valid_flags = self.reid_extractor.extract_features_from_frame(
                            packet.frame, bboxes
                        )
                        
                        valid_detections = [det for det, valid in zip(detections, valid_flags) if valid]
                        
                        if len(valid_detections) > 0:
                            confidences = np.array([det.confidence for det in valid_detections])
                            
                            # Match to gallery (still need lock for gallery)
                            with self.gallery_lock:
                                matches = self.gallery_manager.match_queries_to_gallery(
                                    embeddings, confidences
                                )
                                
                                for i, (det, (person_id, decision, similarity)) in enumerate(zip(valid_detections, matches)):
                                    if decision == MatchDecision.MATCH or decision == MatchDecision.UNCERTAIN:
                                        det.track_id = person_id
                                        self.gallery_manager.update_gallery_entry(
                                            person_id, embeddings[i], det.confidence,
                                            packet.frame_id, det.bbox
                                        )
                                    else:  # NEW
                                        new_id = self.gallery_manager.next_person_id
                                        det.track_id = new_id
                                        success = self.gallery_manager.add_to_gallery(
                                            new_id, embeddings[i], det.confidence,
                                            packet.frame_id, det.bbox
                                        )
                                        if success:
                                            self.stats['total_persons_tracked'] += 1
                                
                                self.gallery_manager.update_frame()
                        
                        detections = valid_detections
                    
                    # Create output packet
                    out_packet = DetectionPacket(
                        camera_id=cam_id,
                        frame_id=packet.frame_id,
                        frame=packet.frame,
                        detections=detections,
                        timestamp=time.time()
                    )
                    
                    self.stats['frames_processed'][cam_id] += 1
                    
                    try:
                        self.output_queues[cam_id].put(out_packet, block=False)
                    except queue.Full:
                        pass
                    
                except queue.Empty:
                    continue  # No frame available from this camera right now
        
        # Send stop signal to all cameras that haven't finished yet
        for cam_id in range(4):
            if active_cameras[cam_id]:
                self.output_queues[cam_id].put(None)
        
        self.logger.info("Processing thread: Stopped")
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Detection], camera_id: int):
        """Draw detections on frame"""
        annotated = frame.copy()
        
        for det in detections:
            if det.track_id is not None:
                x1, y1, x2, y2 = det.bbox.astype(int)
                color = self._get_color_for_id(det.track_id)
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{det.track_id}"
                
                cv2.rectangle(annotated, (x1, y1-20), (x1+80, y1), color, -1)
                cv2.putText(annotated, label, (x1+2, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Camera label
        cv2.rectangle(annotated, (5, 5), (120, 30), (0, 0, 0), -1)
        cv2.putText(annotated, f"Camera {camera_id+1}", (10, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated
    
    def _get_color_for_id(self, track_id: int):
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (255, 128, 0), (128, 255, 0), (0, 255, 128), (128, 0, 255)
        ]
        return colors[track_id % len(colors)]
    
    def _display_thread(self, output_path: Optional[str] = None):
        """Display and save 2x2 grid"""
        self.logger.info("Display thread: Started")
        
        out = None
        last_frames = [None] * 4  # Keep last frame from each camera
        cameras_finished = [False] * 4
        
        while self.running:
            frames = []
            all_finished = True
            
            for cam_id in range(4):
                if cameras_finished[cam_id]:
                    # Use last frame from this camera
                    frames.append(last_frames[cam_id])
                    continue
                
                all_finished = False
                
                try:
                    packet = self.output_queues[cam_id].get(timeout=0.1)
                    
                    if packet is None:
                        cameras_finished[cam_id] = True
                        self.logger.info(f"Display: Camera {cam_id} finished")
                        frames.append(last_frames[cam_id])  # Use last frame
                        continue
                    
                    annotated = self._draw_detections(packet.frame, packet.detections, cam_id)
                    frames.append(annotated)
                    last_frames[cam_id] = annotated  # Save last frame
                    
                except queue.Empty:
                    # No new frame, use last frame
                    frames.append(last_frames[cam_id])
            
            # Exit when all cameras finished
            if all_finished:
                self.logger.info("Display: All cameras finished")
                break
            
            grid_frame = self._create_grid(frames)
            
            with self.gallery_lock:
                gallery_stats = self.gallery_manager.get_statistics()
            
            cv2.rectangle(grid_frame, (10, 10), (300, 80), (0, 0, 0), -1)
            cv2.putText(grid_frame, f"Gallery: {gallery_stats['gallery_size']}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(grid_frame, f"Total Persons: {self.stats['total_persons_tracked']}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if output_path:
                if out is None:
                    h, w = grid_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
                out.write(grid_frame)
            
            cv2.imshow('Multi-Camera ReID (2x2 Grid)', grid_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
        
        if out:
            out.release()
        cv2.destroyAllWindows()
        self.logger.info("Display thread: Stopped")
    
    def _create_grid(self, frames: List[Optional[np.ndarray]]):
        """Create 2x2 grid from 4 frames"""
        target_h, target_w = None, None
        for f in frames:
            if f is not None:
                target_h, target_w = f.shape[:2]
                break
        
        if target_h is None:
            # All frames are None, create default grid
            target_h, target_w = 480, 640
        
        resized = []
        for i, f in enumerate(frames):
            if f is None:
                # Create black frame with "Finished" or "Waiting" label
                black = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                cv2.putText(black, f"Camera {i+1}", (target_w//2-80, target_h//2-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
                cv2.putText(black, "Waiting...", (target_w//2-80, target_h//2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                resized.append(black)
            else:
                if f.shape[:2] != (target_h, target_w):
                    resized.append(cv2.resize(f, (target_w, target_h)))
                else:
                    resized.append(f)
        
        top_row = np.hstack([resized[0], resized[1]])
        bottom_row = np.hstack([resized[2], resized[3]])
        grid = np.vstack([top_row, bottom_row])
        
        return grid
    
    def run(self, video_paths: List[str], output_path: Optional[str] = None):
        """Run multi-camera pipeline"""
        if len(video_paths) != 4:
            raise ValueError("Exactly 4 video paths required")
        
        self.logger.info("="*60)
        self.logger.info("Starting Multi-Camera ReID Pipeline")
        self.logger.info(f"Videos: {video_paths}")
        self.logger.info(f"Output: {output_path}")
        self.logger.info("="*60)
        
        self.running = True
        
        # Warmup
        self.logger.info("Warming up models...")
        self.reid_extractor.warmup(num_iterations=3)
        
        # Start threads
        self.threads = []
        
        # Capture threads (4)
        for i, video_path in enumerate(video_paths):
            t = threading.Thread(target=self._capture_thread, args=(i, video_path), daemon=True)
            self.threads.append(t)
        
        # Single processing thread (detection + ReID for all cameras)
        t = threading.Thread(target=self._processing_thread, daemon=True)
        self.threads.append(t)
        
        # Display thread (1)
        t = threading.Thread(target=self._display_thread, args=(output_path,), daemon=True)
        self.threads.append(t)
        
        # Start all
        for t in self.threads:
            t.start()
        
        try:
            for t in self.threads:
                t.join()
        except KeyboardInterrupt:
            self.logger.info("Stopping...")
            self.running = False
            for t in self.threads:
                t.join(timeout=1.0)
        
        # Cleanup
        if self.reid_extractor.inference_mode == 'tensorrt':
            self.reid_extractor.cleanup()
        
        self.logger.info("="*60)
        self.logger.info("Pipeline Complete")
        self.logger.info(f"Total persons tracked: {self.stats['total_persons_tracked']}")
        with self.gallery_lock:
            gallery_size = self.gallery_manager.get_statistics()['gallery_size']
        self.logger.info(f"Gallery size: {gallery_size}")
        self.logger.info("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Camera ReID Pipeline')
    parser.add_argument('--videos', nargs=4, required=True, help='4 video paths')
    parser.add_argument('--output', '-o', required=True, help='Output video path')
    parser.add_argument('--yolo', default='yolo11n.pt', help='YOLO model')
    parser.add_argument('--reid', required=True, help='ReID model')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--conf', type=float, default=0.3, help='Detection confidence')
    parser.add_argument('--match-thresh', type=float, default=0.5, help='Match threshold')
    parser.add_argument('--new-thresh', type=float, default=0.7, help='New person threshold')
    parser.add_argument('--tensorrt', action='store_true', help='Use TensorRT')
    
    args = parser.parse_args()
    
    pipeline = MultiCameraReIDPipeline(
        yolo_model_path=args.yolo,
        reid_model_path=args.reid,
        device=args.device,
        detection_conf=args.conf,
        reid_threshold_match=args.match_thresh,
        reid_threshold_new=args.new_thresh,
        gallery_max_size=1000,
        reid_batch_size=16,
        use_tensorrt=args.tensorrt,
        tensorrt_precision='fp16'
    )
    
    pipeline.run(args.videos, args.output)