"""
Performance Metrics
Calculate and track performance metrics for the pipeline
"""

import time
from collections import deque
from typing import Deque


class PerformanceMetrics:
    """Track pipeline performance metrics"""

    def __init__(self, window_size: int = 30):
        """
        Initialize metrics tracker

        Args:
            window_size: Window size for moving average (frames)
        """
        self.window_size = window_size

        # Moving windows
        self.fps_window: Deque[float] = deque(maxlen=window_size)
        self.detection_time_window: Deque[float] = deque(maxlen=window_size)
        self.reid_time_window: Deque[float] = deque(maxlen=window_size)
        self.tracking_time_window: Deque[float] = deque(maxlen=window_size)

        # Counters
        self.total_frames = 0
        self.total_detections = 0
        self.total_tracks = 0

        # Timers
        self.start_time = time.time()

    def update(self, detection_time: float, reid_time: float, tracking_time: float,
               num_detections: int, num_tracks: int):
        """Update metrics with new frame data"""
        # Calculate FPS
        frame_time = detection_time + reid_time + tracking_time
        if frame_time > 0:
            fps = 1.0 / frame_time
        else:
            fps = 0.0

        # Update windows
        self.fps_window.append(fps)
        self.detection_time_window.append(detection_time * 1000)  # ms
        self.reid_time_window.append(reid_time * 1000)  # ms
        self.tracking_time_window.append(tracking_time * 1000)  # ms

        # Update counters
        self.total_frames += 1
        self.total_detections += num_detections
        self.total_tracks += num_tracks

    def get_average_fps(self) -> float:
        """Get average FPS over window"""
        if len(self.fps_window) == 0:
            return 0.0
        return sum(self.fps_window) / len(self.fps_window)

    def get_average_detection_time(self) -> float:
        """Get average detection time (ms) over window"""
        if len(self.detection_time_window) == 0:
            return 0.0
        return sum(self.detection_time_window) / len(self.detection_time_window)

    def get_average_reid_time(self) -> float:
        """Get average ReID time (ms) over window"""
        if len(self.reid_time_window) == 0:
            return 0.0
        return sum(self.reid_time_window) / len(self.reid_time_window)

    def get_average_tracking_time(self) -> float:
        """Get average tracking time (ms) over window"""
        if len(self.tracking_time_window) == 0:
            return 0.0
        return sum(self.tracking_time_window) / len(self.tracking_time_window)

    def get_summary(self) -> dict:
        """Get metrics summary"""
        elapsed_time = time.time() - self.start_time

        return {
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "total_tracks": self.total_tracks,
            "elapsed_time_sec": elapsed_time,
            "overall_fps": self.total_frames / elapsed_time if elapsed_time > 0 else 0.0,
            "average_fps": self.get_average_fps(),
            "average_detection_time_ms": self.get_average_detection_time(),
            "average_reid_time_ms": self.get_average_reid_time(),
            "average_tracking_time_ms": self.get_average_tracking_time(),
            "average_total_time_ms": self.get_average_detection_time() + self.get_average_reid_time() + self.get_average_tracking_time()
        }


if __name__ == "__main__":
    # Test metrics
    import random

    metrics = PerformanceMetrics(window_size=10)

    print("Simulating 20 frames...")
    for i in range(20):
        det_time = random.uniform(0.01, 0.03)
        reid_time = random.uniform(0.005, 0.015)
        track_time = random.uniform(0.002, 0.008)
        num_dets = random.randint(1, 10)
        num_tracks = random.randint(1, 8)

        metrics.update(det_time, reid_time, track_time, num_dets, num_tracks)

        if (i + 1) % 5 == 0:
            print(f"\nFrame {i+1}:")
            print(f"  Average FPS: {metrics.get_average_fps():.2f}")
            print(f"  Detection time: {metrics.get_average_detection_time():.2f} ms")
            print(f"  ReID time: {metrics.get_average_reid_time():.2f} ms")

    print("\nFinal summary:")
    summary = metrics.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\nMetrics test passed!")
