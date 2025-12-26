"""
Visualization Utilities
Draw bounding boxes, track IDs, and other visualizations on frames
"""

import cv2
import numpy as np
from typing import Tuple


class Visualizer:
    """Frame visualization utilities"""

    def __init__(self, line_thickness: int = 2, font_scale: float = 0.6):
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Color palette for tracks
        self.colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for track ID"""
        return self.colors[track_id % len(self.colors)]

    def draw_detection(self, frame: np.ndarray, bbox: np.ndarray, conf: float, color: Tuple[int, int, int] = (0, 255, 0)):
        """Draw a single detection"""
        x1, y1, x2, y2 = map(int, bbox[:4])

        # Draw bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)

        # Draw confidence
        label = f"{conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, self.font, self.font_scale, self.line_thickness)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), self.font, self.font_scale, (255, 255, 255), self.line_thickness)

    def draw_track(self, frame: np.ndarray, track: np.ndarray):
        """Draw a single track with ID"""
        # BoxMOT BoTSORT returns: [x1, y1, x2, y2, track_id, conf, cls, index]
        if len(track) == 8:
            x1, y1, x2, y2, track_id, conf, cls, _ = track  # ignore index
        else:
            x1, y1, x2, y2, track_id, conf, cls = track
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)

        color = self.get_color(track_id)

        # Draw bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)

        # Draw track ID and confidence
        label = f"ID:{track_id} {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, self.font, self.font_scale, self.line_thickness)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), self.font, self.font_scale, (255, 255, 255), self.line_thickness)

    def draw_detections(self, frame: np.ndarray, detections: np.ndarray):
        """Draw all detections"""
        for det in detections:
            self.draw_detection(frame, det[:4], det[4])

    def draw_tracks(self, frame: np.ndarray, tracks: np.ndarray):
        """Draw all tracks"""
        for track in tracks:
            self.draw_track(frame, track)

    def draw_stats(self, frame: np.ndarray, stats: dict, position: str = "top-left"):
        """Draw statistics on frame"""
        lines = []
        for key, value in stats.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.2f}")
            else:
                lines.append(f"{key}: {value}")

        # Position
        if position == "top-left":
            x, y = 10, 30
        elif position == "top-right":
            x, y = frame.shape[1] - 200, 30
        else:
            x, y = 10, 30

        # Draw background
        max_width = max([cv2.getTextSize(line, self.font, self.font_scale, self.line_thickness)[0][0] for line in lines])
        total_height = len(lines) * 30
        cv2.rectangle(frame, (x - 5, y - 25), (x + max_width + 5, y + total_height), (0, 0, 0), -1)

        # Draw text
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x, y + i * 30), self.font, self.font_scale, (255, 255, 255), self.line_thickness)


if __name__ == "__main__":
    # Test visualizer
    import numpy as np

    visualizer = Visualizer()

    # Create dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw dummy tracks
    tracks = np.array([
        [100, 100, 200, 300, 1, 0.95, 0],
        [300, 150, 400, 350, 2, 0.88, 0]
    ])

    visualizer.draw_tracks(frame, tracks)

    # Draw stats
    stats = {"FPS": 25.5, "Tracks": 2, "Frame": 100}
    visualizer.draw_stats(frame, stats)

    print("Visualizer test passed!")
    print("To view output, save frame and open with image viewer")
