"""
BoxMOT Tracker Integration
Multi-object tracker using BoTSORT with external TAO ReID embeddings
"""

import numpy as np
import torch
from boxmot import BotSort


class ReIDTracker:
    """Multi-object tracker with external ReID embeddings"""

    def __init__(self, config):
        """
        Initialize BoxMOT tracker with external ReID support

        Args:
            config: Tracker configuration dict
        """
        self.config = config['botsort']
        self.device = torch.device(config.get('device', 'cuda:0'))
        self.fp16 = config.get('fp16', True)

        # Initialize BoTSORT tracker
        print("Initializing BoTSORT tracker...")
        self.tracker = BotSort(
            reid_weights=None,  # No internal ReID model
            device=self.device,
            half=self.fp16,
            with_reid=False,  # Use external embeddings
            track_high_thresh=self.config['track_high_thresh'],
            track_low_thresh=self.config['track_low_thresh'],
            new_track_thresh=self.config['new_track_thresh'],
            track_buffer=self.config['track_buffer'],
            match_thresh=self.config['match_thresh'],
            proximity_thresh=self.config['proximity_thresh'],
            appearance_thresh=self.config['appearance_thresh'],
            fuse_first_associate=self.config['fuse_first_associate']
        )

        # Track history storage
        self.track_history = {}  # track_id -> {embedding, last_seen, bbox}

        print("  ✓ BoTSORT initialized")
        print(f"  Track buffer: {self.config['track_buffer']} frames")
        print(f"  Appearance threshold: {self.config['appearance_thresh']}")

    def update(self, detections: np.ndarray, frame: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """
        Update tracker with new detections and embeddings

        Args:
            detections: [N, 6] array (x1, y1, x2, y2, conf, cls)
            frame: Original frame image (H, W, 3)
            embeddings: [N, 256] TAO ReID embeddings

        Returns:
            tracks: [M, 7] array (x1, y1, x2, y2, track_id, conf, cls)
        """
        if len(detections) == 0:
            # Update tracker with empty detections
            tracks = self.tracker.update(np.empty((0, 6)), frame)
            return tracks

        # Prepare detections for BoxMOT: [x1, y1, x2, y2, conf, cls]
        dets = detections[:, :6]

        # Update tracker with external embeddings
        if embeddings is not None and len(embeddings) > 0:
            tracks = self.tracker.update(dets, frame, embs=embeddings)
        else:
            tracks = self.tracker.update(dets, frame)

        # Update track history
        if len(tracks) > 0:
            for i, track in enumerate(tracks):
                track_id = int(track[4])
                bbox = track[:4]

                # Store TAO embedding for this track
                if i < len(embeddings):
                    self.track_history[track_id] = {
                        "embedding": embeddings[i],
                        "last_seen": len(self.track_history),  # frame counter
                        "bbox": bbox.tolist()
                    }

        return tracks

    def get_track_embedding(self, track_id: int) -> np.ndarray:
        """Get TAO embedding for a specific track"""
        if track_id in self.track_history:
            return self.track_history[track_id]["embedding"]
        return None

    def get_active_track_ids(self) -> list:
        """Get list of all active track IDs"""
        return list(self.track_history.keys())

    def reset(self):
        """Reset tracker state"""
        self.tracker.reset()
        self.track_history.clear()


if __name__ == "__main__":
    # Test script
    import sys
    import yaml
    from pathlib import Path

    # Load config
    config_path = Path("configs/tracker_config.yaml")
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create tracker
    try:
        tracker = ReIDTracker(config)

        # Create dummy detections and embeddings
        dummy_dets = np.array([
            [100, 100, 200, 300, 0.9, 0],  # person 1
            [300, 100, 400, 300, 0.85, 0]  # person 2
        ])

        dummy_embeddings = np.random.randn(2, 256).astype(np.float32)
        dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        print(f"\nTesting tracker with {len(dummy_dets)} detections...")
        tracks = tracker.update(dummy_dets, dummy_frame, dummy_embeddings)

        print(f"  Tracks: {len(tracks)}")
        if len(tracks) > 0:
            print(f"  Track IDs: {[int(t[4]) for t in tracks]}")

        print(f"\nTracker test passed!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
