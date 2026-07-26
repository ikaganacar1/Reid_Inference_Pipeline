"""
BoxMOT Tracker Integration
Multi-object tracker using BoTSORT with external TAO ReID embeddings
"""

import numpy as np
import torch
import importlib

try:
    from boxmot import BotSort
except ImportError:
    # BoxMOT 19+ no longer exposes BotSort at the package top level.
    from boxmot.trackers.bbox.botsort.botsort import BotSort


_botsort_module = importlib.import_module(BotSort.__module__)
_BoxMOTBaseTrack = _botsort_module.BaseTrack


class ExternalReIDBotSort(BotSort):
    """
    BotSort variant that uses external embeddings without loading an internal ReID model.

    Standard BotSort requires reid_weights when with_reid=True, but we use external
    embeddings from Triton. This class works around that by:
    1. Initializing with with_reid=False (skips internal model loading)
    2. Setting with_reid=True after init (enables embedding-based matching)
    """

    def __init__(self, with_reid=True, **kwargs):
        # Store the intended with_reid setting
        self._use_external_reid = with_reid
        # BoxMOT resets a module-global ID counter in every BotSort constructor.
        # Multiple camera trackers share that counter, so preserve it to prevent
        # local ID reuse when another camera tracker is created or reset.
        previous_track_count = int(_BoxMOTBaseTrack._count)

        # Initialize parent with with_reid=False to skip internal model loading
        super().__init__(with_reid=False, **kwargs)
        _BoxMOTBaseTrack._count = max(previous_track_count, int(_BoxMOTBaseTrack._count))
        # Enable embedding-based matching for external embeddings
        if self._use_external_reid:
            self.with_reid = True


class ReIDTracker:
    """Multi-object tracker with external ReID embeddings"""

    def __init__(self, config, with_reid=None):
        """
        Initialize BoxMOT tracker with external ReID support

        Args:
            config: Tracker configuration dict
        """
        self.config = config['botsort']
        self.device = torch.device(config.get('device', 'cuda:0'))
        self.fp16 = config.get('fp16', True)
        self.with_reid = self.config.get('with_reid', True) if with_reid is None else with_reid
        self.frame_count = 0

        # Initialize BoTSORT tracker with external ReID support
        print("Initializing BoTSORT tracker...")
        self.tracker = self._create_tracker()

        # Track history storage
        self.track_history = {}  # track_id -> {embedding, last_seen, bbox}

        print("  ✓ BoTSORT initialized")
        print(f"  Track buffer: {self.config['track_buffer']} frames")
        print(f"  Appearance threshold: {self.config['appearance_thresh']}")
        print(f"  ReID enabled: {self.with_reid}")

    def _create_tracker(self):
        """Build a fresh BoTSORT instance with the configured external-ReID mode."""
        self.tracker = ExternalReIDBotSort(
            reid_weights=None,  # No internal model; embeddings come from the configured ReID backend
            device=self.device,
            half=self.fp16,
            with_reid=self.with_reid,  # Enable embedding-based matching for re-identification
            track_high_thresh=self.config['track_high_thresh'],
            track_low_thresh=self.config['track_low_thresh'],
            new_track_thresh=self.config['new_track_thresh'],
            track_buffer=self.config['track_buffer'],
            match_thresh=self.config['match_thresh'],
            proximity_thresh=self.config['proximity_thresh'],
            appearance_thresh=self.config['appearance_thresh'],
            cmc_method=self.config.get('cmc_method', 'ecc'),
            fuse_first_associate=self.config['fuse_first_associate']
        )
        return self.tracker

    def update(self, detections: np.ndarray, frame: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """
        Update tracker with new detections and embeddings

        Args:
            detections: [N, 6] array (x1, y1, x2, y2, conf, cls)
            frame: Original frame image (H, W, 3)
            embeddings: [N, D] TAO ReID embeddings

        Returns:
            tracks: [M, 8] array (x1, y1, x2, y2, track_id, conf, cls, det_idx)
        """
        self.frame_count += 1

        if len(detections) == 0:
            # Passing an empty array prevents the external-ReID tracker from attempting
            # to call an internal model that was intentionally not loaded.
            empty_embeddings = np.empty((0, 0), dtype=np.float32) if self.with_reid else None
            tracks = self.tracker.update(np.empty((0, 6)), frame, embs=empty_embeddings)
            self._prune_track_history()
            return tracks

        # Prepare detections for BoxMOT: [x1, y1, x2, y2, conf, cls]
        dets = detections[:, :6]

        if self.with_reid:
            if embeddings is None or len(embeddings) == 0:
                raise ValueError("ReID is enabled, but no external embeddings were provided")
            if len(embeddings) != len(dets):
                raise ValueError(
                    f"Detection/embedding count mismatch: {len(dets)} detections, "
                    f"{len(embeddings)} embeddings"
                )
        else:
            embeddings = None

        tracks = self.tracker.update(dets, frame, embs=embeddings)

        # Update track history
        if self.with_reid and len(tracks) > 0:
            for i, track in enumerate(tracks):
                track_id = int(track[4])
                bbox = track[:4]

                # BoTSORT returns the original detection index in track[7].
                if len(track) >= 8:
                    det_idx = int(track[7])
                else:
                    det_idx = i

                if 0 <= det_idx < len(embeddings):
                    self.track_history[track_id] = {
                        "embedding": embeddings[det_idx],
                        "last_seen": self.frame_count,
                        "bbox": bbox.tolist()
                    }

        self._prune_track_history()
        return tracks

    def _prune_track_history(self):
        """Discard cached embeddings after the configured local retention window."""
        track_buffer = self.config['track_buffer']
        expired_ids = [
            track_id
            for track_id, track in self.track_history.items()
            if self.frame_count - track["last_seen"] > track_buffer
        ]
        for track_id in expired_ids:
            del self.track_history[track_id]

    def get_track_embedding(self, track_id: int) -> np.ndarray:
        """Get TAO embedding for a specific track"""
        if track_id in self.track_history:
            return self.track_history[track_id]["embedding"]
        return None

    def get_active_track_ids(self) -> list:
        """Get list of all active track IDs"""
        return [int(track.id) for track in self.tracker.active_tracks]

    def reset(self):
        """Reset tracker state"""
        self.tracker = self._create_tracker()
        self.frame_count = 0
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

        print("\nTracker test passed!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
