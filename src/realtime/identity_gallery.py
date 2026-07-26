"""Global identity gallery for cross-camera realtime ReID."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class GalleryEntry:
    global_id: int
    embedding: np.ndarray
    last_seen: float
    camera_id: str
    local_track_id: int
    exemplars: list[np.ndarray]
    tracklet_embeddings: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)
    tracklet_last_seen: dict[tuple[str, int], float] = field(default_factory=dict)


class IdentityGallery:
    """
    Assign global IDs from appearance embeddings.

    Local BoTSORT IDs are camera-scoped. This gallery maps each
    (camera_id, local_track_id) pair to a global person ID and can match new
    local tracks from other cameras using cosine distance.
    """

    def __init__(
        self,
        match_threshold: float = 0.5,
        ttl_seconds: float = 300.0,
        ema_alpha: float = 0.8,
        max_exemplars: int = 8,
        use_exemplars_for_matching: bool = False,
    ):
        self.match_threshold = float(match_threshold)
        self.ttl_seconds = float(ttl_seconds)
        self.ema_alpha = float(ema_alpha)
        self.max_exemplars = max(1, int(max_exemplars))
        self.use_exemplars_for_matching = bool(use_exemplars_for_matching)
        self.next_global_id = 1
        self.entries: dict[int, GalleryEntry] = {}
        self.local_to_global: dict[tuple[str, int], int] = {}
        self.last_match: dict[str, Any] | None = None

    def get_existing_global_id(self, camera_id: str, local_track_id: int) -> int | None:
        """Return the mapped global ID for an existing local track, if still live."""
        global_id = self.local_to_global.get((camera_id, int(local_track_id)))
        if global_id is not None and global_id in self.entries:
            return global_id
        return None

    def reset(self, reset_id_counter: bool = True) -> None:
        """Clear gallery state, optionally restarting visible IDs from one."""
        self.entries.clear()
        self.local_to_global.clear()
        self.last_match = None
        if reset_id_counter:
            self.next_global_id = 1

    def remove_camera_mappings(self, camera_id: str) -> None:
        """Forget camera-local tracks while retaining global appearance entries."""
        self.local_to_global = {
            key: global_id
            for key, global_id in self.local_to_global.items()
            if key[0] != camera_id
        }

    def preview_match(
        self,
        embedding: np.ndarray,
        timestamp: float | None = None,
        blocked_global_ids: set[int] | None = None,
    ) -> tuple[int | None, dict[str, Any]]:
        """Check whether an embedding matches the gallery without mutating state."""
        now = float(time.time() if timestamp is None else timestamp)
        self.prune(now)
        return self._best_match(self._normalize(embedding), blocked_global_ids=blocked_global_ids)

    def distance_to_global(self, global_id: int, embedding: np.ndarray) -> float | None:
        """Return the best cosine distance from an embedding to one gallery ID."""
        entry = self.entries.get(int(global_id))
        if entry is None:
            return None

        embedding = self._normalize(embedding)
        distance, _, _ = self._distance_to_entry(entry, embedding)
        return float(distance)

    def assign_to_global(
        self,
        camera_id: str,
        local_track_id: int,
        global_id: int,
        embedding: np.ndarray,
        timestamp: float | None = None,
    ) -> int:
        """Force one local track onto an existing global ID and update that gallery entry."""
        now = float(time.time() if timestamp is None else timestamp)
        self.prune(now)

        global_id = int(global_id)
        if global_id not in self.entries:
            raise KeyError(f"Unknown global ID: {global_id}")

        embedding = self._normalize(embedding)
        self._update(global_id, camera_id, local_track_id, embedding, now)
        self.local_to_global[(camera_id, int(local_track_id))] = global_id
        self.last_match = {
            "matched": True,
            "reason": "forced_global_assignment",
            "global_id": global_id,
        }
        return global_id

    def bind_local_to_global(
        self,
        camera_id: str,
        local_track_id: int,
        global_id: int,
    ) -> int:
        """Bind a local track without updating the identity appearance prototype."""
        global_id = int(global_id)
        if global_id not in self.entries:
            raise KeyError(f"Unknown global ID: {global_id}")
        self.local_to_global[(camera_id, int(local_track_id))] = global_id
        self.last_match = {
            "matched": True,
            "reason": "bound_without_gallery_update",
            "global_id": global_id,
        }
        return global_id

    def create_new_identity(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: np.ndarray,
        timestamp: float | None = None,
        match_info: dict[str, Any] | None = None,
        reason: str = "new_identity",
    ) -> int:
        """Create a new global ID for a local track, replacing any old local mapping."""
        now = float(time.time() if timestamp is None else timestamp)
        self.prune(now)

        embedding = self._normalize(embedding)
        global_id = self.next_global_id
        self.next_global_id += 1
        self.entries[global_id] = GalleryEntry(
            global_id=global_id,
            embedding=embedding.copy(),
            last_seen=now,
            camera_id=camera_id,
            local_track_id=int(local_track_id),
            exemplars=[embedding.copy()],
            tracklet_embeddings={(camera_id, int(local_track_id)): embedding.copy()},
            tracklet_last_seen={(camera_id, int(local_track_id)): now},
        )
        self.local_to_global[(camera_id, int(local_track_id))] = global_id
        self.last_match = {
            **(match_info or {}),
            "matched": False,
            "reason": reason,
            "global_id": global_id,
        }
        return global_id

    def assign(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: np.ndarray,
        timestamp: float | None = None,
        blocked_global_ids: set[int] | None = None,
    ) -> int:
        """Return a stable global ID for one local track observation."""
        now = float(time.time() if timestamp is None else timestamp)
        self.prune(now)

        key = (camera_id, int(local_track_id))
        embedding = self._normalize(embedding)

        existing_id = self.local_to_global.get(key)
        if existing_id is not None and existing_id in self.entries:
            self._update(existing_id, camera_id, local_track_id, embedding, now)
            self.last_match = {
                "matched": True,
                "reason": "existing_local_track",
                "global_id": existing_id,
                "best_distance": 0.0,
                "second_distance": None,
            }
            return existing_id

        global_id, match_info = self._best_match(embedding, blocked_global_ids=blocked_global_ids)
        if global_id is None:
            global_id = self.create_new_identity(
                camera_id,
                local_track_id,
                embedding,
                timestamp=now,
                match_info=match_info,
            )
        else:
            self._update(global_id, camera_id, local_track_id, embedding, now)
            self.last_match = {
                **match_info,
                "matched": True,
                "reason": "appearance_match",
                "global_id": global_id,
            }

        self.local_to_global[key] = global_id
        return global_id

    def prune(self, now: float | None = None) -> None:
        """Remove stale global entries and their local mappings."""
        now = float(time.time() if now is None else now)
        expired = [
            global_id
            for global_id, entry in self.entries.items()
            if now - entry.last_seen > self.ttl_seconds
        ]
        for global_id in expired:
            del self.entries[global_id]

        live_ids = set(self.entries)
        self.local_to_global = {
            key: global_id
            for key, global_id in self.local_to_global.items()
            if global_id in live_ids
        }

    def snapshot(self) -> list[dict]:
        """Return JSON-serializable gallery state."""
        return [
            {
                "global_id": entry.global_id,
                "last_seen": entry.last_seen,
                "camera_id": entry.camera_id,
                "local_track_id": entry.local_track_id,
                "exemplars": len(entry.exemplars),
            }
            for entry in sorted(self.entries.values(), key=lambda item: item.global_id)
        ]

    def _best_match(
        self,
        embedding: np.ndarray,
        blocked_global_ids: set[int] | None = None,
    ) -> tuple[int | None, dict[str, Any]]:
        best_id = None
        best_distance = float("inf")
        second_distance = float("inf")
        best_centroid_distance = float("inf")
        best_exemplar_distance = float("inf")
        blocked_best_id = None
        blocked_best_distance = float("inf")
        blocked_global_ids = blocked_global_ids or set()

        for global_id, entry in self.entries.items():
            distance, centroid_distance, exemplar_distance = self._distance_to_entry(entry, embedding)
            if global_id in blocked_global_ids:
                if distance < blocked_best_distance:
                    blocked_best_id = global_id
                    blocked_best_distance = distance
                continue
            if distance < best_distance:
                second_distance = best_distance
                best_distance = distance
                best_id = global_id
                best_centroid_distance = centroid_distance
                best_exemplar_distance = exemplar_distance
            elif distance < second_distance:
                second_distance = distance

        blocked_candidate_is_better = (
            blocked_best_id is not None
            and (best_id is None or blocked_best_distance <= best_distance)
        )
        info = {
            "candidate_global_id": best_id,
            "best_distance": None if best_id is None else float(best_distance),
            "second_distance": None if not np.isfinite(second_distance) else float(second_distance),
            "threshold": self.match_threshold,
            "blocked_global_ids": sorted(blocked_global_ids),
            "centroid_distance": None if best_id is None else float(best_centroid_distance),
            "exemplar_distance": None if best_id is None else float(best_exemplar_distance),
            "use_exemplars_for_matching": self.use_exemplars_for_matching,
            "blocked_candidate_global_id": blocked_best_id,
            "blocked_candidate_distance": (
                None if blocked_best_id is None else float(blocked_best_distance)
            ),
            "blocked_candidate_is_better": blocked_candidate_is_better,
        }
        if blocked_candidate_is_better:
            return None, info
        return (best_id, info) if best_distance <= self.match_threshold else (None, info)

    def _distance_to_entry(
        self,
        entry: GalleryEntry,
        embedding: np.ndarray,
    ) -> tuple[float, float, float]:
        centroid_distance = 1.0 - float(np.dot(entry.embedding, embedding))
        if entry.exemplars:
            exemplar_distance = min(1.0 - float(np.dot(exemplar, embedding)) for exemplar in entry.exemplars)
        else:
            exemplar_distance = centroid_distance

        if self.use_exemplars_for_matching:
            return min(centroid_distance, exemplar_distance), centroid_distance, exemplar_distance
        return centroid_distance, centroid_distance, exemplar_distance

    def _update(
        self,
        global_id: int,
        camera_id: str,
        local_track_id: int,
        embedding: np.ndarray,
        timestamp: float,
    ) -> None:
        entry = self.entries[global_id]
        # Delayed packets must not rewrite a newer online representation. The
        # prime normally processes event time in order, but independent camera
        # connections can still deliver an older observation late.
        if timestamp < entry.last_seen:
            return
        tracklet_key = (camera_id, int(local_track_id))
        previous = entry.tracklet_embeddings.get(tracklet_key)
        if previous is None:
            tracklet_embedding = embedding.copy()
        else:
            mixed = self.ema_alpha * previous + (1.0 - self.ema_alpha) * embedding
            tracklet_embedding = self._normalize(mixed)
        entry.tracklet_embeddings[tracklet_key] = tracklet_embedding
        entry.tracklet_last_seen[tracklet_key] = timestamp
        self._prune_tracklet_prototypes(entry, protected_key=tracklet_key)

        entry.exemplars = [
            prototype.copy() for prototype in entry.tracklet_embeddings.values()
        ]
        entry.embedding = self._normalize(np.mean(entry.exemplars, axis=0))
        entry.last_seen = timestamp
        entry.camera_id = camera_id
        entry.local_track_id = int(local_track_id)

    def _prune_tracklet_prototypes(
        self,
        entry: GalleryEntry,
        protected_key: tuple[str, int],
    ) -> None:
        """Bound memory while retaining at least one prototype per camera when possible."""
        while len(entry.tracklet_embeddings) > self.max_exemplars:
            camera_counts: dict[str, int] = {}
            for camera_id, _ in entry.tracklet_embeddings:
                camera_counts[camera_id] = camera_counts.get(camera_id, 0) + 1

            removable = [
                key
                for key in entry.tracklet_embeddings
                if key != protected_key and camera_counts[key[0]] > 1
            ]
            if not removable:
                removable = [key for key in entry.tracklet_embeddings if key != protected_key]
            if not removable:
                return
            oldest = min(removable, key=lambda key: entry.tracklet_last_seen.get(key, 0.0))
            del entry.tracklet_embeddings[oldest]
            entry.tracklet_last_seen.pop(oldest, None)

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError("Cannot add empty or non-finite embedding to identity gallery")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("Cannot add zero-norm embedding to identity gallery")
        return vector / norm
