"""Global identity assignment shared by realtime and offline replay."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.realtime.camera_topology import CameraTopology
from src.realtime.identity_gallery import IdentityGallery
from src.realtime.protocol import write_jpeg


class GlobalIdentityAssigner:
    """Map per-camera local tracks to global IDs using ReID embeddings."""

    def __init__(self, prime_config: dict[str, Any], output_dir: Path | str = "outputs/realtime"):
        self.prime = prime_config
        self.output_dir = Path(output_dir)
        self.pending_identity_tracks: dict[tuple[str, int], dict[str, Any]] = {}
        self.identity_locations: dict[int, dict[str, float]] = {}
        self.camera_topology = CameraTopology.from_config(self.prime)
        self.gallery = IdentityGallery(
            match_threshold=float(self.prime.get("global_match_threshold", 0.3)),
            ttl_seconds=float(self.prime.get("gallery_ttl_seconds", 300)),
            ema_alpha=float(self.prime.get("gallery_ema_alpha", 0.8)),
            max_exemplars=int(self.prime.get("gallery_max_exemplars", 8)),
            use_exemplars_for_matching=bool(self.prime.get("gallery_use_exemplars_for_matching", False)),
        )

        self.debug_reid = bool(self.prime.get("debug_reid", True))
        self.save_debug_crops = bool(self.prime.get("save_debug_crops", True))
        self.debug_crop_interval_frames = max(1, int(self.prime.get("debug_crop_interval_frames", 10)))
        self.debug_dir = self.output_dir / "reid_debug"
        self.debug_events_path = self.debug_dir / "events.jsonl"

        self.new_identity_min_frames = max(1, int(self.prime.get("new_identity_min_frames", 5)))
        self.new_identity_min_seconds = max(
            0.0,
            float(self.prime.get("new_identity_min_seconds", 0.0)),
        )
        self.new_identity_ambiguous_wait_seconds = max(
            self.new_identity_min_seconds,
            float(self.prime.get("new_identity_ambiguous_wait_seconds", 0.0)),
        )
        self.new_identity_ambiguous_distance = float(
            self.prime.get("new_identity_ambiguous_distance", 0.45)
        )
        self.duplicate_local_track_iou_threshold = float(
            self.prime.get("duplicate_local_track_iou_threshold", 0.85)
        )
        self.pending_identity_ttl_seconds = float(self.prime.get("pending_identity_ttl_seconds", 2.0))
        self.same_camera_conflict_min_frames = max(
            1,
            int(self.prime.get("same_camera_conflict_min_frames", 3)),
        )
        self.same_camera_conflict_min_seconds = max(
            0.0,
            float(self.prime.get("same_camera_conflict_min_seconds", 0.2)),
        )
        self.same_camera_conflict_max_iou = float(
            self.prime.get("same_camera_conflict_max_iou", 0.2)
        )
        self.same_camera_conflict_max_gap_seconds = max(
            0.0,
            float(self.prime.get("same_camera_conflict_max_gap_seconds", 0.5)),
        )
        self.same_camera_conflict_ttl_seconds = max(
            self.pending_identity_ttl_seconds,
            float(
                self.prime.get(
                    "same_camera_conflict_ttl_seconds",
                    self.gallery.ttl_seconds,
                )
            ),
        )
        self.same_camera_track_conflicts: dict[
            tuple[str, int, int], dict[str, Any]
        ] = {}
        self.pending_overlap_link_threshold = float(
            self.prime.get("pending_overlap_link_threshold", 0.22)
        )
        self.pending_overlap_link_active_seconds = max(
            0.0,
            float(self.prime.get("pending_overlap_link_active_seconds", 0.5)),
        )
        self.overlap_lightness_fallback_enabled = bool(
            self.prime.get("overlap_lightness_fallback_enabled", True)
        )
        self.overlap_lightness_min_seconds = max(
            0.0,
            float(self.prime.get("overlap_lightness_min_seconds", 2.0)),
        )
        self.overlap_lightness_min_edge_frames = max(
            1,
            int(self.prime.get("overlap_lightness_min_edge_frames", 5)),
        )
        self.overlap_lightness_max_embedding_distance = float(
            self.prime.get("overlap_lightness_max_embedding_distance", 0.5)
        )
        self.overlap_lightness_max_distance = float(
            self.prime.get("overlap_lightness_max_distance", 0.12)
        )
        self.overlap_lightness_match_margin = float(
            self.prime.get("overlap_lightness_match_margin", 0.04)
        )
        self.overlap_lightness_score_weight = float(
            self.prime.get("overlap_lightness_score_weight", 1.0)
        )
        self.overlap_lightness_min_covisible_seconds = max(
            0.0,
            float(self.prime.get("overlap_lightness_min_covisible_seconds", 1.0)),
        )
        self.identity_lightness_ema_alpha = float(
            self.prime.get("identity_lightness_ema_alpha", 0.8)
        )
        self.identity_lightness_tracklets: dict[
            int, dict[tuple[str, int], float]
        ] = {}
        self.recheck_existing_tracks = bool(self.prime.get("recheck_existing_tracks", True))
        self.existing_track_recheck_threshold = float(self.prime.get("existing_track_recheck_threshold", 0.55))
        self.existing_track_remap_margin = float(self.prime.get("existing_track_remap_margin", 0.08))
        self.existing_track_remap_require_interior = bool(
            self.prime.get("existing_track_remap_require_interior", True)
        )
        self.existing_track_max_distance = float(self.prime.get("existing_track_max_distance", 0.45))
        self.identity_update_max_distance = float(
            self.prime.get("identity_update_max_distance", self.existing_track_max_distance)
        )
        self.split_existing_track_on_drift = bool(self.prime.get("split_existing_track_on_drift", False))

        self.new_track_match_threshold = float(self.prime.get("new_track_match_threshold", 0.3))
        self.new_track_match_margin = float(self.prime.get("new_track_match_margin", 0.08))
        self.new_track_single_candidate_threshold = float(
            self.prime.get("new_track_single_candidate_threshold", 0.16)
        )
        self.overlap_new_track_match_threshold = float(
            self.prime.get("overlap_new_track_match_threshold", 0.35)
        )
        self.overlap_new_track_match_margin = float(
            self.prime.get("overlap_new_track_match_margin", 0.04)
        )
        self.overlap_candidate_active_seconds = max(
            0.0,
            float(self.prime.get("overlap_candidate_active_seconds", 0.5)),
        )
        self.overlap_new_track_min_seconds = max(
            self.new_identity_min_seconds,
            float(
                self.prime.get(
                    "overlap_new_track_min_seconds",
                    self.new_identity_ambiguous_wait_seconds,
                )
            ),
        )
        self.adjacent_new_track_match_threshold = float(
            self.prime.get("adjacent_new_track_match_threshold", 0.3)
        )
        self.adjacent_new_track_match_margin = float(
            self.prime.get("adjacent_new_track_match_margin", 0.05)
        )
        self.adjacent_candidate_max_seconds = max(
            0.0,
            float(self.prime.get("adjacent_candidate_max_seconds", 5.0)),
        )
        self.adjacent_camera_exclusion_seconds = max(
            0.0,
            float(self.prime.get("adjacent_camera_exclusion_seconds", 0.25)),
        )

        self.identity_min_confidence = float(self.prime.get("identity_min_confidence", 0.55))
        self.identity_min_height_ratio = float(self.prime.get("identity_min_height_ratio", 0.08))
        self.identity_min_area_ratio = float(self.prime.get("identity_min_area_ratio", 0.0015))
        self.identity_edge_margin_ratio = float(self.prime.get("identity_edge_margin_ratio", 0.01))
        self.identity_edge_min_confidence = float(
            self.prime.get("identity_edge_min_confidence", max(self.identity_min_confidence, 0.65))
        )
        self.new_identity_require_interior = bool(
            self.prime.get("new_identity_require_interior", True)
        )
        self.cross_camera_exclusion_seconds = max(
            0.0,
            float(self.prime.get("cross_camera_exclusion_seconds", 1.0)),
        )

        if self.debug_reid:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        self.gallery.reset()
        self.pending_identity_tracks.clear()
        self.identity_locations.clear()
        self.same_camera_track_conflicts.clear()
        self.identity_lightness_tracklets.clear()

    def reset_camera(self, camera_id: str) -> None:
        """Clear local state after a camera stream restarts without losing the gallery."""
        self.gallery.remove_camera_mappings(camera_id)
        for key in [key for key in self.pending_identity_tracks if key[0] == camera_id]:
            del self.pending_identity_tracks[key]
        for key in [key for key in self.same_camera_track_conflicts if key[0] == camera_id]:
            del self.same_camera_track_conflicts[key]
        for global_id in list(self.identity_lightness_tracklets):
            tracklets = self.identity_lightness_tracklets[global_id]
            self.identity_lightness_tracklets[global_id] = {
                key: value for key, value in tracklets.items() if key[0] != camera_id
            }
            if not self.identity_lightness_tracklets[global_id]:
                del self.identity_lightness_tracklets[global_id]
        for global_id in list(self.identity_locations):
            self.identity_locations[global_id].pop(camera_id, None)
            if not self.identity_locations[global_id]:
                del self.identity_locations[global_id]

    def assign_tracks(
        self,
        camera_id: str,
        frame_id: int,
        frame_width: int,
        frame_height: int,
        tracks: np.ndarray,
        embeddings: np.ndarray,
        crops: list[np.ndarray] | None = None,
        timestamp: float | None = None,
        blocked_global_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any] | None] = [None] * len(tracks)
        timestamp = time.time() if timestamp is None else float(timestamp)
        frame_global_ids: set[int] = set(blocked_global_ids or set())
        frame_global_ids.update(self.blocked_global_ids_for_camera(camera_id, timestamp))
        self.prune_pending_identity_tracks(timestamp)
        self.prune_same_camera_track_conflicts(timestamp)
        self.observe_same_camera_track_conflicts(
            camera_id,
            frame_id,
            tracks,
            timestamp,
        )

        existing_ids_by_track = {
            index: self.gallery.get_existing_global_id(camera_id, int(track[4]))
            for index, track in enumerate(tracks)
        }
        reserved_existing_ids = {
            global_id for global_id in existing_ids_by_track.values() if global_id is not None
        }

        track_order = sorted(
            range(len(tracks)),
            key=lambda index: self.track_assignment_priority(camera_id, tracks[index]),
        )
        for track_index in track_order:
            track = tracks[track_index]
            local_track_id = int(track[4])
            det_idx = int(track[7]) if len(track) >= 8 else -1
            global_id = None
            match_info = None
            quality_ok = False
            crop = crops[det_idx] if crops is not None and 0 <= det_idx < len(crops) else None
            appearance_lightness = self.torso_lightness(crop)

            if 0 <= det_idx < len(embeddings):
                embedding = embeddings[det_idx]
                duplicate_record = self.find_duplicate_track_record(track, records)
                if duplicate_record is not None:
                    self.pending_identity_tracks.pop((camera_id, local_track_id), None)
                    match_info = {
                        "matched": False,
                        "reason": "duplicate_local_track_suppressed",
                        "global_id": None,
                        "duplicate_local_track_id": duplicate_record["local_track_id"],
                        "duplicate_global_id": duplicate_record.get("global_id"),
                        "duplicate_iou": self.bbox_iou(track[:4], duplicate_record["bbox"]),
                        "iou_threshold": self.duplicate_local_track_iou_threshold,
                    }
                else:
                    existing_global_id = existing_ids_by_track[track_index]
                    track_blocked_global_ids = frame_global_ids | (
                        reserved_existing_ids
                        - ({existing_global_id} if existing_global_id is not None else set())
                    )
                    same_camera_blocked_global_ids = (
                        self.same_camera_conflicting_global_ids(
                            camera_id, local_track_id, existing_global_id
                        )
                    )
                    track_blocked_global_ids.update(same_camera_blocked_global_ids)
                    quality_ok, quality_info = self.identity_observation_quality(
                        track,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                    if (
                        existing_global_id is None
                        and self.new_identity_require_interior
                        and bool(quality_info.get("touches_edge"))
                    ):
                        quality_ok = False
                        quality_info = dict(quality_info)
                        quality_info["ok"] = False
                        quality_info["reasons"] = [
                            *quality_info.get("reasons", []),
                            "edge_partial_new_identity",
                        ]

                    if not quality_ok:
                        if (
                            existing_global_id is None
                            and quality_info.get("reasons")
                            == ["edge_partial_new_identity"]
                        ):
                            global_id, match_info = self.hold_pending_edge_observation(
                                camera_id,
                                local_track_id,
                                frame_id,
                                embedding,
                                timestamp,
                                track_blocked_global_ids,
                                same_camera_blocked_global_ids,
                                quality_info,
                                appearance_lightness,
                            )
                        else:
                            global_id, match_info = self.handle_low_quality_observation(
                                camera_id,
                                local_track_id,
                                existing_global_id,
                                track_blocked_global_ids,
                                quality_info,
                            )
                    else:
                        global_id, match_info = self.assign_embedding(
                            camera_id,
                            local_track_id,
                            frame_id,
                            embedding,
                            timestamp,
                            track_blocked_global_ids,
                            existing_global_id,
                            cannot_link_global_ids=same_camera_blocked_global_ids,
                            appearance_lightness=appearance_lightness,
                            allow_existing_track_remap=(
                                not self.existing_track_remap_require_interior
                                or not bool(quality_info.get("touches_edge"))
                            ),
                        )

                if global_id is not None:
                    frame_global_ids.add(global_id)
                    self.record_identity_location(global_id, camera_id, timestamp)
                    if (
                        quality_ok
                        and not bool(quality_info.get("touches_edge"))
                        and appearance_lightness is not None
                    ):
                        self.update_identity_lightness(
                            global_id,
                            camera_id,
                            local_track_id,
                            appearance_lightness,
                        )

                if self.debug_reid:
                    self.write_reid_debug_event(
                        camera_id,
                        frame_id,
                        track,
                        det_idx,
                        global_id,
                        embedding,
                        match_info,
                        crop,
                        timestamp,
                    )

            records[track_index] = {
                "bbox": [float(x) for x in track[:4]],
                "local_track_id": local_track_id,
                "global_id": global_id,
                "confidence": float(track[5]),
                "class_id": int(track[6]),
                "detection_index": det_idx,
                "match": match_info,
            }

        return [record for record in records if record is not None]

    def observe_same_camera_track_conflicts(
        self,
        camera_id: str,
        frame_id: int,
        tracks: np.ndarray,
        timestamp: float,
    ) -> None:
        """Remember sustained, spatially distinct local-track co-occurrences."""
        for first_index in range(len(tracks)):
            first_id = int(tracks[first_index][4])
            for second_index in range(first_index + 1, len(tracks)):
                second_id = int(tracks[second_index][4])
                if first_id == second_id:
                    continue
                if self.bbox_iou(tracks[first_index][:4], tracks[second_index][:4]) > (
                    self.same_camera_conflict_max_iou
                ):
                    continue

                low_id, high_id = sorted((first_id, second_id))
                key = (camera_id, low_id, high_id)
                state = self.same_camera_track_conflicts.get(key)
                if (
                    state is None
                    or timestamp - float(state["last_seen"])
                    > self.same_camera_conflict_max_gap_seconds
                ):
                    state = {
                        "first_seen": timestamp,
                        "last_seen": timestamp,
                        "last_frame_id": None,
                        "frames": 0,
                        "confirmed": False,
                    }
                    self.same_camera_track_conflicts[key] = state

                if state["last_frame_id"] != int(frame_id):
                    state["frames"] = int(state["frames"]) + 1
                state["last_frame_id"] = int(frame_id)
                state["last_seen"] = timestamp
                elapsed = max(0.0, timestamp - float(state["first_seen"]))
                if (
                    int(state["frames"]) >= self.same_camera_conflict_min_frames
                    and elapsed + 1e-9 >= self.same_camera_conflict_min_seconds
                ):
                    state["confirmed"] = True

    def same_camera_conflicting_global_ids(
        self,
        camera_id: str,
        local_track_id: int,
        existing_global_id: int | None,
    ) -> set[int]:
        """Return IDs belonging to tracks proven distinct in this camera."""
        blocked = set()
        for (conflict_camera, first_id, second_id), state in (
            self.same_camera_track_conflicts.items()
        ):
            if conflict_camera != camera_id or not bool(state.get("confirmed")):
                continue
            if local_track_id == first_id:
                peer_id = second_id
            elif local_track_id == second_id:
                peer_id = first_id
            else:
                continue
            peer_global_id = self.gallery.get_existing_global_id(camera_id, peer_id)
            if peer_global_id is not None and peer_global_id != existing_global_id:
                blocked.add(peer_global_id)
        return blocked

    def prune_same_camera_track_conflicts(self, timestamp: float) -> None:
        cutoff = float(timestamp) - self.same_camera_conflict_ttl_seconds
        for key in list(self.same_camera_track_conflicts):
            if float(self.same_camera_track_conflicts[key]["last_seen"]) < cutoff:
                del self.same_camera_track_conflicts[key]

    def find_duplicate_track_record(
        self,
        track: np.ndarray,
        records: list[dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        if self.duplicate_local_track_iou_threshold > 1.0:
            return None
        for record in records:
            if record is None or int(record["class_id"]) != int(track[6]):
                continue
            if self.bbox_iou(track[:4], record["bbox"]) >= self.duplicate_local_track_iou_threshold:
                return record
        return None

    @staticmethod
    def bbox_iou(first: Any, second: Any) -> float:
        ax1, ay1, ax2, ay2 = [float(value) for value in first[:4]]
        bx1, by1, bx2, by2 = [float(value) for value in second[:4]]
        intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
        intersection = intersection_width * intersection_height
        first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = first_area + second_area - intersection
        return 0.0 if union <= 0.0 else intersection / union

    @staticmethod
    def torso_lightness(crop: np.ndarray | None) -> float | None:
        """Return robust normalized clothing lightness from the upper torso."""
        if crop is None or crop.size == 0 or crop.ndim != 3:
            return None
        height, width = crop.shape[:2]
        top = max(0, int(round(height * 0.15)))
        bottom = min(height, max(top + 1, int(round(height * 0.60))))
        left = max(0, int(round(width * 0.25)))
        right = min(width, max(left + 1, int(round(width * 0.75))))
        torso = crop[top:bottom, left:right]
        if torso.size == 0:
            return None
        lightness = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)[:, :, 0]
        return float(np.median(lightness)) / 255.0

    def update_identity_lightness(
        self,
        global_id: int,
        camera_id: str,
        local_track_id: int,
        lightness: float,
    ) -> None:
        for stale_id in [
            identity_id
            for identity_id in self.identity_lightness_tracklets
            if identity_id not in self.gallery.entries
        ]:
            del self.identity_lightness_tracklets[stale_id]
        tracklets = self.identity_lightness_tracklets.setdefault(int(global_id), {})
        key = (camera_id, int(local_track_id))
        previous = tracklets.get(key)
        tracklets[key] = (
            float(lightness)
            if previous is None
            else self.identity_lightness_ema_alpha * previous
            + (1.0 - self.identity_lightness_ema_alpha) * float(lightness)
        )

    def identity_lightness(self, global_id: int) -> float | None:
        values = self.identity_lightness_tracklets.get(int(global_id), {}).values()
        values = list(values)
        return None if not values else float(np.median(values))

    def track_assignment_priority(self, camera_id: str, track: np.ndarray) -> tuple[bool, float, int]:
        """Reserve IDs for established local tracks before considering new tracks."""
        local_track_id = int(track[4])
        existing_global_id = self.gallery.get_existing_global_id(camera_id, local_track_id)
        return existing_global_id is None, -float(track[5]), local_track_id

    def assign_embedding(
        self,
        camera_id: str,
        local_track_id: int,
        frame_id: int,
        embedding: np.ndarray,
        timestamp: float,
        frame_global_ids: set[int],
        existing_global_id: int | None,
        cannot_link_global_ids: set[int] | None = None,
        appearance_lightness: float | None = None,
        allow_existing_track_remap: bool = True,
    ) -> tuple[int | None, dict[str, Any]]:
        matched_global_id, preview_info = self.gallery.preview_match(
            embedding,
            timestamp=timestamp,
            blocked_global_ids=frame_global_ids,
        )

        if existing_global_id is not None:
            return self.assign_existing_track(
                camera_id,
                local_track_id,
                frame_id,
                embedding,
                timestamp,
                frame_global_ids,
                existing_global_id,
                matched_global_id,
                preview_info,
                allow_existing_track_remap,
            )

        # A first interior crop can be atypical (backlit, occluded, or only just
        # clear of a frame edge).  Confirm every new local track over time before
        # it is allowed to join and mutate an existing global identity.
        return self.pending_identity_response(
            camera_id,
            local_track_id,
            frame_id,
            timestamp,
            preview_info,
            "pending_new_identity",
            embedding,
            blocked_global_ids=frame_global_ids,
            cannot_link_global_ids=cannot_link_global_ids,
            appearance_lightness=appearance_lightness,
        )

    def assign_existing_track(
        self,
        camera_id: str,
        local_track_id: int,
        frame_id: int,
        embedding: np.ndarray,
        timestamp: float,
        frame_global_ids: set[int],
        existing_global_id: int,
        matched_global_id: int | None,
        preview_info: dict[str, Any],
        allow_existing_track_remap: bool = True,
    ) -> tuple[int | None, dict[str, Any]]:
        assigned_distance = self.gallery.distance_to_global(existing_global_id, embedding)
        candidate_distance = preview_info.get("best_distance")
        frame_collision = existing_global_id in frame_global_ids
        is_remap_candidate = (
            self.recheck_existing_tracks
            and matched_global_id is not None
            and matched_global_id != existing_global_id
            and assigned_distance is not None
            and candidate_distance is not None
            and assigned_distance > self.existing_track_recheck_threshold
            and candidate_distance + self.existing_track_remap_margin < assigned_distance
            and self.is_reliable_new_track_match(preview_info)
        )
        should_remap = allow_existing_track_remap and is_remap_candidate

        if frame_collision:
            if (
                allow_existing_track_remap
                and matched_global_id is not None
                and self.is_reliable_new_track_match(preview_info)
            ):
                previous_global_id = existing_global_id
                global_id = self.gallery.assign_to_global(
                    camera_id,
                    local_track_id,
                    matched_global_id,
                    embedding,
                    timestamp=timestamp,
                )
                self.pending_identity_tracks.pop((camera_id, local_track_id), None)
                return global_id, {
                    **preview_info,
                    "matched": True,
                    "reason": "appearance_remap_frame_collision",
                    "global_id": global_id,
                    "previous_global_id": previous_global_id,
                    "assigned_distance": assigned_distance,
                    "remap_margin": self.existing_track_remap_margin,
                }

            return self.pending_identity_response(
                camera_id,
                local_track_id,
                frame_id,
                timestamp,
                preview_info,
                "pending_frame_collision",
                embedding,
                previous_global_id=existing_global_id,
                assigned_distance=assigned_distance,
            )

        if should_remap:
            previous_global_id = existing_global_id
            global_id = self.gallery.assign_to_global(
                camera_id,
                local_track_id,
                matched_global_id,
                embedding,
                timestamp=timestamp,
            )
            self.pending_identity_tracks.pop((camera_id, local_track_id), None)
            return global_id, {
                **preview_info,
                "matched": True,
                "reason": "appearance_remap",
                "global_id": global_id,
                "previous_global_id": previous_global_id,
                "assigned_distance": assigned_distance,
                "remap_margin": self.existing_track_remap_margin,
            }

        if assigned_distance is not None and assigned_distance <= self.existing_track_max_distance:
            if assigned_distance <= self.identity_update_max_distance:
                global_id = self.gallery.assign_to_global(
                    camera_id,
                    local_track_id,
                    existing_global_id,
                    embedding,
                    timestamp=timestamp,
                )
                update_reason = "existing_local_track_verified"
            else:
                global_id = existing_global_id
                update_reason = "existing_local_track_hold_no_gallery_update"

            self.pending_identity_tracks.pop((camera_id, local_track_id), None)
            return global_id, {
                **preview_info,
                "matched": True,
                "reason": update_reason,
                "global_id": global_id,
                "assigned_distance": assigned_distance,
                "candidate_global_id": preview_info.get("candidate_global_id"),
                "candidate_distance": candidate_distance,
            }

        if not self.split_existing_track_on_drift:
            self.pending_identity_tracks.pop((camera_id, local_track_id), None)
            match_info = {
                **preview_info,
                "matched": True,
                "reason": "existing_local_track_drift_hold",
                "global_id": existing_global_id,
                "assigned_distance": assigned_distance,
                "candidate_global_id": preview_info.get("candidate_global_id"),
                "candidate_distance": candidate_distance,
            }
            if is_remap_candidate and not allow_existing_track_remap:
                match_info["remap_suppressed_reason"] = "edge_partial"
            return existing_global_id, match_info

        return self.pending_identity_response(
            camera_id,
            local_track_id,
            frame_id,
            timestamp,
            preview_info,
            "pending_identity_drift",
            embedding,
            previous_global_id=existing_global_id,
            assigned_distance=assigned_distance,
        )

    def handle_low_quality_observation(
        self,
        camera_id: str,
        local_track_id: int,
        existing_global_id: int | None,
        frame_global_ids: set[int],
        quality_info: dict[str, Any],
    ) -> tuple[int | None, dict[str, Any]]:
        if existing_global_id is not None and existing_global_id not in frame_global_ids:
            self.pending_identity_tracks.pop((camera_id, local_track_id), None)
            return existing_global_id, {
                "matched": True,
                "reason": "low_quality_existing_track_hold",
                "global_id": existing_global_id,
                "quality": quality_info,
            }
        return None, {
            "matched": False,
            "reason": "low_quality_no_identity",
            "global_id": None,
            "quality": quality_info,
        }

    def hold_pending_edge_observation(
        self,
        camera_id: str,
        local_track_id: int,
        frame_id: int,
        embedding: np.ndarray,
        timestamp: float,
        blocked_global_ids: set[int],
        cannot_link_global_ids: set[int],
        quality_info: dict[str, Any],
        appearance_lightness: float | None,
    ) -> tuple[None, dict[str, Any]]:
        """Accumulate edge evidence without assigning or updating the gallery."""
        pending_frames, pending_embedding, pending_seconds = self.update_pending_identity_track(
            camera_id,
            local_track_id,
            frame_id,
            timestamp,
            embedding,
            appearance_lightness=appearance_lightness,
        )
        pending_state = self.pending_identity_tracks[(camera_id, int(local_track_id))]
        if pending_state.get("last_edge_frame_id") != int(frame_id):
            pending_state["edge_frames"] = int(pending_state.get("edge_frames", 0)) + 1
        pending_state["last_edge_frame_id"] = int(frame_id)
        persistent_cannot_links = pending_state.setdefault(
            "cannot_link_global_ids",
            set(),
        )
        persistent_cannot_links.update(cannot_link_global_ids)
        persistent_cannot_links.update(
            self.pending_overlap_cannot_link_global_ids(
                camera_id,
                local_track_id,
                pending_embedding,
                timestamp,
            )
        )
        blocked = set(blocked_global_ids) | set(persistent_cannot_links)
        self.observe_pending_overlap_identity_candidates(
            camera_id,
            pending_state,
            timestamp,
            blocked,
        )
        _, preview_info = self.gallery.preview_match(
            pending_embedding,
            timestamp=timestamp,
            blocked_global_ids=blocked,
        )
        return None, {
            **preview_info,
            "matched": False,
            "reason": "pending_edge_identity",
            "global_id": None,
            "pending_frames": pending_frames,
            "pending_seconds": pending_seconds,
            "pending_lightness": self.pending_track_lightness(pending_state),
            "edge_frames": int(pending_state.get("edge_frames", 0)),
            "cannot_link_global_ids": sorted(persistent_cannot_links),
            "quality": quality_info,
        }

    def identity_observation_quality(
        self,
        track: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> tuple[bool, dict[str, Any]]:
        x1, y1, x2, y2 = [float(value) for value in track[:4]]
        confidence = float(track[5])
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        frame_area = max(1.0, float(frame_width * frame_height))
        height_ratio = box_height / max(1.0, float(frame_height))
        area_ratio = (box_width * box_height) / frame_area
        edge_margin_x = float(frame_width) * self.identity_edge_margin_ratio
        edge_margin_y = float(frame_height) * self.identity_edge_margin_ratio
        touches_edge = (
            x1 <= edge_margin_x
            or y1 <= edge_margin_y
            or x2 >= float(frame_width) - edge_margin_x
            or y2 >= float(frame_height) - edge_margin_y
        )

        reasons = []
        if confidence < self.identity_min_confidence:
            reasons.append("low_confidence")
        if height_ratio < self.identity_min_height_ratio:
            reasons.append("small_height")
        if area_ratio < self.identity_min_area_ratio:
            reasons.append("small_area")
        if touches_edge and confidence < self.identity_edge_min_confidence:
            reasons.append("edge_partial_low_confidence")

        info = {
            "ok": not reasons,
            "reasons": reasons,
            "confidence": confidence,
            "height_ratio": height_ratio,
            "area_ratio": area_ratio,
            "touches_edge": touches_edge,
            "min_confidence": self.identity_min_confidence,
            "min_height_ratio": self.identity_min_height_ratio,
            "min_area_ratio": self.identity_min_area_ratio,
            "edge_min_confidence": self.identity_edge_min_confidence,
        }
        return not reasons, info

    def is_reliable_new_track_match(self, match_info: dict[str, Any]) -> bool:
        """Return true when a new local track should attach to an existing global ID."""
        best_distance = match_info.get("best_distance")
        if best_distance is None or best_distance > self.new_track_match_threshold:
            return False

        second_distance = match_info.get("second_distance")
        if second_distance is None:
            return float(best_distance) <= self.new_track_single_candidate_threshold
        return float(second_distance) - float(best_distance) >= self.new_track_match_margin

    def pending_identity_response(
        self,
        camera_id: str,
        local_track_id: int,
        frame_id: int,
        timestamp: float,
        preview_info: dict[str, Any],
        reason: str,
        embedding: np.ndarray,
        previous_global_id: int | None = None,
        assigned_distance: float | None = None,
        blocked_global_ids: set[int] | None = None,
        cannot_link_global_ids: set[int] | None = None,
        appearance_lightness: float | None = None,
    ) -> tuple[int | None, dict[str, Any]]:
        """Hold an uncertain track before creating a visible global ID."""
        pending_frames, pending_embedding, pending_seconds = self.update_pending_identity_track(
            camera_id,
            local_track_id,
            frame_id,
            timestamp,
            embedding,
            appearance_lightness=appearance_lightness,
        )
        pending_key = (camera_id, int(local_track_id))
        pending_state = self.pending_identity_tracks[pending_key]
        pending_lightness = self.pending_track_lightness(pending_state)
        persistent_cannot_links = pending_state.setdefault(
            "cannot_link_global_ids",
            set(),
        )
        persistent_cannot_links.update(cannot_link_global_ids or set())
        propagated_cannot_links = self.pending_overlap_cannot_link_global_ids(
            camera_id,
            local_track_id,
            pending_embedding,
            timestamp,
        )
        persistent_cannot_links.update(propagated_cannot_links)
        if (
            pending_frames < self.new_identity_min_frames
            or pending_seconds + 1e-9 < self.new_identity_min_seconds
        ):
            return None, {
                **preview_info,
                "matched": False,
                "reason": reason,
                "global_id": None,
                "previous_global_id": previous_global_id,
                "assigned_distance": assigned_distance,
                "pending_frames": pending_frames,
                "required_frames": self.new_identity_min_frames,
                "pending_seconds": pending_seconds,
                "required_seconds": self.new_identity_min_seconds,
                "pending_lightness": pending_lightness,
                "cannot_link_global_ids": sorted(persistent_cannot_links),
            }

        confirmation_info = preview_info
        if reason == "pending_new_identity":
            blocked = (
                set(blocked_global_ids)
                if blocked_global_ids is not None
                else set(preview_info.get("blocked_global_ids") or [])
            )
            blocked.update(persistent_cannot_links)
            matched_global_id, confirmation_info = self.gallery.preview_match(
                pending_embedding,
                timestamp=timestamp,
                blocked_global_ids=blocked,
            )
            if (
                matched_global_id is not None
                and self.is_reliable_new_track_match(confirmation_info)
            ):
                global_id = self.gallery.assign_to_global(
                    camera_id,
                    local_track_id,
                    matched_global_id,
                    pending_embedding,
                    timestamp=timestamp,
                )
                self.pending_identity_tracks.pop((camera_id, local_track_id), None)
                return global_id, {
                    **confirmation_info,
                    "matched": True,
                    "reason": "appearance_match_verified_pending_average",
                    "global_id": global_id,
                    "confirmation_frames": pending_frames,
                    "confirmation_seconds": pending_seconds,
                }

            overlap_global_id, overlap_info = self.reliable_active_overlap_match(
                camera_id,
                pending_embedding,
                timestamp,
                blocked,
            )
            if overlap_global_id is not None:
                if pending_seconds + 1e-9 < self.overlap_new_track_min_seconds:
                    return None, {
                        **overlap_info,
                        "matched": False,
                        "reason": "pending_active_overlap_confirmation",
                        "global_id": None,
                        "previous_global_id": previous_global_id,
                        "assigned_distance": assigned_distance,
                        "pending_frames": pending_frames,
                        "required_frames": self.new_identity_min_frames,
                        "pending_seconds": pending_seconds,
                        "required_seconds": self.overlap_new_track_min_seconds,
                    }
                global_id = self.gallery.assign_to_global(
                    camera_id,
                    local_track_id,
                    overlap_global_id,
                    pending_embedding,
                    timestamp=timestamp,
                )
                self.pending_identity_tracks.pop((camera_id, local_track_id), None)
                return global_id, {
                    **overlap_info,
                    "matched": True,
                    "reason": "appearance_match_verified_active_overlap",
                    "global_id": global_id,
                    "confirmation_frames": pending_frames,
                    "confirmation_seconds": pending_seconds,
                }

            lightness_global_id, lightness_info = (
                self.reliable_active_overlap_lightness_match(
                    camera_id,
                    pending_embedding,
                    pending_lightness,
                    pending_state,
                    pending_seconds,
                    timestamp,
                    blocked,
                )
            )
            if lightness_global_id is not None:
                global_id = self.gallery.assign_to_global(
                    camera_id,
                    local_track_id,
                    lightness_global_id,
                    embedding,
                    timestamp=timestamp,
                )
                self.pending_identity_tracks.pop((camera_id, local_track_id), None)
                return global_id, {
                    **lightness_info,
                    "matched": True,
                    "reason": "appearance_match_verified_overlap_lightness",
                    "global_id": global_id,
                    "confirmation_frames": pending_frames,
                    "confirmation_seconds": pending_seconds,
                    "gallery_updated": True,
                    "gallery_update_source": "current_interior_embedding",
                }
            confirmation_info = {
                **confirmation_info,
                "overlap_lightness": lightness_info,
            }

            adjacent_global_id, adjacent_info = self.reliable_adjacent_handoff_match(
                camera_id,
                pending_embedding,
                timestamp,
                blocked,
            )
            if adjacent_global_id is not None:
                global_id = self.gallery.assign_to_global(
                    camera_id,
                    local_track_id,
                    adjacent_global_id,
                    pending_embedding,
                    timestamp=timestamp,
                )
                self.pending_identity_tracks.pop((camera_id, local_track_id), None)
                return global_id, {
                    **adjacent_info,
                    "matched": True,
                    "reason": "appearance_match_verified_adjacent_handoff",
                    "global_id": global_id,
                    "confirmation_frames": pending_frames,
                    "confirmation_seconds": pending_seconds,
                }

            if (
                pending_seconds + 1e-9 < self.new_identity_ambiguous_wait_seconds
                and self.is_ambiguous_new_identity(confirmation_info)
            ):
                return None, {
                    **confirmation_info,
                    "matched": False,
                    "reason": "pending_ambiguous_identity",
                    "global_id": None,
                    "previous_global_id": previous_global_id,
                    "assigned_distance": assigned_distance,
                    "pending_frames": pending_frames,
                    "required_frames": self.new_identity_min_frames,
                    "pending_seconds": pending_seconds,
                    "required_seconds": self.new_identity_ambiguous_wait_seconds,
                    "ambiguous_distance": self.new_identity_ambiguous_distance,
                    "cannot_link_global_ids": sorted(persistent_cannot_links),
                }

        confirmed_reason = {
            "pending_new_identity": "new_identity",
            "pending_identity_drift": "new_identity_identity_drift",
            "pending_frame_collision": "new_identity_frame_collision",
        }.get(reason, f"new_identity_{reason}")
        global_id = self.gallery.create_new_identity(
            camera_id,
            local_track_id,
            pending_embedding,
            timestamp=timestamp,
            match_info={
                **confirmation_info,
                "confirmation_frames": pending_frames,
                "confirmation_seconds": pending_seconds,
            },
            reason=confirmed_reason,
        )
        self.pending_identity_tracks.pop((camera_id, local_track_id), None)
        return global_id, self.gallery.last_match

    def pending_overlap_cannot_link_global_ids(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: np.ndarray,
        timestamp: float,
    ) -> set[int]:
        """Propagate cannot-link evidence between strongly matching pending views."""
        if self.pending_overlap_link_threshold < 0:
            return set()

        normalized = self.gallery._normalize(embedding)
        propagated = set()
        for (other_camera, other_local_id), state in self.pending_identity_tracks.items():
            if other_camera == camera_id and other_local_id == int(local_track_id):
                continue
            if not self.camera_topology.may_overlap(camera_id, other_camera):
                continue
            age = timestamp - float(state.get("last_seen", timestamp))
            if age < 0.0 or age > self.pending_overlap_link_active_seconds:
                continue
            cannot_links = set(state.get("cannot_link_global_ids") or set())
            embedding_sum = state.get("embedding_sum")
            if not cannot_links or embedding_sum is None:
                continue
            other_embedding = self.gallery._normalize(embedding_sum)
            distance = 1.0 - float(np.dot(normalized, other_embedding))
            if distance <= self.pending_overlap_link_threshold:
                propagated.update(cannot_links)
        return propagated

    @staticmethod
    def pending_track_lightness(state: dict[str, Any]) -> float | None:
        count = int(state.get("lightness_count", 0))
        if count <= 0:
            return None
        return float(state.get("lightness_sum", 0.0)) / count

    def observe_pending_overlap_identity_candidates(
        self,
        camera_id: str,
        pending_state: dict[str, Any],
        timestamp: float,
        blocked_global_ids: set[int],
    ) -> None:
        """Record sustained co-visibility while an edge track remains unassigned."""
        if not self.overlap_lightness_fallback_enabled:
            return
        self.prune_identity_locations(timestamp)
        evidence = pending_state.setdefault("overlap_identity_evidence", {})
        for global_id, locations in self.identity_locations.items():
            if global_id in blocked_global_ids or global_id not in self.gallery.entries:
                continue
            source_last_seen = max(
                (
                    last_seen
                    for other_camera, last_seen in locations.items()
                    if other_camera != camera_id
                    and self.camera_topology.may_overlap(camera_id, other_camera)
                    and 0.0
                    <= timestamp - last_seen
                    <= self.overlap_candidate_active_seconds
                ),
                default=None,
            )
            if source_last_seen is None:
                continue
            candidate = evidence.get(int(global_id))
            if candidate is None:
                evidence[int(global_id)] = {
                    "first_seen": float(source_last_seen),
                    "last_seen": float(source_last_seen),
                }
            else:
                candidate["last_seen"] = max(
                    float(candidate["last_seen"]),
                    float(source_last_seen),
                )

    def reliable_active_overlap_lightness_match(
        self,
        camera_id: str,
        embedding: np.ndarray,
        lightness: float | None,
        pending_state: dict[str, Any],
        pending_seconds: float,
        timestamp: float,
        blocked_global_ids: set[int],
    ) -> tuple[int | None, dict[str, Any]]:
        """Recover a long edge-clipped overlap using conservative clothing brightness."""
        policy = "active_camera_overlap_lightness"
        edge_frames = int(pending_state.get("edge_frames", 0))
        if (
            not self.overlap_lightness_fallback_enabled
            or lightness is None
            or pending_seconds + 1e-9 < self.overlap_lightness_min_seconds
            or edge_frames < self.overlap_lightness_min_edge_frames
        ):
            return None, {
                "candidate_global_id": None,
                "match_policy": policy,
                "enabled": self.overlap_lightness_fallback_enabled,
                "pending_lightness": lightness,
                "pending_seconds": pending_seconds,
                "required_pending_seconds": self.overlap_lightness_min_seconds,
                "edge_frames": edge_frames,
                "required_edge_frames": self.overlap_lightness_min_edge_frames,
            }

        candidates = []
        evidence = pending_state.get("overlap_identity_evidence") or {}
        for raw_global_id, observation in evidence.items():
            global_id = int(raw_global_id)
            if global_id in blocked_global_ids or global_id not in self.gallery.entries:
                continue
            first_seen = float(observation["first_seen"])
            last_seen = float(observation["last_seen"])
            covisible_seconds = max(0.0, last_seen - first_seen)
            if (
                covisible_seconds + 1e-9 < self.overlap_lightness_min_covisible_seconds
                or timestamp - last_seen > self.overlap_candidate_active_seconds
            ):
                continue
            identity_lightness = self.identity_lightness(global_id)
            embedding_distance = self.gallery.distance_to_global(global_id, embedding)
            if identity_lightness is None or embedding_distance is None:
                continue
            lightness_distance = abs(float(lightness) - identity_lightness)
            if (
                embedding_distance > self.overlap_lightness_max_embedding_distance
                or lightness_distance > self.overlap_lightness_max_distance
            ):
                continue
            score = embedding_distance + self.overlap_lightness_score_weight * lightness_distance
            candidates.append(
                (
                    float(score),
                    int(global_id),
                    float(embedding_distance),
                    float(lightness_distance),
                    float(identity_lightness),
                    covisible_seconds,
                )
            )

        candidates.sort(key=lambda item: (item[0], item[1]))
        if not candidates:
            return None, {
                "candidate_global_id": None,
                "match_policy": policy,
                "pending_lightness": lightness,
                "blocked_global_ids": sorted(blocked_global_ids),
            }
        best = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else None
        margin_ok = (
            second_score is None
            or second_score - best[0] >= self.overlap_lightness_match_margin
        )
        info = {
            "candidate_global_id": best[1],
            "best_distance": best[2],
            "second_score": second_score,
            "combined_score": best[0],
            "required_score_margin": self.overlap_lightness_match_margin,
            "pending_lightness": lightness,
            "candidate_lightness": best[4],
            "lightness_distance": best[3],
            "max_lightness_distance": self.overlap_lightness_max_distance,
            "max_embedding_distance": self.overlap_lightness_max_embedding_distance,
            "covisible_seconds": best[5],
            "required_covisible_seconds": self.overlap_lightness_min_covisible_seconds,
            "match_policy": policy,
            "blocked_global_ids": sorted(blocked_global_ids),
        }
        return (best[1], info) if margin_ok else (None, info)

    def reliable_active_overlap_match(
        self,
        camera_id: str,
        embedding: np.ndarray,
        timestamp: float,
        blocked_global_ids: set[int],
    ) -> tuple[int | None, dict[str, Any]]:
        """Match a confirmed track to identities active in known overlapping views."""
        self.prune_identity_locations(timestamp)
        candidates = []
        for global_id, locations in self.identity_locations.items():
            if global_id in blocked_global_ids or global_id not in self.gallery.entries:
                continue
            overlapping_cameras = sorted(
                other_camera
                for other_camera, last_seen in locations.items()
                if other_camera != camera_id
                and self.camera_topology.may_overlap(camera_id, other_camera)
                and 0.0 <= timestamp - last_seen <= self.overlap_candidate_active_seconds
            )
            if not overlapping_cameras:
                continue
            distance = self.gallery.distance_to_global(global_id, embedding)
            if distance is not None:
                candidates.append((float(distance), int(global_id), overlapping_cameras))

        candidates.sort(key=lambda item: (item[0], item[1]))
        if not candidates:
            return None, {
                "candidate_global_id": None,
                "best_distance": None,
                "second_distance": None,
                "threshold": self.overlap_new_track_match_threshold,
                "match_policy": "active_camera_overlap",
                "blocked_global_ids": sorted(blocked_global_ids),
            }

        best_distance, best_global_id, best_cameras = candidates[0]
        second_distance = candidates[1][0] if len(candidates) > 1 else None
        margin_ok = (
            second_distance is None
            or second_distance - best_distance >= self.overlap_new_track_match_margin
        )
        info = {
            "candidate_global_id": best_global_id,
            "best_distance": best_distance,
            "second_distance": second_distance,
            "threshold": self.overlap_new_track_match_threshold,
            "required_margin": self.overlap_new_track_match_margin,
            "candidate_cameras": best_cameras,
            "candidate_active_seconds": self.overlap_candidate_active_seconds,
            "required_confirmation_seconds": self.overlap_new_track_min_seconds,
            "match_policy": "active_camera_overlap",
            "blocked_global_ids": sorted(blocked_global_ids),
        }
        if best_distance > self.overlap_new_track_match_threshold or not margin_ok:
            return None, info
        return best_global_id, info

    def reliable_adjacent_handoff_match(
        self,
        camera_id: str,
        embedding: np.ndarray,
        timestamp: float,
        blocked_global_ids: set[int],
    ) -> tuple[int | None, dict[str, Any]]:
        """Match a confirmed track to a recent identity on an adjacent camera."""
        self.prune_identity_locations(timestamp)
        candidates = []
        for global_id, locations in self.identity_locations.items():
            if global_id in blocked_global_ids or global_id not in self.gallery.entries:
                continue
            source_cameras = sorted(
                other_camera
                for other_camera, last_seen in locations.items()
                if self.camera_topology.are_adjacent(camera_id, other_camera)
                and self.adjacent_camera_exclusion_seconds
                < timestamp - last_seen
                <= self.adjacent_candidate_max_seconds
            )
            if not source_cameras:
                continue
            distance = self.gallery.distance_to_global(global_id, embedding)
            if distance is not None:
                candidates.append((float(distance), int(global_id), source_cameras))

        candidates.sort(key=lambda item: (item[0], item[1]))
        if not candidates:
            return None, {
                "candidate_global_id": None,
                "best_distance": None,
                "second_distance": None,
                "threshold": self.adjacent_new_track_match_threshold,
                "match_policy": "adjacent_camera_handoff",
                "blocked_global_ids": sorted(blocked_global_ids),
            }

        best_distance, best_global_id, source_cameras = candidates[0]
        second_distance = candidates[1][0] if len(candidates) > 1 else None
        margin_ok = (
            second_distance is None
            or second_distance - best_distance >= self.adjacent_new_track_match_margin
        )
        info = {
            "candidate_global_id": best_global_id,
            "best_distance": best_distance,
            "second_distance": second_distance,
            "threshold": self.adjacent_new_track_match_threshold,
            "required_margin": self.adjacent_new_track_match_margin,
            "source_cameras": source_cameras,
            "candidate_max_seconds": self.adjacent_candidate_max_seconds,
            "match_policy": "adjacent_camera_handoff",
            "blocked_global_ids": sorted(blocked_global_ids),
        }
        if best_distance > self.adjacent_new_track_match_threshold or not margin_ok:
            return None, info
        return best_global_id, info

    def update_pending_identity_track(
        self,
        camera_id: str,
        local_track_id: int,
        frame_id: int,
        timestamp: float,
        embedding: np.ndarray,
        appearance_lightness: float | None = None,
    ) -> tuple[int, np.ndarray, float]:
        key = (camera_id, int(local_track_id))
        state = self.pending_identity_tracks.setdefault(
            key,
            {
                "frames": 0,
                "last_frame_id": None,
                "first_seen": timestamp,
                "last_seen": timestamp,
                "embedding_sum": None,
            },
        )
        if state.get("last_frame_id") != frame_id:
            state["frames"] = int(state.get("frames", 0)) + 1
            normalized = self.gallery._normalize(embedding)
            embedding_sum = state.get("embedding_sum")
            state["embedding_sum"] = (
                normalized.copy()
                if embedding_sum is None
                else np.asarray(embedding_sum, dtype=np.float32) + normalized
            )
            if appearance_lightness is not None:
                state["lightness_sum"] = float(state.get("lightness_sum", 0.0)) + float(
                    appearance_lightness
                )
                state["lightness_count"] = int(state.get("lightness_count", 0)) + 1
        state["last_frame_id"] = int(frame_id)
        state["last_seen"] = timestamp
        averaged = self.gallery._normalize(state["embedding_sum"])
        pending_seconds = max(0.0, timestamp - float(state.get("first_seen", timestamp)))
        return int(state["frames"]), averaged, pending_seconds

    def is_ambiguous_new_identity(self, match_info: dict[str, Any]) -> bool:
        distances = [
            match_info.get("best_distance"),
            match_info.get("blocked_candidate_distance"),
        ]
        nearest = min(
            (float(distance) for distance in distances if distance is not None),
            default=None,
        )
        return nearest is not None and nearest <= self.new_identity_ambiguous_distance

    def prune_pending_identity_tracks(self, timestamp: float) -> None:
        expired = [
            key
            for key, state in self.pending_identity_tracks.items()
            if timestamp - float(state.get("last_seen", 0.0)) > self.pending_identity_ttl_seconds
        ]
        for key in expired:
            del self.pending_identity_tracks[key]

    def blocked_global_ids_for_camera(self, camera_id: str, timestamp: float) -> set[int]:
        """Block identities observed recently in another non-overlapping camera."""
        if (
            self.cross_camera_exclusion_seconds <= 0
            and self.adjacent_camera_exclusion_seconds <= 0
        ):
            return set()

        self.prune_identity_locations(timestamp)
        blocked = set()
        for global_id, locations in self.identity_locations.items():
            if any(
                other_camera != camera_id
                and not self.camera_topology.may_overlap(camera_id, other_camera)
                and timestamp - last_seen
                <= (
                    self.adjacent_camera_exclusion_seconds
                    if self.camera_topology.are_adjacent(camera_id, other_camera)
                    else self.cross_camera_exclusion_seconds
                )
                for other_camera, last_seen in locations.items()
            ):
                blocked.add(global_id)
        return blocked

    def cameras_may_overlap(self, first_camera: str, second_camera: str) -> bool:
        return self.camera_topology.may_overlap(first_camera, second_camera)

    def record_identity_location(self, global_id: int, camera_id: str, timestamp: float) -> None:
        locations = self.identity_locations.setdefault(int(global_id), {})
        locations[camera_id] = max(float(timestamp), float(locations.get(camera_id, timestamp)))

    def prune_identity_locations(self, timestamp: float) -> None:
        retention_seconds = max(
            self.cross_camera_exclusion_seconds,
            self.overlap_candidate_active_seconds,
            self.adjacent_camera_exclusion_seconds,
            self.adjacent_candidate_max_seconds,
        )
        cutoff = float(timestamp) - retention_seconds
        for global_id in list(self.identity_locations):
            locations = self.identity_locations[global_id]
            self.identity_locations[global_id] = {
                camera_id: last_seen
                for camera_id, last_seen in locations.items()
                if last_seen >= cutoff
            }
            if not self.identity_locations[global_id]:
                del self.identity_locations[global_id]

    def write_reid_debug_event(
        self,
        camera_id: str,
        frame_id: int,
        track: np.ndarray,
        det_idx: int,
        global_id: int | None,
        embedding: np.ndarray,
        match_info: dict[str, Any] | None,
        crop: np.ndarray | None,
        timestamp: float,
    ) -> None:
        crop_path = None
        should_save_crop = frame_id % self.debug_crop_interval_frames == 0
        if self.save_debug_crops and should_save_crop and crop is not None:
            crop_dir = self.debug_dir / camera_id
            crop_dir.mkdir(parents=True, exist_ok=True)
            gid_label = global_id if global_id is not None else "pending"
            timestamp_ms = int(timestamp * 1000)
            crop_path = (
                crop_dir
                / f"t_{timestamp_ms}_frame_{frame_id:08d}_det_{det_idx:02d}_gid_{gid_label}.jpg"
            )
            write_jpeg(crop_path, crop)

        event = {
            "time": timestamp,
            "camera_id": camera_id,
            "frame_id": int(frame_id),
            "local_track_id": int(track[4]),
            "global_id": int(global_id) if global_id is not None else None,
            "det_idx": int(det_idx),
            "bbox": [float(x) for x in track[:4]],
            "confidence": float(track[5]),
            "embedding_norm": float(np.linalg.norm(embedding)),
            "match": match_info,
            "crop_path": str(crop_path) if crop_path is not None else None,
        }
        with self.debug_events_path.open("a") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
