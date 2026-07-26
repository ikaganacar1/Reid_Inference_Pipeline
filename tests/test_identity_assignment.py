from pathlib import Path

import numpy as np
import pytest

from src.realtime.identity_assignment import GlobalIdentityAssigner


def config(**overrides):
    values = {
        "debug_reid": False,
        "new_identity_min_frames": 1,
        "new_identity_min_seconds": 0.0,
        "new_identity_ambiguous_wait_seconds": 0.0,
        "duplicate_local_track_iou_threshold": 1.01,
        "global_match_threshold": 0.3,
        "new_track_match_threshold": 0.3,
        "new_track_single_candidate_threshold": 0.16,
        "new_track_match_margin": 0.08,
        "identity_min_confidence": 0.0,
        "identity_min_height_ratio": 0.0,
        "identity_min_area_ratio": 0.0,
        "identity_edge_min_confidence": 0.0,
        "cross_camera_exclusion_seconds": 1.0,
    }
    values.update(overrides)
    return values


def track(local_id, det_idx, confidence=0.9):
    return np.array([10, 10, 50, 100, local_id, confidence, 0, det_idx], dtype=np.float32)


def edge_track(local_id, det_idx, confidence=0.9):
    return np.array([0, 10, 50, 100, local_id, confidence, 0, det_idx], dtype=np.float32)


def test_existing_tracks_reserve_ids_before_new_tracks(tmp_path: Path):
    assigner = GlobalIdentityAssigner(config(new_identity_min_frames=2), tmp_path)
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    gid = assigner.gallery.create_new_identity("cam1", 1, embedding, timestamp=1.0)

    records = assigner.assign_tracks(
        "cam1",
        2,
        100,
        120,
        np.stack([track(2, 0, 0.99), track(1, 1, 0.8)]),
        np.stack([embedding, embedding]),
        timestamp=1.1,
    )

    assert records[0]["global_id"] is None
    assert records[0]["match"]["blocked_candidate_global_id"] == gid
    assert records[1]["global_id"] == gid


def test_existing_track_cannot_steal_another_visible_tracks_id(tmp_path: Path):
    assigner = GlobalIdentityAssigner(config(existing_track_max_distance=0.3), tmp_path)
    first = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    second = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    gid1 = assigner.gallery.create_new_identity("cam1", 1, first, timestamp=1.0)
    gid2 = assigner.gallery.create_new_identity("cam1", 2, second, timestamp=1.0)

    # Track 1's crop is corrupted by overlap and looks exactly like track 2.
    records = assigner.assign_tracks(
        "cam1",
        2,
        100,
        120,
        np.stack([track(1, 0, 0.99), track(2, 1, 0.8)]),
        np.stack([second, second]),
        timestamp=1.1,
    )

    assert records[0]["global_id"] == gid1
    assert records[0]["match"]["reason"] == "existing_local_track_drift_hold"
    assert records[0]["match"]["blocked_candidate_global_id"] == gid2
    assert records[1]["global_id"] == gid2


def test_reset_camera_mutates_pending_state_in_place(tmp_path: Path):
    assigner = GlobalIdentityAssigner(config(), tmp_path)
    pending_state = assigner.pending_identity_tracks
    assigner.pending_identity_tracks[("cam1", 1)] = {"last_seen": 1.0}
    assigner.pending_identity_tracks[("cam2", 2)] = {"last_seen": 1.0}

    assigner.reset_camera("cam1")

    assert assigner.pending_identity_tracks is pending_state
    assert ("cam1", 1) not in pending_state
    assert ("cam2", 2) in pending_state


def test_recent_other_camera_location_blocks_then_expires(tmp_path: Path):
    assigner = GlobalIdentityAssigner(config(new_identity_min_frames=2), tmp_path)
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    gid = assigner.gallery.create_new_identity("cam1", 1, embedding, timestamp=1.0)
    assigner.record_identity_location(gid, "cam1", 1.0)

    blocked = assigner.assign_tracks(
        "cam2", 1, 100, 120, np.stack([track(1, 0)]), np.stack([embedding]), timestamp=1.5
    )
    allowed = assigner.assign_tracks(
        "cam2", 2, 100, 120, np.stack([track(1, 0)]), np.stack([embedding]), timestamp=2.1
    )

    assert blocked[0]["global_id"] is None
    assert allowed[0]["global_id"] == gid


def test_configured_overlapping_camera_pair_allows_simultaneous_identity(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(
            new_identity_min_frames=2,
            overlapping_camera_pairs=[["cam1", "cam2"]],
        ),
        tmp_path,
    )
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    gid = assigner.gallery.create_new_identity("cam1", 1, embedding, timestamp=1.0)
    assigner.record_identity_location(gid, "cam1", 1.0)

    pending_overlap = assigner.assign_tracks(
        "cam2", 1, 100, 120, np.stack([track(1, 0)]), np.stack([embedding]), timestamp=1.0
    )
    overlapping = assigner.assign_tracks(
        "cam2", 2, 100, 120, np.stack([track(1, 0)]), np.stack([embedding]), timestamp=1.1
    )
    disjoint = assigner.assign_tracks(
        "cam3", 1, 100, 120, np.stack([track(1, 0)]), np.stack([embedding]), timestamp=1.1
    )

    assert pending_overlap[0]["global_id"] is None
    assert overlapping[0]["global_id"] == gid
    assert disjoint[0]["global_id"] is None


def test_pending_track_uses_relaxed_gate_only_for_active_overlap_candidate(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(
            new_identity_min_frames=2,
            cross_camera_exclusion_seconds=0,
            overlapping_camera_pairs=[["cam1", "cam2"]],
            overlap_new_track_match_threshold=0.35,
            overlap_new_track_match_margin=0.04,
        ),
        tmp_path,
    )
    target = np.array([1.0, 0.0], dtype=np.float32)
    distractor = np.array([0.087, 0.996], dtype=np.float32)
    observation = np.array([0.75, 0.6614378], dtype=np.float32)
    target_id = assigner.gallery.create_new_identity("cam1", 1, target, timestamp=1.0)
    distractor_id = assigner.gallery.create_new_identity("cam3", 1, distractor, timestamp=1.0)
    assigner.record_identity_location(target_id, "cam1", 1.0)
    assigner.record_identity_location(distractor_id, "cam3", 1.0)

    pending = assigner.assign_tracks(
        "cam2", 1, 100, 120, np.stack([track(1, 0)]), np.stack([observation]), timestamp=1.1
    )
    confirmed = assigner.assign_tracks(
        "cam2", 2, 100, 120, np.stack([track(1, 0)]), np.stack([observation]), timestamp=1.2
    )

    assert pending[0]["global_id"] is None
    assert confirmed[0]["global_id"] == target_id
    assert confirmed[0]["match"]["reason"] == "appearance_match_verified_active_overlap"
    assert confirmed[0]["match"]["candidate_cameras"] == ["cam1"]


def test_relaxed_overlap_match_waits_for_configured_confirmation_time(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(
            new_identity_min_frames=2,
            cross_camera_exclusion_seconds=0,
            overlapping_camera_pairs=[["cam1", "cam2"]],
            overlap_new_track_match_threshold=0.35,
            overlap_new_track_match_margin=0.04,
            overlap_candidate_active_seconds=2.0,
            overlap_new_track_min_seconds=1.0,
        ),
        tmp_path,
    )
    target = np.array([1.0, 0.0], dtype=np.float32)
    distractor = np.array([0.087, 0.996], dtype=np.float32)
    observation = np.array([0.75, 0.6614378], dtype=np.float32)
    target_id = assigner.gallery.create_new_identity("cam1", 1, target, timestamp=1.0)
    distractor_id = assigner.gallery.create_new_identity("cam3", 1, distractor, timestamp=1.0)
    assigner.record_identity_location(target_id, "cam1", 1.0)
    assigner.record_identity_location(distractor_id, "cam3", 1.0)

    assigner.assign_tracks(
        "cam2", 1, 100, 120, np.stack([track(1, 0)]), np.stack([observation]), timestamp=1.1
    )
    waiting = assigner.assign_tracks(
        "cam2", 2, 100, 120, np.stack([track(1, 0)]), np.stack([observation]), timestamp=1.2
    )
    confirmed = assigner.assign_tracks(
        "cam2", 3, 100, 120, np.stack([track(1, 0)]), np.stack([observation]), timestamp=2.1
    )

    assert waiting[0]["global_id"] is None
    assert waiting[0]["match"]["reason"] == "pending_active_overlap_confirmation"
    assert waiting[0]["match"]["required_seconds"] == 1.0
    assert confirmed[0]["global_id"] == target_id
    assert confirmed[0]["match"]["reason"] == "appearance_match_verified_active_overlap"


def test_pending_track_can_use_confirmed_adjacent_camera_handoff(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(
            new_identity_min_frames=2,
            adjacent_camera_pairs=[["cam1", "cam2"]],
            adjacent_camera_exclusion_seconds=0.25,
            adjacent_candidate_max_seconds=5.0,
        ),
        tmp_path,
    )
    target = np.array([1.0, 0.0], dtype=np.float32)
    observation = np.array([0.75, 0.6614378], dtype=np.float32)
    target_id = assigner.gallery.create_new_identity("cam1", 1, target, timestamp=1.0)
    assigner.record_identity_location(target_id, "cam1", 1.0)

    pending = assigner.assign_tracks(
        "cam2", 1, 100, 120, np.stack([track(1, 0)]), np.stack([observation]), timestamp=1.3
    )
    confirmed = assigner.assign_tracks(
        "cam2", 2, 100, 120, np.stack([track(1, 0)]), np.stack([observation]), timestamp=1.4
    )

    assert pending[0]["global_id"] is None
    assert confirmed[0]["global_id"] == target_id
    assert confirmed[0]["match"]["reason"] == "appearance_match_verified_adjacent_handoff"
    assert confirmed[0]["match"]["source_cameras"] == ["cam1"]


def test_pending_identity_uses_average_embedding(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(new_identity_min_frames=2, cross_camera_exclusion_seconds=0),
        tmp_path,
    )
    first = np.array([1.0, 0.0], dtype=np.float32)
    second = np.array([0.0, 1.0], dtype=np.float32)

    assigner.assign_tracks(
        "cam1", 1, 100, 120, np.stack([track(1, 0)]), np.stack([first]), timestamp=1.0
    )
    records = assigner.assign_tracks(
        "cam1", 2, 100, 120, np.stack([track(1, 0)]), np.stack([second]), timestamp=1.1
    )

    gid = records[0]["global_id"]
    expected = np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0)
    np.testing.assert_allclose(assigner.gallery.entries[gid].embedding, expected, atol=1e-6)


def test_existing_identity_match_waits_for_new_track_confirmation(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(
            new_identity_min_frames=2,
            new_identity_min_seconds=0.4,
            cross_camera_exclusion_seconds=0,
        ),
        tmp_path,
    )
    embedding = np.array([1.0, 0.0], dtype=np.float32)
    gid = assigner.gallery.create_new_identity("cam1", 1, embedding, timestamp=1.0)

    first = assigner.assign_tracks(
        "cam2", 1, 100, 120, np.stack([track(1, 0)]), np.stack([embedding]), timestamp=2.0
    )
    confirmed = assigner.assign_tracks(
        "cam2", 2, 100, 120, np.stack([track(1, 0)]), np.stack([embedding]), timestamp=2.4
    )

    assert first[0]["global_id"] is None
    assert first[0]["match"]["reason"] == "pending_new_identity"
    assert confirmed[0]["global_id"] == gid
    assert confirmed[0]["match"]["reason"] == "appearance_match_verified_pending_average"


def test_edge_observations_accumulate_but_wait_for_interior_before_matching(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(
            new_identity_min_frames=2,
            new_identity_min_seconds=0.4,
            cross_camera_exclusion_seconds=0,
        ),
        tmp_path,
    )
    embedding = np.array([1.0, 0.0], dtype=np.float32)
    gid = assigner.gallery.create_new_identity("cam1", 1, embedding, timestamp=1.0)

    first_edge = assigner.assign_tracks(
        "cam2", 1, 100, 120, np.stack([edge_track(1, 0)]), np.stack([embedding]), timestamp=2.0
    )
    second_edge = assigner.assign_tracks(
        "cam2", 2, 100, 120, np.stack([edge_track(1, 0)]), np.stack([embedding]), timestamp=2.2
    )
    interior = assigner.assign_tracks(
        "cam2", 3, 100, 120, np.stack([track(1, 0)]), np.stack([embedding]), timestamp=2.4
    )

    assert first_edge[0]["global_id"] is None
    assert second_edge[0]["global_id"] is None
    assert second_edge[0]["match"]["reason"] == "pending_edge_identity"
    assert interior[0]["global_id"] == gid
    assert interior[0]["match"]["reason"] == "appearance_match_verified_pending_average"


def test_sustained_same_camera_cooccurrence_prevents_later_identity_merge(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(
            identity_min_confidence=0.5,
            same_camera_conflict_min_frames=2,
            same_camera_conflict_min_seconds=0.1,
            same_camera_conflict_max_iou=0.2,
        ),
        tmp_path,
    )
    embedding = np.array([1.0, 0.0], dtype=np.float32)
    gid = assigner.gallery.create_new_identity("cam1", 1, embedding, timestamp=1.0)

    established = track(1, 0)
    separate_low_quality = track(2, 1, confidence=0.1).copy()
    separate_low_quality[:4] = [80, 10, 120, 100]
    for frame_id, timestamp in [(1, 1.0), (2, 1.1)]:
        assigner.assign_tracks(
            "cam1",
            frame_id,
            200,
            120,
            np.stack([established, separate_low_quality]),
            np.stack([embedding, embedding]),
            timestamp=timestamp,
        )

    separate = separate_low_quality.copy()
    separate[5] = 0.9
    separate[7] = 0
    record = assigner.assign_tracks(
        "cam1",
        3,
        200,
        120,
        np.stack([separate]),
        np.stack([embedding]),
        timestamp=1.2,
    )[0]

    assert record["global_id"] != gid
    assert record["match"]["blocked_candidate_global_id"] == gid


def test_transient_same_camera_cooccurrence_does_not_create_cannot_link(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(
            identity_min_confidence=0.5,
            same_camera_conflict_min_frames=3,
            same_camera_conflict_min_seconds=0.2,
            same_camera_conflict_max_iou=0.2,
        ),
        tmp_path,
    )
    embedding = np.array([1.0, 0.0], dtype=np.float32)
    gid = assigner.gallery.create_new_identity("cam1", 1, embedding, timestamp=1.0)
    separate = track(2, 1, confidence=0.1).copy()
    separate[:4] = [80, 10, 120, 100]

    assigner.assign_tracks(
        "cam1",
        1,
        200,
        120,
        np.stack([track(1, 0), separate]),
        np.stack([embedding, embedding]),
        timestamp=1.0,
    )
    separate[5] = 0.9
    separate[7] = 0
    record = assigner.assign_tracks(
        "cam1",
        2,
        200,
        120,
        np.stack([separate]),
        np.stack([embedding]),
        timestamp=1.1,
    )[0]

    assert record["global_id"] == gid


def test_matching_pending_overlap_propagates_cannot_link(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(
            overlapping_camera_pairs=[["cam1", "cam2"]],
            pending_overlap_link_threshold=0.22,
            pending_overlap_link_active_seconds=0.5,
        ),
        tmp_path,
    )
    embedding = np.array([1.0, 0.0], dtype=np.float32)
    assigner.pending_identity_tracks[("cam1", 1)] = {
        "embedding_sum": embedding.copy(),
        "last_seen": 1.0,
        "cannot_link_global_ids": {7},
    }

    propagated = assigner.pending_overlap_cannot_link_global_ids(
        "cam2",
        2,
        embedding,
        timestamp=1.1,
    )

    assert propagated == {7}


def test_long_edge_overlap_can_use_lightness_without_relaxing_deep_threshold(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(
            overlapping_camera_pairs=[["cam1", "cam2"]],
            overlap_candidate_active_seconds=1.0,
            overlap_lightness_fallback_enabled=True,
            overlap_lightness_min_seconds=2.0,
            overlap_lightness_min_edge_frames=5,
            overlap_lightness_min_covisible_seconds=1.0,
            overlap_lightness_max_embedding_distance=0.5,
            overlap_lightness_max_distance=0.12,
        ),
        tmp_path,
    )
    identity_embedding = np.array([1.0, 0.0], dtype=np.float32)
    observation = np.array([0.6, 0.8], dtype=np.float32)
    gid = assigner.gallery.create_new_identity(
        "cam1", 1, identity_embedding, timestamp=10.0
    )
    assigner.record_identity_location(gid, "cam1", 10.0)
    assigner.update_identity_lightness(gid, "cam1", 1, 0.50)
    pending_state = {
        "edge_frames": 10,
        "overlap_identity_evidence": {
            gid: {"first_seen": 8.0, "last_seen": 10.0}
        },
    }

    matched, info = assigner.reliable_active_overlap_lightness_match(
        "cam2",
        observation,
        0.53,
        pending_state,
        pending_seconds=3.0,
        timestamp=10.5,
        blocked_global_ids=set(),
    )

    assert matched == gid
    assert info["best_distance"] == pytest.approx(0.4)
    assert info["lightness_distance"] == pytest.approx(0.03)

    blocked, _ = assigner.reliable_active_overlap_lightness_match(
        "cam2",
        observation,
        0.80,
        pending_state,
        pending_seconds=3.0,
        timestamp=10.5,
        blocked_global_ids=set(),
    )
    assert blocked is None


def test_pending_average_is_rematched_before_creating_duplicate_identity(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(new_identity_min_frames=2, cross_camera_exclusion_seconds=0),
        tmp_path,
    )
    existing = np.array([1.0, 0.0], dtype=np.float32)
    first = np.array([0.75, 0.6614378], dtype=np.float32)
    second = np.array([0.75, -0.6614378], dtype=np.float32)
    gid = assigner.gallery.create_new_identity("cam1", 1, existing, timestamp=1.0)

    pending = assigner.assign_tracks(
        "cam2", 1, 100, 120, np.stack([track(1, 0)]), np.stack([first]), timestamp=1.1
    )
    confirmed = assigner.assign_tracks(
        "cam2", 2, 100, 120, np.stack([track(1, 0)]), np.stack([second]), timestamp=1.2
    )

    assert pending[0]["global_id"] is None
    assert confirmed[0]["global_id"] == gid
    assert confirmed[0]["match"]["reason"] == "appearance_match_verified_pending_average"
    assert len(assigner.gallery.entries) == 1


def test_edge_partial_cannot_remap_established_track(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(
            cross_camera_exclusion_seconds=0,
            existing_track_recheck_threshold=0.55,
            existing_track_remap_margin=0.08,
            existing_track_max_distance=0.45,
            existing_track_remap_require_interior=True,
        ),
        tmp_path,
    )
    original = np.array([1.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0], dtype=np.float32)
    original_id = assigner.gallery.create_new_identity("cam1", 1, original, timestamp=1.0)
    other_id = assigner.gallery.create_new_identity("cam2", 2, other, timestamp=1.0)

    edge_records = assigner.assign_tracks(
        "cam1",
        2,
        100,
        120,
        np.stack([edge_track(1, 0)]),
        np.stack([other]),
        timestamp=1.1,
    )

    assert edge_records[0]["global_id"] == original_id
    assert edge_records[0]["match"]["reason"] == "existing_local_track_drift_hold"
    assert edge_records[0]["match"]["remap_suppressed_reason"] == "edge_partial"

    interior_records = assigner.assign_tracks(
        "cam1",
        3,
        100,
        120,
        np.stack([track(1, 0)]),
        np.stack([other]),
        timestamp=1.2,
    )

    assert interior_records[0]["global_id"] == other_id
    assert interior_records[0]["match"]["reason"] == "appearance_remap"


def test_edge_partial_cannot_seed_new_global_identity(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(new_identity_min_frames=1, new_identity_require_interior=True),
        tmp_path,
    )
    embedding = np.array([1.0, 0.0], dtype=np.float32)

    records = assigner.assign_tracks(
        "cam1",
        1,
        100,
        120,
        np.stack([edge_track(1, 0)]),
        np.stack([embedding]),
        timestamp=1.0,
    )

    assert records[0]["global_id"] is None
    assert records[0]["match"]["reason"] == "pending_edge_identity"
    assert "edge_partial_new_identity" in records[0]["match"]["quality"]["reasons"]
    assert not assigner.gallery.entries


def test_new_identity_confirmation_requires_elapsed_time_at_high_fps(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(new_identity_min_frames=2, new_identity_min_seconds=0.4),
        tmp_path,
    )
    embedding = np.array([1.0, 0.0], dtype=np.float32)

    records = []
    for frame_id, timestamp in enumerate((1.0, 1.1, 1.2, 1.4), start=1):
        records = assigner.assign_tracks(
            "cam1",
            frame_id,
            100,
            120,
            np.stack([track(1, 0)]),
            np.stack([embedding]),
            timestamp=timestamp,
        )
        if timestamp < 1.4:
            assert records[0]["global_id"] is None

    assert records[0]["global_id"] == 1
    assert np.isclose(records[0]["match"]["confirmation_seconds"], 0.4)


def test_ambiguous_near_match_waits_for_a_better_view(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(
            new_identity_ambiguous_wait_seconds=1.0,
            new_identity_ambiguous_distance=0.45,
            cross_camera_exclusion_seconds=0,
        ),
        tmp_path,
    )
    existing = np.array([1.0, 0.0], dtype=np.float32)
    ambiguous = np.array([0.6, 0.8], dtype=np.float32)
    assigner.gallery.create_new_identity("cam1", 1, existing, timestamp=1.0)

    pending = assigner.assign_tracks(
        "cam2",
        1,
        100,
        120,
        np.stack([track(2, 0)]),
        np.stack([ambiguous]),
        timestamp=1.1,
    )
    confirmed = assigner.assign_tracks(
        "cam2",
        2,
        100,
        120,
        np.stack([track(2, 0)]),
        np.stack([ambiguous]),
        timestamp=2.2,
    )

    assert pending[0]["global_id"] is None
    assert pending[0]["match"]["reason"] == "pending_ambiguous_identity"
    assert confirmed[0]["global_id"] == 2


def test_duplicate_local_track_cannot_create_second_global_identity(tmp_path: Path):
    assigner = GlobalIdentityAssigner(
        config(duplicate_local_track_iou_threshold=0.85),
        tmp_path,
    )
    embedding = np.array([1.0, 0.0], dtype=np.float32)
    gid = assigner.gallery.create_new_identity("cam1", 1, embedding, timestamp=1.0)

    records = assigner.assign_tracks(
        "cam1",
        2,
        100,
        120,
        np.stack([track(2, 0, 0.99), track(1, 1, 0.8)]),
        np.stack([embedding, embedding]),
        timestamp=1.1,
    )

    assert records[1]["global_id"] == gid
    assert records[0]["global_id"] is None
    assert records[0]["match"]["reason"] == "duplicate_local_track_suppressed"
    assert len(assigner.gallery.entries) == 1
