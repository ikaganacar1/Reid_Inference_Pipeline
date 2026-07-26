import numpy as np

from scripts.export_identity_videos import (
    IdentityObservation,
    discover_recordings,
    resolve_historical_global_id,
    select_identity_observation,
    touches_frame_edge,
)


def test_discover_recordings_supports_flat_channel_files(tmp_path):
    session_dir = tmp_path / "cafe"
    session_dir.mkdir()
    ch201 = session_dir / "channel_201_20260722T100000Z_180s.mkv"
    ch301 = session_dir / "channel_301_20260722T100000Z_180s.mkv"
    ch201.touch()
    ch301.touch()

    recordings = discover_recordings(tmp_path, "cafe", "unused.mkv")

    assert recordings == {"ch201": ch201, "ch301": ch301}


def observation(camera_id, *, edge, quality):
    return IdentityObservation(
        camera_id=camera_id,
        timestamp=1.0,
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        record={},
        policy_note=None,
        touches_edge=edge,
        quality=quality,
    )


def test_overlap_export_keeps_current_interior_camera_to_avoid_flicker():
    current = observation("ch201", edge=False, quality=(1.0, 0.2, 0.03, 0.8))
    larger = observation("ch301", edge=False, quality=(1.0, 0.5, 0.1, 0.9))

    selected = select_identity_observation([larger, current], "ch201")

    assert selected is current


def test_overlap_export_leaves_current_camera_for_an_interior_crop():
    edge = observation("ch201", edge=True, quality=(0.0, 0.4, 0.1, 0.9))
    interior = observation("ch301", edge=False, quality=(1.0, 0.2, 0.03, 0.7))

    selected = select_identity_observation([edge, interior], "ch201")

    assert selected is interior


def record(global_id, local_id, bbox, reason="existing_local_track_verified", **match):
    return {
        "global_id": global_id,
        "local_track_id": local_id,
        "bbox": bbox,
        "match": {"reason": reason, **match},
    }


def test_touches_frame_edge_uses_configured_margin():
    assert touches_frame_edge([0, 10, 50, 90], 100, 100, 0.01)
    assert touches_frame_edge([10, 10, 100, 90], 100, 100, 0.01)
    assert not touches_frame_edge([10, 10, 90, 90], 100, 100, 0.01)


def test_historical_edge_remap_guard_keeps_previous_id_for_track():
    overrides = {}
    remap = record(
        9,
        4,
        [20, 0, 60, 50],
        "appearance_remap",
        previous_global_id=7,
    )

    global_id, note = resolve_historical_global_id(
        "cam1", remap, 100, 100, overrides, True
    )
    held_id, held_note = resolve_historical_global_id(
        "cam1",
        record(9, 4, [20, 0, 60, 40]),
        100,
        100,
        overrides,
        True,
    )

    assert global_id == 7
    assert note == "edge_remap_suppressed"
    assert held_id == 7
    assert held_note == "edge_remap_suppressed_hold"


def test_interior_remap_replaces_historical_override():
    overrides = {("cam1", 4): 7}
    interior_remap = record(
        9,
        4,
        [20, 20, 60, 80],
        "appearance_remap",
        previous_global_id=7,
    )

    global_id, note = resolve_historical_global_id(
        "cam1", interior_remap, 100, 100, overrides, True
    )

    assert global_id == 9
    assert note is None
    assert ("cam1", 4) not in overrides
