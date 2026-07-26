from types import SimpleNamespace

import numpy as np

from scripts.debug_reid_recordings import (
    build_track_prototype_diagnostics,
    closest_embedding_pairs,
    discover_recordings,
    source_frame_at,
    synchronized_replay_timing,
)


def test_closest_embedding_pairs_is_bounded_sorted_and_unique():
    distances = np.array(
        [
            [0.0, 0.3, 0.1, 0.8],
            [0.3, 0.0, 0.2, 0.4],
            [0.1, 0.2, 0.0, 0.5],
            [0.8, 0.4, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    metadata = [{"index": index} for index in range(4)]

    pairs = closest_embedding_pairs(distances, metadata, limit=3)

    assert [round(pair["distance"], 2) for pair in pairs] == [0.1, 0.2, 0.3]
    assert [(pair["a"]["index"], pair["b"]["index"]) for pair in pairs] == [
        (0, 2),
        (1, 2),
        (0, 1),
    ]


def test_discover_recordings_uses_exclusions_without_requiring_allow_list(tmp_path):
    session = tmp_path / "session"
    for channel in ("101", "301", "1301", "2301", "2601"):
        camera_dir = session / f"camera_ch{channel}"
        camera_dir.mkdir(parents=True)
        (camera_dir / "capture.mkv").touch()

    args = SimpleNamespace(
        recordings_root=tmp_path,
        session="session",
        file_name="capture.mkv",
        exclude="2601",
        channels="",
        limit_cameras=0,
    )

    recordings = discover_recordings(args)

    assert [channel for channel, _ in recordings] == ["101", "1301", "2301", "301"]


def test_discover_recordings_supports_flat_cafe_session(tmp_path):
    session = tmp_path / "cafe"
    session.mkdir()
    for channel in ("201", "301", "401"):
        (session / f"channel_{channel}_20260722T100000Z_180s.mkv").touch()

    args = SimpleNamespace(
        recordings_root=tmp_path,
        session="cafe",
        file_name="unused.mkv",
        exclude="401",
        channels="",
        limit_cameras=0,
    )

    assert [channel for channel, _ in discover_recordings(args)] == ["201", "301"]


def test_synchronized_timeline_maps_mixed_fps_by_wall_clock():
    timeline_fps, start, stop, step = synchronized_replay_timing(
        {"ch201": 20.0, "ch501": 25.0},
        {"ch201": 180.0, "ch501": 180.0},
        start_frame=1000,
        max_frames=200,
        stride=2,
    )

    assert (timeline_fps, start, stop, step) == (20.0, 50.0, 60.0, 0.1)
    assert source_frame_at(50.0, 20.0) == 1000
    assert source_frame_at(50.0, 25.0) == 1250


def test_synchronized_timeline_can_run_at_fastest_camera_rate():
    timeline_fps, start, stop, step = synchronized_replay_timing(
        {"ch201": 20.0, "ch501": 25.0},
        {"ch201": 180.0, "ch501": 180.0},
        start_frame=0,
        max_frames=0,
        stride=1,
        replay_fps=25.0,
    )

    source_ids = [source_frame_at(index * step, 20.0) for index in range(25)]

    assert (timeline_fps, start, stop, step) == (25.0, 0.0, 180.0, 0.04)
    assert len(set(source_ids)) == 20


def test_track_prototype_diagnostics_average_local_track_embeddings():
    embeddings = np.array(
        [
            [0.8, 0.6],
            [0.8, -0.6],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    metadata = [
        {"camera_id": "cam1", "local_track_id": 1, "global_id": 1, "frame_id": 1},
        {"camera_id": "cam1", "local_track_id": 1, "global_id": 1, "frame_id": 2},
        {"camera_id": "cam2", "local_track_id": 7, "global_id": 2, "frame_id": 1},
        {"camera_id": "cam2", "local_track_id": 7, "global_id": 2, "frame_id": 2},
        {"camera_id": "cam3", "local_track_id": None, "global_id": None, "frame_id": 1},
    ]

    diagnostics = build_track_prototype_diagnostics(embeddings, metadata)

    assert diagnostics["track_prototype_diagnostics"]["local_track_count"] == 2
    assert diagnostics["track_prototype_diagnostics"]["cross_camera_pair_count"] == 1
    pair = diagnostics["closest_cross_camera_track_prototypes"][0]
    assert pair["distance"] < 0.01
    assert pair["a"]["global_ids"] == [1]
    assert pair["b"]["global_ids"] == [2]
