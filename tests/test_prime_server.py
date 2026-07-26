import asyncio
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.realtime.prime_server import RealtimePrimeServer
from src.realtime.protocol import FramePacket, encode_jpeg


class EmptyReIDClient:
    def infer(self, crops, max_batch_size):
        assert crops == []
        return np.empty((0, 2), dtype=np.float32)


class EmptyTracker:
    def __init__(self):
        self.reset_count = 0

    def update(self, detections, frame, embeddings):
        return np.empty((0, 8), dtype=np.float32)

    def reset(self):
        self.reset_count += 1


class RecordingAssigner:
    def __init__(self):
        self.calls = []
        self.reset_cameras = []

    def assign_tracks(
        self,
        camera_id,
        frame_id,
        frame_width,
        frame_height,
        tracks,
        embeddings,
        crops,
        timestamp,
    ):
        self.calls.append((camera_id, timestamp))
        return []

    def reset_camera(self, camera_id):
        self.reset_cameras.append(camera_id)


def packet(camera_id, frame_id, timestamp):
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    return FramePacket(
        camera_id=camera_id,
        frame_id=frame_id,
        timestamp=timestamp,
        width=12,
        height=8,
        detections=np.empty((0, 6), dtype=np.float32),
        frame_jpeg=encode_jpeg(image, 90),
        crop_jpegs=[],
        received_at=timestamp,
    )


def bare_server():
    server = object.__new__(RealtimePrimeServer)
    server.reid_client = EmptyReIDClient()
    server.reid_batch_size = 4
    server.max_capture_clock_skew_seconds = 5.0
    server.trackers = {"older": EmptyTracker(), "newer": EmptyTracker()}
    server.identity_assigner = RecordingAssigner()
    server.camera_input_state = defaultdict(dict)
    server.camera_runtime = defaultdict(dict)
    server.camera_stats = defaultdict(dict)
    server.camera_tracker_reset_seconds = 3.0
    server.viewer_quality = 75
    server.filter_edge_false_positives = False
    server.save_video = False
    server.writers = {}
    return server


def test_packet_microbatch_is_applied_in_event_time_order():
    server = bare_server()

    results = server.process_packets(
        [packet("newer", 2, 2.0), packet("older", 1, 1.0)]
    )

    assert [camera_id for camera_id, _ in server.identity_assigner.calls] == ["older", "newer"]
    assert [result["camera_id"] for result in results] == ["older", "newer"]


def test_camera_reset_clears_runtime_fps_state():
    server = bare_server()
    tracker = server.trackers["older"]
    server.camera_input_state["older"] = {"last_frame_id": 10, "last_timestamp": 10.0}
    server.camera_runtime["older"] = {"fps": 99.0}

    _, reason = server.prepare_camera_tracker(packet("older", 0, 11.0), 11.0)

    assert reason == "frame_id_rollback"
    assert tracker.reset_count == 1
    assert "older" not in server.camera_runtime
    assert server.identity_assigner.reset_cameras == ["older"]


def test_slow_viewer_is_dropped_without_blocking_other_viewers():
    class Viewer:
        def __init__(self, delay):
            self.delay = delay
            self.messages = []

        async def send_str(self, message):
            await asyncio.sleep(self.delay)
            self.messages.append(message)

    server = object.__new__(RealtimePrimeServer)
    fast = Viewer(0)
    slow = Viewer(0.2)
    server.viewers = {fast, slow}
    server.viewer_send_timeout_seconds = 0.01

    asyncio.run(server.broadcast({"type": "test"}))

    assert fast in server.viewers
    assert fast.messages
    assert slow not in server.viewers


def test_recording_rotates_and_finalizes_segments(monkeypatch, tmp_path: Path):
    class Writer:
        def __init__(self, path, *_):
            self.path = Path(path)
            self.released = False
            self.frames = 0

        def isOpened(self):
            return True

        def write(self, _frame):
            self.frames += 1

        def release(self):
            self.released = True

    writers = []

    def create_writer(*args):
        writer = Writer(*args)
        writers.append(writer)
        return writer

    times = iter([10.0, 12.0])
    monkeypatch.setattr("src.realtime.prime_server.time.monotonic", lambda: next(times))
    monkeypatch.setattr("src.realtime.prime_server.cv2.VideoWriter", create_writer)

    server = object.__new__(RealtimePrimeServer)
    server.save_video = True
    server.output_fps = 10
    server.recording_segment_seconds = 1
    server.recording_dir = tmp_path
    server.writers = {}
    server.writer_started_at = {}
    server.writer_segment_index = defaultdict(int)
    frame = np.zeros((8, 12, 3), dtype=np.uint8)

    server.write_video("cam1", frame)
    server.write_video("cam1", frame)

    assert len(writers) == 2
    assert writers[0].released is True
    assert writers[0].path.name == "cam1_000001_processed.mp4"
    assert writers[1].path.name == "cam1_000002_processed.mp4"
    assert writers[1].frames == 1


def test_low_disk_space_releases_writers_and_pauses_recording(monkeypatch, tmp_path: Path):
    class Writer:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    writer = Writer()
    disk = {"free": 512}
    monkeypatch.setattr(
        "src.realtime.prime_server.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=disk["free"]),
    )
    server = object.__new__(RealtimePrimeServer)
    server.recording_min_free_bytes = 1024
    server.recording_disk_check_seconds = 10
    server.recording_last_disk_check = 0
    server.recording_free_bytes = None
    server.recording_paused_reason = None
    server.recording_dir = tmp_path
    server.writers = {"cam1": writer}
    server.writer_started_at = {"cam1": 1.0}

    assert not server.recording_space_available(now=11.0)
    assert writer.released
    assert server.writers == {}
    assert server.recording_paused_reason == "low_disk_space"

    disk["free"] = 2048
    assert server.recording_space_available(now=22.0)
    assert server.recording_paused_reason is None


def test_runtime_mount_loss_pauses_before_disk_usage(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Path, "is_mount", lambda _path: False)
    monkeypatch.setattr(
        "src.realtime.prime_server.shutil.disk_usage",
        lambda _path: (_ for _ in ()).throw(AssertionError("must not inspect root disk")),
    )
    server = object.__new__(RealtimePrimeServer)
    server.recording_min_free_bytes = 1024
    server.recording_disk_check_seconds = 10
    server.recording_last_disk_check = 0
    server.recording_free_bytes = None
    server.recording_paused_reason = None
    server.recording_mountpoint = tmp_path
    server.recording_dir = tmp_path / "session"
    server.writers = {}
    server.writer_started_at = {}

    assert not server.recording_space_available(now=11.0)
    assert server.recording_paused_reason == "recording_mountpoint_unavailable"
