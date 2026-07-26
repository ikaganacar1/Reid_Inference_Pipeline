from pathlib import Path

import pytest

from src.realtime.worker_control import RealtimeWorkerControl


def control_config(camera_ids=None):
    return {
        "worker": {"camera_id": "cam2"},
        "control": {
            "worker": {
                "auto_scan": True,
                "sources": [],
                "camera_ids": ["cam2"] if camera_ids is None else camera_ids,
            }
        },
    }


def test_no_discovered_camera_is_reported_as_failure(tmp_path: Path):
    control = RealtimeWorkerControl(control_config(), tmp_path)
    control.worker_processes = lambda include_launcher=False: []
    control.discover_sources = lambda: []

    result = control.start_workers(force_scan=True)

    assert result["ok"] is False
    assert result["sources"] == []


def test_configured_camera_id_is_used_for_single_discovered_source(tmp_path: Path):
    control = RealtimeWorkerControl(control_config(), tmp_path)
    control.sources = ["/dev/v4l/by-path/usb-port-video-index0"]

    assert control.camera_ids_for_sources() == ["cam2"]


def test_duplicate_configured_camera_ids_are_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="must be unique"):
        RealtimeWorkerControl(control_config(["cam2", "cam2"]), tmp_path)


def test_source_count_must_match_configured_global_camera_ids(tmp_path: Path):
    control = RealtimeWorkerControl(control_config(["cam2"]), tmp_path)
    control.worker_processes = lambda include_launcher=False: []
    control.discover_sources = lambda: ["/dev/video0", "/dev/video2"]

    result = control.start_workers(force_scan=True)

    assert result["ok"] is False
    assert "2 camera(s)" in result["message"]
    assert control.camera_ids_for_sources() == []


def test_running_worker_source_mapping_detects_usb_port_change(tmp_path: Path):
    control = RealtimeWorkerControl(control_config(["cam2"]), tmp_path)
    workers = [
        {
            "pid": 42,
            "cmd": (
                "python scripts/realtime_worker.py --camera-id cam2 "
                "--source /dev/v4l/by-path/old-port-video-index0"
            ),
        }
    ]

    assert control.workers_match_sources(
        workers,
        ["/dev/v4l/by-path/old-port-video-index0"],
    )
    assert not control.workers_match_sources(
        workers,
        ["/dev/v4l/by-path/new-port-video-index0"],
    )


def test_start_on_launch_and_retry_are_loaded(tmp_path: Path):
    config = control_config()
    config["control"]["worker"].update(
        {"start_on_launch": True, "auto_start_retry_seconds": 2.5}
    )

    control = RealtimeWorkerControl(config, tmp_path)

    assert control.auto_start_enabled is True
    assert control.auto_start_retry_seconds == 2.5


def test_public_status_redacts_camera_url_credentials(tmp_path: Path):
    control = RealtimeWorkerControl(control_config(), tmp_path)
    userinfo = ":".join(("camera-user", "camera-pass"))
    credentialed_source = f"rtsp://{userinfo}@camera.local/stream"
    control.sources = [credentialed_source]
    control.worker_processes = lambda include_launcher=False: [
        {
            "pid": 42,
            "cmd": (
                "python scripts/realtime_worker.py --source "
                f"{credentialed_source}"
            ),
        }
    ]
    control.discover_sources = lambda: control.sources
    control.tail_logs = lambda: {}
    control.worker_metrics = lambda: {}

    status = control.status()

    assert "camera-user" not in status["sources"][0]
    assert "camera-pass" not in status["workers"][0]["cmd"]
    assert "rtsp://***@camera.local/stream" in status["sources"][0]


def test_metrics_support_non_cam_prefixed_camera_ids(tmp_path: Path):
    control = RealtimeWorkerControl(control_config(), tmp_path)
    control.log_dir.mkdir(parents=True)
    (control.log_dir / "entrance.log").write_text(
        "camera=entrance frame=120 detections=2 sent_fps=9.75\n"
    )
    (control.log_dir / "control.log").write_text(
        "camera=wrong frame=999 detections=9 sent_fps=1.0\n"
    )

    metrics = control.worker_metrics()

    assert metrics["entrance"]["frame_id"] == 120
    assert "wrong" not in metrics
