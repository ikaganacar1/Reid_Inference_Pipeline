import os
from pathlib import Path

import pytest

from src.runtime_config import (
    apply_reid_overrides,
    apply_realtime_overrides,
    apply_yolo_overrides,
    load_runtime_environment,
    runtime_paths,
)


@pytest.fixture(autouse=True)
def restore_process_environment():
    original = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original)


def test_dotenv_loads_types_later_and_preserves_explicit_environment(
    monkeypatch,
    tmp_path: Path,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "PIPELINE_ROLE=worker",
                "PRIME_URL=ws://from-file:8765",
                'CAMERA_IDS="cam2,cam3"',
                "SAVE_VIDEO=false",
            ]
        )
    )
    monkeypatch.setenv("PRIME_URL", "ws://explicit:9000")

    load_runtime_environment(env_file)

    assert os.environ["PIPELINE_ROLE"] == "worker"
    assert os.environ["PRIME_URL"] == "ws://explicit:9000"


def test_realtime_environment_overrides_are_typed(monkeypatch):
    monkeypatch.setenv("PRIME_PORT", "9000")
    monkeypatch.setenv("SAVE_VIDEO", "false")
    monkeypatch.setenv("CAMERA_IDS", "cam1,cam2")
    monkeypatch.setenv("CAMERA_SOURCES", "0,rtsp://camera.local/stream")
    monkeypatch.setenv("CAMERA_AUTO_SCAN", "false")
    monkeypatch.setenv("OVERLAPPING_CAMERA_PAIRS", "cam1:cam2")
    monkeypatch.setenv(
        "WORKER_NODES",
        "prime=http://prime.local:8787,worker=http://worker.local:8787",
    )
    base = {
        "network": {"prime_port": 8765},
        "worker": {"camera_id": "old"},
        "prime": {"save_video": True},
        "control": {"worker": {}},
    }

    config = apply_realtime_overrides(base)

    assert config["network"]["prime_port"] == 9000
    assert config["prime"]["save_video"] is False
    assert config["control"]["worker"]["camera_ids"] == ["cam1", "cam2"]
    assert config["control"]["worker"]["sources"] == [
        0,
        "rtsp://camera.local/stream",
    ]
    assert config["control"]["worker"]["auto_scan"] is False
    assert config["prime"]["overlapping_camera_pairs"] == [["cam1", "cam2"]]
    assert [node["name"] for node in config["control"]["worker_nodes"]] == [
        "prime",
        "worker",
    ]


def test_reid_and_yolo_environment_overrides(monkeypatch):
    monkeypatch.setenv("REID_INPUT_HEIGHT", "384")
    monkeypatch.setenv("REID_INPUT_WIDTH", "192")
    monkeypatch.setenv("REID_BATCH_SIZE", "6")
    monkeypatch.setenv("REID_PROVIDERS", "CUDAExecutionProvider")
    monkeypatch.setenv("YOLO_CONFIDENCE", "0.5")
    monkeypatch.setenv("YOLO_HALF", "false")

    reid = apply_reid_overrides(
        {
            "model": {"input_shape": [256, 128]},
            "inference": {"max_batch_size": 4},
        }
    )
    yolo = apply_yolo_overrides(
        {
            "detection": {"conf_threshold": 0.25},
            "inference": {"half": True},
        }
    )

    assert reid["model"]["input_shape"] == [384, 192]
    assert reid["inference"]["max_batch_size"] == 6
    assert reid["onnxruntime"]["providers"] == ["CUDAExecutionProvider"]
    assert yolo["detection"]["conf_threshold"] == 0.5
    assert yolo["inference"]["half"] is False


def test_worker_role_selects_worker_defaults(monkeypatch):
    for name in ("REALTIME_CONFIG", "YOLO_CONFIG", "CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)

    paths = runtime_paths("worker")

    assert paths.realtime.name == "realtime_config.worker.yaml"
    assert paths.yolo.name == "yolo_config.worker.yaml"


def test_invalid_boolean_fails_fast(monkeypatch):
    monkeypatch.setenv("SAVE_VIDEO", "sometimes")

    with pytest.raises(ValueError, match="Invalid SAVE_VIDEO"):
        apply_realtime_overrides({"prime": {}})


def test_dotenv_rejects_unquoted_spaces(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("CAMERA_SOURCE=path with spaces\n")

    with pytest.raises(ValueError, match="must be quoted"):
        load_runtime_environment(env_file, override=True)


def test_example_env_is_valid_shell_compatible_dotenv(monkeypatch):
    for key in list(os.environ):
        if key.startswith(
            (
                "PIPELINE_",
                "CAMERA_",
                "YOLO_",
                "REID_",
                "TRACKER_",
                "PRIME_",
            )
        ):
            monkeypatch.delenv(key, raising=False)

    path = load_runtime_environment(Path(".env.example"), override=True)

    assert path.name == ".env.example"
    assert runtime_paths().realtime.is_file()
