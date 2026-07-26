import hashlib

from scripts.jetson_preflight import (
    Check,
    check_model_integrity,
    check_output_directory,
    resolve_reid_model_path,
    resolve_yolo_model_path,
    valid_prime_url,
)


def test_yolo_model_environment_override(monkeypatch, tmp_path):
    model = tmp_path / "detector.pt"
    model.touch()
    monkeypatch.setenv("YOLO_MODEL_PATH", str(model))

    assert resolve_yolo_model_path({"model": {"path": "missing.pt"}}) == model.resolve()


def test_reid_model_environment_override(monkeypatch, tmp_path):
    model = tmp_path / "reid.onnx"
    model.touch()
    monkeypatch.setenv("REID_MODEL_PATH", str(model))

    assert resolve_reid_model_path({"model": {"onnx_path": "missing.onnx"}}) == model.resolve()


def test_prime_url_requires_websocket_host_and_port():
    assert valid_prime_url("ws://10.0.0.1:8765")[0]
    assert valid_prime_url("wss://prime.local:443")[0]
    assert not valid_prime_url("http://10.0.0.1:8765")[0]
    assert not valid_prime_url("ws://10.0.0.1")[0]


def test_model_manifest_hash_is_verified(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"tested model bytes")
    manifest = {
        "reid": {
            "size_bytes": model.stat().st_size,
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        }
    }
    checks: list[Check] = []

    check_model_integrity(checks, "reid", model, manifest, skip_hash=False)

    assert checks == [
        Check(
            "reid_model_integrity",
            True,
            f"sha256={manifest['reid']['sha256']}",
        )
    ]


def test_model_manifest_size_mismatch_fails_before_hashing(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"wrong")
    checks: list[Check] = []

    check_model_integrity(
        checks,
        "yolo",
        model,
        {"yolo": {"size_bytes": 99, "sha256": "unused"}},
        skip_hash=False,
    )

    assert checks[0].ok is False
    assert "expected=99" in checks[0].detail


def test_output_directory_environment_override(monkeypatch, tmp_path):
    output_dir = tmp_path / "external-recordings"
    monkeypatch.setenv("REALTIME_OUTPUT_DIR", str(output_dir))
    checks: list[Check] = []

    check_output_directory(checks, {"prime": {"save_video": False}})

    assert checks == [Check("output_directory", True, str(output_dir.resolve()))]


def test_missing_required_recording_mountpoint_fails_before_directory_creation(
    monkeypatch,
    tmp_path,
):
    mountpoint = tmp_path / "not-mounted"
    output_dir = mountpoint / "reid"
    monkeypatch.setenv("REALTIME_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("RECORDING_MOUNTPOINT", str(mountpoint))
    checks: list[Check] = []

    check_output_directory(checks, {"prime": {"save_video": True}})

    assert checks[0].name == "recording_mountpoint"
    assert checks[0].ok is False
    assert not output_dir.exists()
