#!/usr/bin/env python3
"""Fail-fast deployment checks for a prime or camera-worker Jetson role."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["prime", "worker"], required=True)
    parser.add_argument("--realtime-config", type=Path)
    parser.add_argument("--reid-config", type=Path, default=Path("configs/reid_config.yaml"))
    parser.add_argument("--tracker-config", type=Path, default=Path("configs/tracker_config.yaml"))
    parser.add_argument("--yolo-config", type=Path)
    parser.add_argument("--load-models", action="store_true")
    parser.add_argument("--check-camera", action="store_true")
    parser.add_argument("--allow-non-jetson", action="store_true")
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=Path("deploy/model_manifest.yaml"),
    )
    parser.add_argument("--skip-model-hash", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as config_file:
        value = yaml.safe_load(config_file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def resolve_yolo_model_path(config: dict[str, Any]) -> Path:
    configured = config.get("model", {}).get("path")
    if not configured:
        raise ValueError("YOLO config is missing model.path")
    return Path(os.environ.get("YOLO_MODEL_PATH", str(configured))).expanduser().resolve()


def resolve_reid_model_path(config: dict[str, Any]) -> Path:
    model = config.get("model", {})
    configured_value = model.get("onnx_path")
    if not configured_value:
        raise ValueError("ReID config is missing model.onnx_path")
    configured = Path(str(configured_value)).expanduser()
    candidates = []
    if os.environ.get("REID_MODEL_PATH"):
        candidates.append(Path(os.environ["REID_MODEL_PATH"]).expanduser())
    candidates.extend([configured, Path.home() / configured])
    for value in model.get("search_paths", []):
        search_path = Path(str(value)).expanduser()
        candidates.append(search_path / configured.name if search_path.is_dir() else search_path)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return candidates[0].resolve() if candidates else configured.resolve()


def valid_prime_url(value: str) -> tuple[bool, str]:
    parsed = urlparse(value)
    ok = parsed.scheme in {"ws", "wss"} and bool(parsed.hostname) and parsed.port is not None
    return ok, f"{parsed.scheme or '-'}://{parsed.netloc or '-'}"


def add_import_check(checks: list[Check], module_name: str) -> Any | None:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "version unavailable")
        checks.append(Check(f"import:{module_name}", True, str(version)))
        return module
    except Exception as exc:
        checks.append(Check(f"import:{module_name}", False, f"{type(exc).__name__}: {exc}"))
        return None


def check_platform(checks: list[Check], allow_non_jetson: bool) -> bool:
    machine = platform.machine()
    tegra_release = Path("/etc/nv_tegra_release")
    jetson = machine == "aarch64" and tegra_release.is_file()
    detail = f"machine={machine} tegra_release={tegra_release.is_file()}"
    checks.append(Check("jetson_platform", jetson, detail, required=not allow_non_jetson))
    return jetson


def check_clock_sync(checks: list[Check], required: bool) -> None:
    try:
        result = subprocess.run(
            ["timedatectl", "show", "--property=NTPSynchronized", "--value"],
            text=True,
            capture_output=True,
            timeout=3,
            check=False,
        )
        value = result.stdout.strip().lower()
        checks.append(Check("clock_synchronized", value == "yes", f"NTPSynchronized={value or '-'}", required))
    except Exception as exc:
        checks.append(Check("clock_synchronized", False, f"{type(exc).__name__}: {exc}", required))


def check_model_integrity(
    checks: list[Check],
    label: str,
    path: Path,
    manifest: dict[str, Any],
    skip_hash: bool,
) -> None:
    expected = manifest.get(label)
    if not isinstance(expected, dict):
        checks.append(Check(f"{label}_model_integrity", False, "manifest entry missing"))
        return
    if not path.is_file():
        return
    actual_size = path.stat().st_size
    expected_size = int(expected.get("size_bytes", -1))
    if actual_size != expected_size:
        checks.append(
            Check(
                f"{label}_model_integrity",
                False,
                f"size={actual_size} expected={expected_size}",
            )
        )
        return
    if skip_hash:
        checks.append(Check(f"{label}_model_integrity", True, f"size={actual_size}; hash skipped"))
        return
    sha256 = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest()
    expected_hash = str(expected.get("sha256", ""))
    checks.append(
        Check(
            f"{label}_model_integrity",
            actual_hash == expected_hash,
            f"sha256={actual_hash}",
        )
    )


def check_torch_cuda(checks: list[Check]) -> Any | None:
    torch = add_import_check(checks, "torch")
    if torch is None:
        return None
    try:
        available = bool(torch.cuda.is_available())
        detail = (
            f"torch={torch.__version__} cuda={torch.version.cuda} "
            f"device={torch.cuda.get_device_name(0) if available else '-'}"
        )
        checks.append(Check("torch_cuda", available, detail))
    except Exception as exc:
        checks.append(Check("torch_cuda", False, f"{type(exc).__name__}: {exc}"))
    return torch


def check_output_directory(checks: list[Check], config: dict[str, Any]) -> None:
    output_dir = Path(
        os.environ.get(
            "REALTIME_OUTPUT_DIR",
            config.get("prime", {}).get("output_dir", "outputs/realtime"),
        )
    ).expanduser()
    save_video = bool(config.get("prime", {}).get("save_video", True))
    recording_mountpoint = os.environ.get("RECORDING_MOUNTPOINT")
    if save_video and recording_mountpoint:
        mountpoint = Path(recording_mountpoint).expanduser().resolve()
        resolved_output = output_dir.resolve()
        output_on_mount = resolved_output == mountpoint or mountpoint in resolved_output.parents
        mount_ok = mountpoint.is_mount() and output_on_mount
        checks.append(
            Check(
                "recording_mountpoint",
                mount_ok,
                f"mountpoint={mountpoint} output={resolved_output}",
            )
        )
        if not mount_ok:
            return
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_dir):
            pass
    except Exception as exc:
        checks.append(Check("output_directory", False, f"{output_dir}: {exc}"))
        return

    checks.append(Check("output_directory", True, str(output_dir.resolve())))
    if save_video:
        try:
            minimum_free_bytes = int(
                float(
                    os.environ.get(
                        "RECORDING_MIN_FREE_GB",
                        config.get("prime", {}).get("recording_min_free_gb", 5),
                    )
                )
                * 1024**3
            )
            free_bytes = shutil.disk_usage(output_dir).free
            checks.append(
                Check(
                    "recording_free_space",
                    free_bytes >= minimum_free_bytes,
                    (
                        f"free_gib={free_bytes / 1024**3:.1f} "
                        f"minimum_gib={minimum_free_bytes / 1024**3:.1f}"
                    ),
                )
            )
        except Exception as exc:
            checks.append(Check("recording_free_space", False, f"{type(exc).__name__}: {exc}"))


def check_prime(
    checks: list[Check],
    realtime_config: dict[str, Any],
    reid_config_path: Path,
    tracker_config_path: Path,
    load_models: bool,
    model_manifest: dict[str, Any],
    skip_model_hash: bool,
) -> None:
    reid_config = load_yaml(reid_config_path)
    tracker_config = load_yaml(tracker_config_path)
    add_import_check(checks, "boxmot")
    ort = add_import_check(checks, "onnxruntime")
    if ort is not None:
        providers = ort.get_available_providers()
        checks.append(
            Check(
                "onnxruntime_cuda_provider",
                "CUDAExecutionProvider" in providers,
                f"providers={providers}",
            )
        )

    model_path = resolve_reid_model_path(reid_config)
    checks.append(Check("reid_model", model_path.is_file(), str(model_path)))
    check_model_integrity(checks, "reid", model_path, model_manifest, skip_model_hash)
    check_output_directory(checks, realtime_config)

    if load_models and model_path.is_file() and ort is not None:
        try:
            import numpy as np

            from src.reid_client import create_reid_client

            client = create_reid_client(reid_config)
            test_batch_size = max(
                1,
                min(
                    int(realtime_config.get("prime", {}).get("reid_batch_size", 1)),
                    int(reid_config.get("inference", {}).get("max_batch_size", 1)),
                ),
            )
            crops = [
                np.zeros((256, 128, 3), dtype=np.uint8)
                for _ in range(test_batch_size)
            ]
            embedding = client.infer(crops, max_batch_size=test_batch_size)
            expected_dim = int(reid_config["model"]["embedding_dim"])
            ok = embedding.shape == (test_batch_size, expected_dim) and bool(
                np.all(np.isfinite(embedding))
            )
            checks.append(Check("reid_inference", ok, f"shape={embedding.shape}"))
            client.close()
        except Exception as exc:
            checks.append(Check("reid_inference", False, f"{type(exc).__name__}: {exc}"))

        try:
            from src.tracker import ReIDTracker

            tracker = ReIDTracker(tracker_config)
            checks.append(Check("tracker_initialization", tracker is not None, "BoTSORT initialized"))
        except Exception as exc:
            checks.append(Check("tracker_initialization", False, f"{type(exc).__name__}: {exc}"))


def discover_camera_sources() -> list[str]:
    by_path = sorted(Path("/dev/v4l/by-path").glob("*video-index0"))
    sources = [str(path) for path in by_path if path.exists()]
    if sources:
        return sources
    return [
        str(path)
        for path in sorted(Path("/dev").glob("video*"))
        if path.name[len("video") :].isdigit()
        and int(path.name[len("video") :]) % 2 == 0
    ]


def check_worker_cameras(checks: list[Check], camera_ids: list[str]) -> bool:
    sources = discover_camera_sources()
    if not sources:
        checks.append(Check("camera_capture", False, "no camera source discovered"))
        return False

    mapping_ok = bool(camera_ids) and len(camera_ids) == len(sources)
    checks.append(
        Check(
            "camera_id_mapping",
            mapping_ok,
            f"sources={len(sources)} configured_ids={camera_ids}",
        )
    )
    captures_ok = True
    for source in sources:
        capture = None
        try:
            import cv2

            capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
            ok, frame = capture.read()
            frame_ok = bool(ok and frame is not None)
            captures_ok &= frame_ok
            checks.append(
                Check(
                    f"camera_capture:{source}",
                    frame_ok,
                    f"shape={getattr(frame, 'shape', None)}",
                )
            )
        except Exception as exc:
            captures_ok = False
            checks.append(
                Check(
                    f"camera_capture:{source}",
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
            )
        finally:
            if capture is not None:
                capture.release()
    return mapping_ok and captures_ok


def check_worker(
    checks: list[Check],
    realtime_config: dict[str, Any],
    yolo_config_path: Path,
    load_models: bool,
    check_camera: bool,
    model_manifest: dict[str, Any],
    skip_model_hash: bool,
) -> None:
    add_import_check(checks, "ultralytics")
    yolo_config = load_yaml(yolo_config_path)
    model_path = resolve_yolo_model_path(yolo_config)
    checks.append(Check("yolo_model", model_path.is_file(), str(model_path)))
    check_model_integrity(checks, "yolo", model_path, model_manifest, skip_model_hash)

    prime_url = os.environ.get(
        "PRIME_URL",
        str(realtime_config.get("network", {}).get("prime_url", "")),
    )
    url_ok, url_detail = valid_prime_url(prime_url)
    checks.append(Check("prime_url", url_ok, url_detail))
    camera_ids = [
        str(value)
        for value in realtime_config.get("control", {}).get("worker", {}).get("camera_ids", [])
    ]
    checks.append(
        Check(
            "camera_ids",
            bool(camera_ids) and len(camera_ids) == len(set(camera_ids)),
            str(camera_ids),
        )
    )
    cameras_ready = check_worker_cameras(checks, camera_ids) if check_camera else True

    if load_models and model_path.is_file() and cameras_ready:
        try:
            import numpy as np

            from src.detector import YOLOPersonDetector

            detector = YOLOPersonDetector(yolo_config)
            configured_device = str(yolo_config.get("model", {}).get("device", ""))
            if detector.backend == "ultralytics":
                actual_device = str(next(detector.model.model.parameters()).device)
                device_ok = not configured_device.startswith("cuda") or actual_device.startswith("cuda")
            else:
                actual_device = f"cuda:{detector.model.device_id}"
                device_ok = actual_device == configured_device
            checks.append(
                Check(
                    "yolo_model_device",
                    device_ok,
                    f"configured={configured_device} actual={actual_device}",
                )
            )
            detections, crops = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
            checks.append(
                Check(
                    "yolo_inference",
                    len(detections) == len(crops),
                    f"detections={len(detections)} crops={len(crops)}",
                )
            )
        except Exception as exc:
            checks.append(Check("yolo_inference", False, f"{type(exc).__name__}: {exc}"))


def run_preflight(args: argparse.Namespace) -> list[Check]:
    os.chdir(ROOT)
    realtime_path = args.realtime_config or Path(
        "configs/realtime_config.yaml" if args.role == "prime" else "configs/realtime_config.worker.yaml"
    )
    yolo_path = args.yolo_config or Path(
        "configs/yolo_config.yaml" if args.role == "prime" else "configs/yolo_config.worker.yaml"
    )
    checks: list[Check] = []
    is_jetson = check_platform(checks, args.allow_non_jetson)
    check_clock_sync(checks, required=is_jetson)
    for module_name in ("yaml", "numpy", "cv2", "aiohttp"):
        add_import_check(checks, module_name)
    check_torch_cuda(checks)

    try:
        realtime_config = load_yaml(realtime_path)
        checks.append(Check("realtime_config", True, str(realtime_path.resolve())))
    except Exception as exc:
        checks.append(Check("realtime_config", False, f"{type(exc).__name__}: {exc}"))
        return checks

    try:
        model_manifest = load_yaml(args.model_manifest)
        checks.append(Check("model_manifest", True, str(args.model_manifest.resolve())))
    except Exception as exc:
        checks.append(Check("model_manifest", False, f"{type(exc).__name__}: {exc}"))
        return checks

    if args.role == "prime":
        check_prime(
            checks,
            realtime_config,
            args.reid_config,
            args.tracker_config,
            args.load_models,
            model_manifest,
            args.skip_model_hash,
        )
    else:
        check_worker(
            checks,
            realtime_config,
            yolo_path,
            args.load_models,
            args.check_camera,
            model_manifest,
            args.skip_model_hash,
        )
    return checks


def main() -> None:
    args = parse_args()
    checks = run_preflight(args)
    failed = [check for check in checks if check.required and not check.ok]
    if args.json_output:
        print(json.dumps({"ok": not failed, "role": args.role, "checks": [asdict(item) for item in checks]}, indent=2))
    else:
        print(f"Jetson preflight role={args.role}")
        for check in checks:
            status = "PASS" if check.ok else ("FAIL" if check.required else "WARN")
            print(f"[{status}] {check.name}: {check.detail}")
        print(f"Result: {'READY' if not failed else 'NOT READY'} ({len(failed)} required failures)")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
