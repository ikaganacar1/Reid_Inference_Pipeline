#!/usr/bin/env python3
"""Smoke-test runtime configuration and the lightweight realtime data path."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import importlib
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any
from urllib.parse import urlparse

import numpy as np
from aiohttp import ClientSession, WSMsgType, web


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.realtime.camera_topology import CameraTopology  # noqa: E402
from src.realtime.identity_gallery import IdentityGallery  # noqa: E402
from src.realtime.protocol import (  # noqa: E402
    CAMERA_ID_PATTERN,
    encode_jpeg,
    pack_frame,
    unpack_frame,
)
from src.runtime_config import (  # noqa: E402
    load_prime_pipeline_configs,
    load_realtime_config,
    load_runtime_environment,
    load_yolo_config,
    pipeline_role,
    redact_url,
    runtime_paths,
)


@dataclass
class SmokeCheck:
    name: str
    ok: bool
    detail: str
    required: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--load-models", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args()


def add_check(
    checks: list[SmokeCheck],
    name: str,
    ok: bool,
    detail: str,
    *,
    required: bool = True,
) -> None:
    checks.append(SmokeCheck(name, bool(ok), detail, required))


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def validate_url(
    value: str,
    schemes: set[str],
    *,
    require_port: bool = True,
) -> tuple[bool, str]:
    parsed = urlparse(value)
    try:
        port = parsed.port
    except ValueError:
        return False, redact_url(value)
    ok = (
        parsed.scheme in schemes
        and bool(parsed.hostname)
        and (not require_port or port is not None)
    )
    return ok, redact_url(value)


def validate_configuration(
    checks: list[SmokeCheck],
    role: str,
    realtime: dict[str, Any],
    yolo: dict[str, Any],
    pipeline_configs: dict[str, Any] | None,
) -> None:
    network = realtime.get("network", {})
    worker = realtime.get("worker", {})
    prime = realtime.get("prime", {})
    worker_control = realtime.get("control", {}).get("worker", {})

    prime_url = str(network.get("prime_url", ""))
    ok, detail = validate_url(prime_url, {"ws", "wss"})
    add_check(checks, "prime_url", ok, detail)
    prime_host = (urlparse(prime_url).hostname or "").lower()
    loopback_allowed = parse_bool(
        os.environ.get("ALLOW_LOOPBACK_PRIME"),
        False,
    )
    add_check(
        checks,
        "worker_prime_address",
        role != "worker"
        or loopback_allowed
        or prime_host not in {"127.0.0.1", "::1", "localhost"},
        f"host={prime_host or '-'} loopback_allowed={loopback_allowed}",
    )

    prime_port = int(network.get("prime_port", 0))
    control_port = int(worker_control.get("port", 0))
    add_check(checks, "prime_port", 1 <= prime_port <= 65535, str(prime_port))
    add_check(
        checks,
        "worker_control_port",
        1 <= control_port <= 65535,
        str(control_port),
    )

    target_fps = float(worker.get("target_fps", 0))
    capture_fps = float(worker.get("capture_fps", 0))
    add_check(
        checks,
        "camera_rates",
        target_fps > 0 and capture_fps >= target_fps,
        f"target={target_fps:g} capture={capture_fps:g}",
    )
    for name, value in (
        ("frame_jpeg_quality", int(worker.get("frame_jpeg_quality", 0))),
        ("crop_jpeg_quality", int(worker.get("crop_jpeg_quality", 0))),
        ("viewer_jpeg_quality", int(prime.get("viewer_jpeg_quality", 0))),
    ):
        add_check(checks, name, 1 <= value <= 100, str(value))

    camera_ids = [str(item) for item in worker_control.get("camera_ids", [])]
    camera_sources = [str(item) for item in worker_control.get("sources", [])]
    local_camera = role == "worker" or parse_bool(
        os.environ.get("LOCAL_CAMERA_ENABLED"),
        True,
    )
    add_check(
        checks,
        "camera_ids",
        not local_camera
        or (
            bool(camera_ids)
            and len(camera_ids) == len(set(camera_ids))
            and all(CAMERA_ID_PATTERN.fullmatch(item) for item in camera_ids)
        ),
        f"count={len(camera_ids)} unique={len(set(camera_ids))}",
    )
    auto_scan = bool(worker_control.get("auto_scan", True))
    source_mapping_ok = auto_scan or (
        bool(camera_sources) and len(camera_sources) == len(camera_ids)
    )
    add_check(
        checks,
        "camera_source_mapping",
        not local_camera or source_mapping_ok,
        (
            f"auto_scan={auto_scan} ids={len(camera_ids)} "
            f"configured_sources={len(camera_sources)}"
        ),
    )

    yolo_detection = yolo.get("detection", {})
    yolo_confidence = float(yolo_detection.get("conf_threshold", -1))
    yolo_iou = float(yolo_detection.get("iou_threshold", -1))
    add_check(
        checks,
        "yolo_thresholds",
        0 < yolo_confidence <= 1 and 0 < yolo_iou <= 1,
        f"confidence={yolo_confidence:g} iou={yolo_iou:g}",
    )

    try:
        CameraTopology.from_config(prime)
        add_check(checks, "camera_topology", True, "valid")
    except Exception as exc:
        add_check(
            checks,
            "camera_topology",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    worker_nodes = realtime.get("control", {}).get("worker_nodes", [])
    node_errors = []
    node_names = []
    for node in worker_nodes:
        node_names.append(str(node.get("name", "")))
        valid, _ = validate_url(str(node.get("url", "")), {"http", "https"})
        if not node.get("name") or not valid:
            node_errors.append(str(node.get("name", "<unnamed>")))
    if len(node_names) != len(set(node_names)):
        node_errors.append("duplicate names")
    add_check(
        checks,
        "worker_nodes",
        not node_errors,
        f"count={len(worker_nodes)} invalid={node_errors}",
    )

    if pipeline_configs is not None:
        reid = pipeline_configs["reid"]
        tracker = pipeline_configs["tracker"]
        shape = reid.get("model", {}).get("input_shape", [])
        add_check(
            checks,
            "reid_contract",
            (
                reid.get("backend") in {
                    "onnxruntime_direct",
                    "onnxruntime",
                    "onnx",
                    "tensorrt_direct",
                    "tensorrt",
                    "triton",
                }
                and isinstance(shape, list)
                and len(shape) == 2
                and all(int(item) > 0 for item in shape)
                and int(reid.get("model", {}).get("embedding_dim", 0)) > 0
            ),
            (
                f"backend={reid.get('backend')} input_shape={shape} "
                f"embedding_dim={reid.get('model', {}).get('embedding_dim')}"
            ),
        )
        add_check(
            checks,
            "tracker_contract",
            int(tracker.get("botsort", {}).get("track_buffer", 0)) > 0,
            (
                f"buffer={tracker.get('botsort', {}).get('track_buffer')} "
                f"device={tracker.get('device')}"
            ),
        )


def check_imports(checks: list[SmokeCheck]) -> None:
    for name in ("yaml", "numpy", "cv2", "aiohttp"):
        try:
            module = importlib.import_module(name)
            add_check(
                checks,
                f"import:{name}",
                True,
                str(getattr(module, "__version__", "available")),
            )
        except Exception as exc:
            add_check(
                checks,
                f"import:{name}",
                False,
                f"{type(exc).__name__}: {exc}",
            )


def protocol_packet() -> bytes:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[8:42, 20:44] = (30, 120, 220)
    crop = frame[8:42, 20:44]
    detections = np.asarray(
        [[20, 8, 44, 42, 0.91, 0]],
        dtype=np.float32,
    )
    return pack_frame(
        camera_id="smoke-cam",
        frame_id=7,
        detections=detections,
        frame_jpeg=encode_jpeg(frame, 75),
        crop_jpegs=[encode_jpeg(crop, 85)],
        width=frame.shape[1],
        height=frame.shape[0],
        timestamp=1234.5,
    )


async def websocket_roundtrip(packet: bytes) -> dict[str, Any]:
    received: dict[str, Any] = {}

    async def ingest(request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        message = await websocket.receive()
        if message.type == WSMsgType.BINARY:
            decoded = unpack_frame(message.data, received_at=1234.6)
            received.update(
                {
                    "camera_id": decoded.camera_id,
                    "frame_id": decoded.frame_id,
                    "detections": len(decoded.detections),
                    "crops": len(decoded.crop_jpegs),
                }
            )
            await websocket.send_json(received)
        await websocket.close()
        return websocket

    app = web.Application()
    app.router.add_get("/ws/ingest", ingest)
    runner = web.AppRunner(app)
    await runner.setup()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    try:
        async with ClientSession() as session:
            async with session.ws_connect(
                f"http://127.0.0.1:{port}/ws/ingest"
            ) as websocket:
                await websocket.send_bytes(packet)
                response = await websocket.receive_json()
    finally:
        await runner.cleanup()
    return response


def check_data_path(checks: list[SmokeCheck]) -> None:
    try:
        packet = protocol_packet()
        decoded = unpack_frame(packet, received_at=1234.6)
        ok = (
            decoded.camera_id == "smoke-cam"
            and decoded.frame_id == 7
            and decoded.detections.shape == (1, 6)
            and len(decoded.crop_jpegs) == 1
        )
        add_check(
            checks,
            "binary_protocol",
            ok,
            (
                f"bytes={len(packet)} detections={len(decoded.detections)} "
                f"crops={len(decoded.crop_jpegs)}"
            ),
        )

        response = asyncio.run(websocket_roundtrip(packet))
        add_check(
            checks,
            "websocket_roundtrip",
            response
            == {
                "camera_id": "smoke-cam",
                "frame_id": 7,
                "detections": 1,
                "crops": 1,
            },
            json.dumps(response, sort_keys=True),
        )
    except Exception as exc:
        add_check(
            checks,
            "realtime_data_path",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    try:
        gallery = IdentityGallery(match_threshold=0.3)
        first = gallery.assign(
            "cam1",
            1,
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            timestamp=1.0,
        )
        same_person = gallery.assign(
            "cam2",
            7,
            np.asarray([0.99, 0.01, 0.0, 0.0], dtype=np.float32),
            timestamp=2.0,
        )
        different_person = gallery.assign(
            "cam2",
            8,
            np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            timestamp=2.1,
            blocked_global_ids={same_person},
        )
        add_check(
            checks,
            "global_gallery",
            first == same_person and different_person != first,
            (
                f"cam1_id={first} matched_cam2_id={same_person} "
                f"different_person_id={different_person}"
            ),
        )
    except Exception as exc:
        add_check(
            checks,
            "global_gallery",
            False,
            f"{type(exc).__name__}: {exc}",
        )


def resolve_model_path(
    configured: str,
    search_paths: list[str] | None = None,
) -> Path:
    path = Path(configured).expanduser()
    candidates = [path if path.is_absolute() else ROOT / path]
    for search_path in search_paths or []:
        directory = Path(search_path).expanduser()
        if not directory.is_absolute():
            directory = ROOT / directory
        candidates.append(directory / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def check_models(
    checks: list[SmokeCheck],
    role: str,
    realtime: dict[str, Any],
    yolo: dict[str, Any],
    pipeline_configs: dict[str, Any] | None,
    *,
    load_models: bool,
) -> None:
    local_camera = role == "worker" or parse_bool(
        os.environ.get("LOCAL_CAMERA_ENABLED"),
        True,
    )
    yolo_path = resolve_model_path(str(yolo.get("model", {}).get("path", "")))
    add_check(
        checks,
        "yolo_model_file",
        yolo_path.is_file(),
        str(yolo_path),
        required=load_models and local_camera,
    )

    if pipeline_configs is not None:
        reid = pipeline_configs["reid"]
        reid_model = reid.get("model", {})
        reid_path = resolve_model_path(
            str(reid_model.get("onnx_path", "")),
            [str(item) for item in reid_model.get("search_paths", [])],
        )
        add_check(
            checks,
            "reid_model_file",
            reid_path.is_file(),
            str(reid_path),
            required=load_models,
        )
    else:
        reid_path = None

    if not load_models:
        return

    if local_camera and yolo_path.is_file():
        try:
            from src.detector import YOLOPersonDetector

            detector = YOLOPersonDetector(yolo)
            height = int(realtime.get("worker", {}).get("capture_height", 480))
            width = int(realtime.get("worker", {}).get("capture_width", 640))
            detections, crops = detector.detect(
                np.zeros((height, width, 3), dtype=np.uint8)
            )
            if detector.backend == "ultralytics":
                actual_device = str(next(detector.model.model.parameters()).device)
            else:
                actual_device = f"cuda:{detector.model.device_id}"
            gpu_required = parse_bool(
                os.environ.get("YOLO_REQUIRE_GPU"),
                True,
            )
            add_check(
                checks,
                "yolo_inference",
                len(detections) == len(crops)
                and (not gpu_required or actual_device.startswith("cuda")),
                (
                    f"backend={detector.backend} configured={detector.device} "
                    f"actual={actual_device} gpu_required={gpu_required} "
                    f"detections={len(detections)}"
                ),
            )
        except Exception as exc:
            add_check(
                checks,
                "yolo_inference",
                False,
                f"{type(exc).__name__}: {exc}",
            )

    if pipeline_configs is not None and reid_path is not None and reid_path.is_file():
        client = None
        try:
            from src.reid_client import create_reid_client

            reid = pipeline_configs["reid"]
            client = create_reid_client(reid)
            height, width = [int(item) for item in reid["model"]["input_shape"]]
            embeddings = client.infer(
                [np.zeros((height, width, 3), dtype=np.uint8)],
                max_batch_size=1,
            )
            metadata = client.get_model_metadata() or {}
            providers = metadata.get("providers", [])
            gpu_required = parse_bool(
                os.environ.get("REID_REQUIRE_GPU"),
                True,
            )
            provider_ok = (
                not gpu_required
                or reid.get("backend") in {"tensorrt", "tensorrt_direct"}
                or "CUDAExecutionProvider" in providers
            )
            expected = (1, int(reid["model"]["embedding_dim"]))
            add_check(
                checks,
                "reid_inference",
                embeddings.shape == expected
                and bool(np.all(np.isfinite(embeddings)))
                and provider_ok,
                (
                    f"shape={embeddings.shape} providers={providers} "
                    f"gpu_required={gpu_required}"
                ),
            )
        except Exception as exc:
            add_check(
                checks,
                "reid_inference",
                False,
                f"{type(exc).__name__}: {exc}",
            )
        finally:
            if client is not None:
                client.close()

        try:
            from src.tracker import ReIDTracker

            tracker = ReIDTracker(pipeline_configs["tracker"])
            add_check(
                checks,
                "tracker_initialization",
                tracker is not None,
                type(tracker).__name__,
            )
        except Exception as exc:
            add_check(
                checks,
                "tracker_initialization",
                False,
                f"{type(exc).__name__}: {exc}",
            )


def run_smoke(args: argparse.Namespace) -> tuple[str, Path, list[SmokeCheck]]:
    checks: list[SmokeCheck] = []
    env_path = load_runtime_environment(args.env_file, required=True)
    role = pipeline_role()
    paths = runtime_paths(role)
    add_check(checks, "environment_file", True, str(env_path))

    selected_paths = {
        "realtime_config": paths.realtime,
        "yolo_config": paths.yolo,
        "model_manifest": paths.model_manifest,
    }
    if role == "prime":
        selected_paths.update(
            {
                "reid_config": paths.reid,
                "tracker_config": paths.tracker,
                "pipeline_config": paths.pipeline,
            }
        )
    for name, path in selected_paths.items():
        add_check(checks, name, path.is_file(), str(path))

    try:
        realtime = load_realtime_config(paths.realtime)
        yolo = load_yolo_config(paths.yolo)
        pipeline_configs = (
            load_prime_pipeline_configs(paths.realtime.parent)
            if role == "prime"
            else None
        )
        validate_configuration(checks, role, realtime, yolo, pipeline_configs)
    except Exception as exc:
        add_check(
            checks,
            "configuration_load",
            False,
            f"{type(exc).__name__}: {exc}",
        )
        return role, env_path, checks

    check_imports(checks)
    check_data_path(checks)
    check_models(
        checks,
        role,
        realtime,
        yolo,
        pipeline_configs,
        load_models=args.load_models,
    )
    return role, env_path, checks


def main() -> None:
    args = parse_args()
    role, env_path, checks = run_smoke(args)
    failed = [check for check in checks if check.required and not check.ok]

    if args.json_output:
        print(
            json.dumps(
                {
                    "ok": not failed,
                    "role": role,
                    "environment_file": str(env_path),
                    "checks": [asdict(check) for check in checks],
                },
                indent=2,
            )
        )
    else:
        print(f"Realtime smoke test role={role}")
        print(f"Environment: {env_path}")
        for check in checks:
            status = "PASS" if check.ok else ("FAIL" if check.required else "WARN")
            print(f"[{status}] {check.name}: {check.detail}")
        print(
            f"Result: {'READY' if not failed else 'NOT READY'} "
            f"({len(failed)} required failures)"
        )

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
