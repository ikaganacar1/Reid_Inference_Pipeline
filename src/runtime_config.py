"""Runtime configuration loaded from one repository-local .env file.

YAML files remain the checked-in defaults. Environment variables contain the
device-specific values and override those defaults with explicit type parsing.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
from typing import Any, Callable

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = ROOT / ".env"
ENV_FILE_VARIABLE = "REID_ENV_FILE"
_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved files selected by the current device role."""

    role: str
    realtime: Path
    yolo: Path
    yoloe: Path
    reid: Path
    tracker: Path
    pipeline: Path
    model_manifest: Path


@dataclass(frozen=True)
class EnvOverride:
    name: str
    path: tuple[str, ...]
    parser: Callable[[str], Any]


def load_runtime_environment(
    env_file: Path | str | None = None,
    *,
    override: bool = False,
    required: bool = False,
) -> Path:
    """Load a small, shell-compatible subset of dotenv syntax.

    Existing process variables win unless ``override`` is true. Startup shell
    scripts source the same file before Python starts, while this loader keeps
    direct Python entrypoints consistent.
    """

    selected = Path(
        env_file or os.environ.get(ENV_FILE_VARIABLE) or DEFAULT_ENV_FILE
    ).expanduser()
    if not selected.is_absolute():
        selected = (ROOT / selected).resolve()
    else:
        selected = selected.resolve()

    if not selected.is_file():
        if required:
            raise FileNotFoundError(
                f"Runtime environment file not found: {selected}. "
                "Run scripts/reidctl.sh init first."
            )
        return selected

    for line_number, raw_line in enumerate(
        selected.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        parsed = _parse_env_line(raw_line, selected, line_number)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value

    if env_file is not None or override or ENV_FILE_VARIABLE not in os.environ:
        os.environ[ENV_FILE_VARIABLE] = str(selected)
    return selected


def _parse_env_line(
    raw_line: str,
    path: Path,
    line_number: int,
) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[7:].lstrip()
    if "=" not in line:
        raise ValueError(f"{path}:{line_number}: expected KEY=value")

    key, raw_value = line.split("=", 1)
    key = key.strip()
    if _ENV_KEY.fullmatch(key) is None:
        raise ValueError(f"{path}:{line_number}: invalid environment key {key!r}")

    try:
        values = shlex.split(raw_value, comments=True, posix=True)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if len(values) > 1:
        raise ValueError(
            f"{path}:{line_number}: values containing spaces must be quoted"
        )
    return key, values[0] if values else ""


def pipeline_role(value: str | None = None) -> str:
    role = str(value or os.environ.get("PIPELINE_ROLE", "prime")).strip().lower()
    if role not in {"prime", "worker"}:
        raise ValueError("PIPELINE_ROLE must be 'prime' or 'worker'")
    return role


def runtime_paths(
    role: str | None = None,
    config_dir: Path | str | None = None,
) -> RuntimePaths:
    selected_role = pipeline_role(role)
    directory = _repo_path(
        os.environ.get("CONFIG_DIR") or config_dir or "configs"
    )
    realtime_default = (
        directory / "realtime_config.yaml"
        if selected_role == "prime"
        else directory / "realtime_config.worker.yaml"
    )
    yolo_default = (
        directory / "yolo_config.yaml"
        if selected_role == "prime"
        else directory / "yolo_config.worker.yaml"
    )
    return RuntimePaths(
        role=selected_role,
        realtime=_env_path("REALTIME_CONFIG", realtime_default),
        yolo=_env_path("YOLO_CONFIG", yolo_default),
        yoloe=_env_path("YOLOE_CONFIG", directory / "yoloe_config.yaml"),
        reid=_env_path("REID_CONFIG", directory / "reid_config.yaml"),
        tracker=_env_path("TRACKER_CONFIG", directory / "tracker_config.yaml"),
        pipeline=_env_path("PIPELINE_CONFIG", directory / "pipeline_config.yaml"),
        model_manifest=_env_path(
            "MODEL_MANIFEST",
            ROOT / "deploy" / "model_manifest.yaml",
        ),
    )


def load_yaml_mapping(path: Path | str) -> dict[str, Any]:
    selected = _repo_path(path)
    if not selected.is_file():
        raise FileNotFoundError(f"Configuration file not found: {selected}")
    with selected.open(encoding="utf-8") as config_file:
        value = yaml.safe_load(config_file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {selected}")
    return value


def load_realtime_config(path: Path | str) -> dict[str, Any]:
    return apply_realtime_overrides(load_yaml_mapping(path))


def load_yolo_config(path: Path | str) -> dict[str, Any]:
    return apply_yolo_overrides(load_yaml_mapping(path))


def load_reid_config(path: Path | str) -> dict[str, Any]:
    return apply_reid_overrides(load_yaml_mapping(path))


def load_tracker_config(path: Path | str) -> dict[str, Any]:
    return apply_tracker_overrides(load_yaml_mapping(path))


def load_prime_pipeline_configs(
    config_dir: Path | str | None = None,
) -> dict[str, Any]:
    paths = runtime_paths("prime", config_dir)
    configs = {
        "yolo": load_yolo_config(paths.yolo),
        "yoloe": load_yaml_mapping(paths.yoloe),
        "reid": load_reid_config(paths.reid),
        "tracker": load_tracker_config(paths.tracker),
        "pipeline": load_yaml_mapping(paths.pipeline),
    }
    return configs


def apply_realtime_overrides(config: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(config)
    _apply_overrides(value, _REALTIME_OVERRIDES)

    camera_ids = _optional_list("CAMERA_IDS")
    if camera_ids is not None:
        _set_nested(value, ("control", "worker", "camera_ids"), camera_ids)
        if len(camera_ids) == 1 and "CAMERA_ID" not in os.environ:
            _set_nested(value, ("worker", "camera_id"), camera_ids[0])

    camera_sources = _optional_list("CAMERA_SOURCES")
    if camera_sources is not None:
        _set_nested(
            value,
            ("control", "worker", "sources"),
            [_camera_source(item) for item in camera_sources],
        )

    worker_nodes = os.environ.get("WORKER_NODES")
    if worker_nodes is not None:
        _set_nested(value, ("control", "worker_nodes"), _parse_worker_nodes(worker_nodes))

    overlapping = os.environ.get("OVERLAPPING_CAMERA_PAIRS")
    if overlapping is not None:
        _set_nested(
            value,
            ("prime", "overlapping_camera_pairs"),
            _parse_camera_pairs(overlapping),
        )

    adjacent = os.environ.get("ADJACENT_CAMERA_PAIRS")
    if adjacent is not None:
        _set_nested(
            value,
            ("prime", "adjacent_camera_pairs"),
            _parse_camera_pairs(adjacent),
        )
    return value


def apply_yolo_overrides(config: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(config)
    _apply_overrides(value, _YOLO_OVERRIDES)
    return value


def apply_reid_overrides(config: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(config)
    _apply_overrides(value, _REID_OVERRIDES)

    input_shape = list(value.get("model", {}).get("input_shape", [256, 128]))
    if len(input_shape) != 2:
        raise ValueError("ReID model.input_shape must contain [height, width]")
    if os.environ.get("REID_INPUT_HEIGHT"):
        input_shape[0] = _parse_int(os.environ["REID_INPUT_HEIGHT"])
    if os.environ.get("REID_INPUT_WIDTH"):
        input_shape[1] = _parse_int(os.environ["REID_INPUT_WIDTH"])
    _set_nested(value, ("model", "input_shape"), input_shape)

    if (
        os.environ.get("REID_BATCH_SIZE")
        and not os.environ.get("REID_MAX_BATCH_SIZE")
    ):
        _set_nested(
            value,
            ("inference", "max_batch_size"),
            _parse_int(os.environ["REID_BATCH_SIZE"]),
        )

    providers = _optional_list("REID_PROVIDERS")
    if providers is not None:
        _set_nested(value, ("onnxruntime", "providers"), providers)
    return value


def apply_tracker_overrides(config: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(config)
    _apply_overrides(value, _TRACKER_OVERRIDES)
    return value


def redact_url(value: str) -> str:
    """Hide URL user information before displaying runtime configuration."""
    return re.sub(r"(://)[^/@\s]+@", r"\1***@", value)


def _apply_overrides(
    config: dict[str, Any],
    overrides: tuple[EnvOverride, ...],
) -> None:
    for override in overrides:
        raw_value = os.environ.get(override.name)
        if raw_value is None or raw_value == "":
            continue
        try:
            parsed = override.parser(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {override.name}={raw_value!r}: {exc}") from exc
        _set_nested(config, override.path, parsed)


def _set_nested(
    config: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    target = config
    for key in path[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            child = {}
            target[key] = child
        target = child
    target[path[-1]] = value


def _repo_path(value: Path | str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _env_path(name: str, default: Path | str) -> Path:
    raw_value = os.environ.get(name)
    return _repo_path(raw_value if raw_value else default)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("expected true/false, yes/no, on/off, or 1/0")


def _parse_int(value: str) -> int:
    return int(value.strip())


def _parse_float(value: str) -> float:
    return float(value.strip())


def _camera_source(value: str) -> int | str:
    stripped = value.strip()
    return int(stripped) if stripped.isdigit() else stripped


def _optional_list(name: str) -> list[str] | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_camera_pairs(value: str) -> list[list[str]]:
    pairs = []
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        camera_ids = [camera_id.strip() for camera_id in item.split(":")]
        if len(camera_ids) != 2 or not all(camera_ids):
            raise ValueError(
                "camera pairs must use cam1:cam2,cam3:cam4 syntax"
            )
        if camera_ids[0] == camera_ids[1]:
            raise ValueError("a camera cannot be paired with itself")
        pairs.append(camera_ids)
    return pairs


def _parse_worker_nodes(value: str) -> list[dict[str, str]]:
    nodes = []
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        name, separator, url = item.partition("=")
        if not separator or not name.strip() or not url.strip():
            raise ValueError(
                "WORKER_NODES must use name=http://host:port syntax"
            )
        nodes.append({"name": name.strip(), "url": url.strip().rstrip("/")})
    return nodes


_REALTIME_OVERRIDES = (
    EnvOverride("PRIME_URL", ("network", "prime_url"), str),
    EnvOverride("PRIME_BIND_HOST", ("network", "prime_bind_host"), str),
    EnvOverride("PRIME_PORT", ("network", "prime_port"), _parse_int),
    EnvOverride("INGEST_PATH", ("network", "ingest_path"), str),
    EnvOverride("VIEWER_PATH", ("network", "viewer_path"), str),
    EnvOverride("CAMERA_ID", ("worker", "camera_id"), str),
    EnvOverride("CAMERA_SOURCE", ("worker", "source"), _camera_source),
    EnvOverride("CAMERA_TARGET_FPS", ("worker", "target_fps"), _parse_float),
    EnvOverride("CAMERA_CAPTURE_WIDTH", ("worker", "capture_width"), _parse_int),
    EnvOverride("CAMERA_CAPTURE_HEIGHT", ("worker", "capture_height"), _parse_int),
    EnvOverride("CAMERA_CAPTURE_FPS", ("worker", "capture_fps"), _parse_float),
    EnvOverride("FRAME_JPEG_QUALITY", ("worker", "frame_jpeg_quality"), _parse_int),
    EnvOverride("CROP_JPEG_QUALITY", ("worker", "crop_jpeg_quality"), _parse_int),
    EnvOverride(
        "CAMERA_RECONNECT_SECONDS",
        ("worker", "reconnect_seconds"),
        _parse_float,
    ),
    EnvOverride(
        "CAMERA_PRINT_EVERY_FRAMES",
        ("worker", "print_every_frames"),
        _parse_int,
    ),
    EnvOverride("REALTIME_OUTPUT_DIR", ("prime", "output_dir"), str),
    EnvOverride("SAVE_VIDEO", ("prime", "save_video"), _parse_bool),
    EnvOverride("OUTPUT_FPS", ("prime", "output_fps"), _parse_float),
    EnvOverride(
        "RECORDING_SEGMENT_SECONDS",
        ("prime", "recording_segment_seconds"),
        _parse_float,
    ),
    EnvOverride(
        "RECORDING_MIN_FREE_GB",
        ("prime", "recording_min_free_gb"),
        _parse_float,
    ),
    EnvOverride(
        "RECORDING_DISK_CHECK_SECONDS",
        ("prime", "recording_disk_check_seconds"),
        _parse_float,
    ),
    EnvOverride(
        "VIEWER_JPEG_QUALITY",
        ("prime", "viewer_jpeg_quality"),
        _parse_int,
    ),
    EnvOverride(
        "VIEWER_SEND_TIMEOUT_SECONDS",
        ("prime", "viewer_send_timeout_seconds"),
        _parse_float,
    ),
    EnvOverride(
        "CAMERA_OFFLINE_SECONDS",
        ("prime", "camera_offline_seconds"),
        _parse_float,
    ),
    EnvOverride("REID_BATCH_SIZE", ("prime", "reid_batch_size"), _parse_int),
    EnvOverride(
        "SYNC_BATCH_WINDOW_MS",
        ("prime", "sync_batch_window_ms"),
        _parse_float,
    ),
    EnvOverride(
        "SYNC_BATCH_MAX_PACKETS",
        ("prime", "sync_batch_max_packets"),
        _parse_int,
    ),
    EnvOverride("MAX_QUEUE_SIZE", ("prime", "max_queue_size"), _parse_int),
    EnvOverride(
        "CLIENT_MAX_SIZE_MB",
        ("prime", "client_max_size_mb"),
        _parse_int,
    ),
    EnvOverride(
        "MAX_CAPTURE_CLOCK_SKEW_SECONDS",
        ("prime", "max_capture_clock_skew_seconds"),
        _parse_float,
    ),
    EnvOverride(
        "CAMERA_TRACKER_RESET_SECONDS",
        ("prime", "camera_tracker_reset_seconds"),
        _parse_float,
    ),
    EnvOverride(
        "GLOBAL_MATCH_THRESHOLD",
        ("prime", "global_match_threshold"),
        _parse_float,
    ),
    EnvOverride(
        "GALLERY_TTL_SECONDS",
        ("prime", "gallery_ttl_seconds"),
        _parse_float,
    ),
    EnvOverride(
        "GALLERY_EMA_ALPHA",
        ("prime", "gallery_ema_alpha"),
        _parse_float,
    ),
    EnvOverride(
        "GALLERY_MAX_EXEMPLARS",
        ("prime", "gallery_max_exemplars"),
        _parse_int,
    ),
    EnvOverride("DEBUG_REID", ("prime", "debug_reid"), _parse_bool),
    EnvOverride(
        "SAVE_DEBUG_CROPS",
        ("prime", "save_debug_crops"),
        _parse_bool,
    ),
    EnvOverride(
        "NEW_IDENTITY_MIN_FRAMES",
        ("prime", "new_identity_min_frames"),
        _parse_int,
    ),
    EnvOverride(
        "NEW_IDENTITY_MIN_SECONDS",
        ("prime", "new_identity_min_seconds"),
        _parse_float,
    ),
    EnvOverride(
        "CROSS_CAMERA_EXCLUSION_SECONDS",
        ("prime", "cross_camera_exclusion_seconds"),
        _parse_float,
    ),
    EnvOverride(
        "ALLOW_ALL_CAMERA_OVERLAP",
        ("prime", "allow_all_camera_overlap"),
        _parse_bool,
    ),
    EnvOverride(
        "NEW_TRACK_MATCH_THRESHOLD",
        ("prime", "new_track_match_threshold"),
        _parse_float,
    ),
    EnvOverride(
        "IDENTITY_MIN_CONFIDENCE",
        ("prime", "identity_min_confidence"),
        _parse_float,
    ),
    EnvOverride(
        "WORKER_CONTROL_BIND_HOST",
        ("control", "worker", "bind_host"),
        str,
    ),
    EnvOverride(
        "WORKER_CONTROL_PORT",
        ("control", "worker", "port"),
        _parse_int,
    ),
    EnvOverride(
        "CAMERA_AUTO_SCAN",
        ("control", "worker", "auto_scan"),
        _parse_bool,
    ),
    EnvOverride(
        "CAMERA_START_ON_LAUNCH",
        ("control", "worker", "start_on_launch"),
        _parse_bool,
    ),
    EnvOverride(
        "CAMERA_AUTO_START_RETRY_SECONDS",
        ("control", "worker", "auto_start_retry_seconds"),
        _parse_float,
    ),
)


_YOLO_OVERRIDES = (
    EnvOverride("YOLO_MODEL_PATH", ("model", "path"), str),
    EnvOverride("YOLO_DEVICE", ("model", "device"), str),
    EnvOverride(
        "YOLO_CONFIDENCE",
        ("detection", "conf_threshold"),
        _parse_float,
    ),
    EnvOverride("YOLO_IOU_THRESHOLD", ("detection", "iou_threshold"), _parse_float),
    EnvOverride("YOLO_IMGSZ", ("detection", "imgsz"), _parse_int),
    EnvOverride("YOLO_HALF", ("inference", "half"), _parse_bool),
    EnvOverride("YOLO_MAX_DETECTIONS", ("inference", "max_det"), _parse_int),
)


_REID_OVERRIDES = (
    EnvOverride("REID_BACKEND", ("backend",), str),
    EnvOverride("REID_MODEL_PATH", ("model", "onnx_path"), str),
    EnvOverride("REID_EMBEDDING_DIM", ("model", "embedding_dim"), _parse_int),
    EnvOverride(
        "REID_MAX_BATCH_SIZE",
        ("inference", "max_batch_size"),
        _parse_int,
    ),
    EnvOverride(
        "REID_ALLOW_CPU_FALLBACK",
        ("onnxruntime", "allow_cpu_fallback"),
        _parse_bool,
    ),
    EnvOverride("REID_ENGINE_PATH", ("tensorrt", "engine_path"), str),
    EnvOverride("TRITON_SERVER_URL", ("triton", "server_url"), str),
    EnvOverride("TRITON_MODEL_NAME", ("triton", "model_name"), str),
)


_TRACKER_OVERRIDES = (
    EnvOverride("TRACKER_DEVICE", ("device",), str),
    EnvOverride("TRACKER_FP16", ("fp16",), _parse_bool),
    EnvOverride(
        "TRACKER_BUFFER_FRAMES",
        ("botsort", "track_buffer"),
        _parse_int,
    ),
    EnvOverride(
        "TRACKER_MATCH_THRESHOLD",
        ("botsort", "match_thresh"),
        _parse_float,
    ),
    EnvOverride(
        "TRACKER_APPEARANCE_THRESHOLD",
        ("botsort", "appearance_thresh"),
        _parse_float,
    ),
    EnvOverride(
        "TRACKER_PROXIMITY_THRESHOLD",
        ("botsort", "proximity_thresh"),
        _parse_float,
    ),
)
