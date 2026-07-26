"""
Binary WebSocket protocol for realtime Jetson-to-prime frame transport.

Packet layout:
    magic:      4 bytes, b"RTP1"
    header_len: 4 bytes, big-endian unsigned integer
    header:     UTF-8 JSON
    payload:    frame JPEG followed by crop JPEGs

The header contains byte lengths for each payload section, so receivers can
slice the payload without scanning for delimiters.
"""

from __future__ import annotations

import json
import io
import re
import struct
import time
from dataclasses import dataclass
from os import PathLike
from typing import Any

import numpy as np
from PIL import Image


MAGIC = b"RTP1"
HEADER_STRUCT = struct.Struct(">4sI")
MAX_HEADER_SIZE = 4 * 1024 * 1024
MAX_DETECTIONS = 512
CAMERA_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True)
class FramePacket:
    camera_id: str
    frame_id: int
    timestamp: float
    width: int
    height: int
    detections: np.ndarray
    frame_jpeg: bytes
    crop_jpegs: list[bytes]
    received_at: float | None = None


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    """Encode a BGR image to JPEG bytes."""
    quality = max(1, min(100, int(quality)))
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError(f"Expected a non-empty BGR HxWx3 image, got {getattr(image, 'shape', None)}")
    output = io.BytesIO()
    Image.fromarray(image[:, :, ::-1], mode="RGB").save(output, format="JPEG", quality=quality)
    return output.getvalue()


def decode_jpeg(data: bytes) -> np.ndarray:
    """Decode JPEG bytes to a BGR image."""
    if not data:
        raise ValueError("JPEG payload must not be empty")
    try:
        with Image.open(io.BytesIO(data)) as image:
            rgb = np.asarray(image.convert("RGB"))
    except Exception as exc:
        raise RuntimeError("Failed to JPEG-decode image") from exc
    return np.ascontiguousarray(rgb[:, :, ::-1])


def write_jpeg(path: str | PathLike[str], image: np.ndarray, quality: int = 90) -> None:
    """Write a BGR image with the same stable codec used by transport."""
    with open(path, "wb") as output:
        output.write(encode_jpeg(image, quality))


def pack_frame(
    camera_id: str,
    frame_id: int,
    detections: np.ndarray,
    frame_jpeg: bytes,
    crop_jpegs: list[bytes],
    width: int,
    height: int,
    timestamp: float | None = None,
) -> bytes:
    """Serialize one frame packet for binary WebSocket transport."""
    camera_id = _validate_camera_id(camera_id)
    detections = np.asarray(detections, dtype=np.float32)
    if detections.size == 0:
        detections = np.empty((0, 6), dtype=np.float32)
        detections_list: list[list[float]] = []
    else:
        if detections.ndim != 2 or detections.shape[1] != 6:
            raise ValueError(f"Detections must have shape [N, 6], got {detections.shape}")
        if not np.all(np.isfinite(detections)):
            raise ValueError("Detections must contain only finite values")
        detections_list = detections.astype(float).tolist()
    if len(detections) > MAX_DETECTIONS:
        raise ValueError(f"Too many detections: {len(detections)} > {MAX_DETECTIONS}")
    if len(crop_jpegs) != len(detections):
        raise ValueError(
            f"Crop/detection count mismatch: {len(crop_jpegs)} crops for {len(detections)} detections"
        )
    if int(frame_id) < 0:
        raise ValueError("frame_id must be non-negative")
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError(f"Frame dimensions must be positive, got {width}x{height}")
    if not frame_jpeg:
        raise ValueError("frame_jpeg must not be empty")
    if any(not crop for crop in crop_jpegs):
        raise ValueError("crop_jpegs must not contain empty payloads")
    event_timestamp = float(time.time() if timestamp is None else timestamp)
    if not np.isfinite(event_timestamp):
        raise ValueError("timestamp must be finite")

    header: dict[str, Any] = {
        "camera_id": camera_id,
        "frame_id": int(frame_id),
        "timestamp": event_timestamp,
        "width": int(width),
        "height": int(height),
        "detections": detections_list,
        "frame_size": len(frame_jpeg),
        "crop_sizes": [len(crop) for crop in crop_jpegs],
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload = frame_jpeg + b"".join(crop_jpegs)
    return HEADER_STRUCT.pack(MAGIC, len(header_bytes)) + header_bytes + payload


def unpack_frame(data: bytes, received_at: float | None = None) -> FramePacket:
    """Deserialize one binary frame packet."""
    if len(data) < HEADER_STRUCT.size:
        raise ValueError("Frame packet too short")

    magic, header_len = HEADER_STRUCT.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("Invalid realtime packet magic")
    if header_len <= 0 or header_len > MAX_HEADER_SIZE:
        raise ValueError(f"Invalid realtime header length: {header_len}")

    header_start = HEADER_STRUCT.size
    header_end = header_start + header_len
    if len(data) < header_end:
        raise ValueError("Incomplete realtime packet header")

    header = json.loads(data[header_start:header_end].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("Realtime packet header must be a JSON object")
    payload = memoryview(data)[header_end:]

    frame_size = int(header["frame_size"])
    crop_sizes = [int(size) for size in header.get("crop_sizes", [])]
    if frame_size <= 0 or any(size <= 0 for size in crop_sizes):
        raise ValueError("Frame and crop payload sizes must be positive")
    expected_payload_size = frame_size + sum(crop_sizes)
    if len(payload) != expected_payload_size:
        raise ValueError(
            f"Invalid payload length: got {len(payload)}, expected {expected_payload_size}"
        )

    offset = 0
    frame_jpeg = bytes(payload[offset:offset + frame_size])
    offset += frame_size

    crop_jpegs = []
    for crop_size in crop_sizes:
        crop_jpegs.append(bytes(payload[offset:offset + crop_size]))
        offset += crop_size

    raw_detections = header.get("detections", [])
    if not isinstance(raw_detections, list):
        raise ValueError("Detections must be a JSON list")
    detections = np.asarray(raw_detections, dtype=np.float32)
    if len(raw_detections) == 0:
        detections = np.empty((0, 6), dtype=np.float32)
    else:
        if detections.ndim != 2 or detections.shape[1] != 6:
            raise ValueError(f"Detections must have shape [N, 6], got {detections.shape}")
    if len(detections) > MAX_DETECTIONS:
        raise ValueError(f"Too many detections: {len(detections)} > {MAX_DETECTIONS}")
    if len(crop_sizes) != len(detections):
        raise ValueError(
            f"Crop/detection count mismatch: {len(crop_sizes)} crops for {len(detections)} detections"
        )
    if not np.all(np.isfinite(detections)):
        raise ValueError("Detections must contain only finite values")

    camera_id = _validate_camera_id(header["camera_id"])
    frame_id = int(header["frame_id"])
    width = int(header["width"])
    height = int(header["height"])
    timestamp = float(header["timestamp"])
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    if width <= 0 or height <= 0:
        raise ValueError(f"Frame dimensions must be positive, got {width}x{height}")
    if not np.isfinite(timestamp):
        raise ValueError("timestamp must be finite")

    return FramePacket(
        camera_id=camera_id,
        frame_id=frame_id,
        timestamp=timestamp,
        width=width,
        height=height,
        detections=detections,
        frame_jpeg=frame_jpeg,
        crop_jpegs=crop_jpegs,
        received_at=float(time.time() if received_at is None else received_at),
    )


def _validate_camera_id(value: Any) -> str:
    camera_id = str(value)
    if CAMERA_ID_PATTERN.fullmatch(camera_id) is None:
        raise ValueError(
            "camera_id must be 1-64 characters using letters, digits, '.', '_', or '-', "
            "and must start with a letter or digit"
        )
    return camera_id
