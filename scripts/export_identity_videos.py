#!/usr/bin/env python3
"""Export one chronological person-crop video per global ReID identity."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


DEFAULT_RECORDINGS_ROOT = Path("recordings")


@dataclass
class CaptureState:
    capture: cv2.VideoCapture
    last_frame_id: int | None = None


@dataclass
class IdentityVideoState:
    global_id: int
    path: Path
    writer: cv2.VideoWriter
    observations: int = 0
    transition_frames: int = 0
    first_time: float | None = None
    last_time: float | None = None
    last_camera: str | None = None
    cameras: list[str] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class IdentityObservation:
    camera_id: str
    timestamp: float
    frame: np.ndarray
    record: dict[str, Any]
    policy_note: str | None
    touches_edge: bool
    quality: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, help="Path to offline_tracks.jsonl.")
    parser.add_argument("--recordings-root", type=Path, default=DEFAULT_RECORDINGS_ROOT)
    parser.add_argument("--session", default="session")
    parser.add_argument("--file-name", default="recording.mkv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--global-ids", default="", help="Optional comma-separated ID allow-list.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--header-height", type=int, default=88)
    parser.add_argument("--padding-ratio", type=float, default=0.08)
    parser.add_argument("--transition-seconds", type=float, default=0.6)
    parser.add_argument("--min-observations", type=int, default=5)
    parser.add_argument(
        "--apply-edge-remap-guard",
        action="store_true",
        help=(
            "Apply the current edge-remap policy when exporting an older event log. "
            "A remap triggered by a boundary-cut crop keeps the previous ID."
        ),
    )
    parser.add_argument("--edge-margin-ratio", type=float, default=0.01)
    return parser.parse_args()


def discover_recordings(root: Path, session: str, file_name: str) -> dict[str, Path]:
    session_dir = root / session
    recordings: dict[str, Path] = {}
    for path in sorted(session_dir.glob(f"cam*_ch*/{file_name}")):
        match = re.search(r"_ch(\d+)$", path.parent.name)
        if match:
            recordings[f"ch{match.group(1)}"] = path
    for path in sorted(session_dir.glob("channel_*.mkv")):
        match = re.match(r"channel_(\d+)(?:_|$)", path.stem)
        if match:
            recordings.setdefault(f"ch{match.group(1)}", path)
    return recordings


def touches_frame_edge(
    bbox: list[float],
    frame_width: int,
    frame_height: int,
    margin_ratio: float,
) -> bool:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    margin_x = frame_width * margin_ratio
    margin_y = frame_height * margin_ratio
    return (
        x1 <= margin_x
        or y1 <= margin_y
        or x2 >= frame_width - margin_x
        or y2 >= frame_height - margin_y
    )


def observation_quality(
    record: dict[str, Any],
    frame_width: int,
    frame_height: int,
    edge_margin_ratio: float = 0.01,
) -> tuple[bool, tuple[float, float, float, float]]:
    x1, y1, x2, y2 = [float(value) for value in record["bbox"]]
    visible_width = max(0.0, min(float(frame_width), x2) - max(0.0, x1))
    visible_height = max(0.0, min(float(frame_height), y2) - max(0.0, y1))
    edge = touches_frame_edge(
        record["bbox"],
        frame_width,
        frame_height,
        edge_margin_ratio,
    )
    height_ratio = visible_height / max(1.0, float(frame_height))
    area_ratio = (visible_width * visible_height) / max(
        1.0,
        float(frame_width * frame_height),
    )
    confidence = float(record.get("confidence", 0.0))
    return edge, (float(not edge), height_ratio, area_ratio, confidence)


def select_identity_observation(
    observations: list[IdentityObservation],
    current_camera: str | None,
) -> IdentityObservation:
    """Select one view per identity and timestamp without overlap-camera flicker."""
    if not observations:
        raise ValueError("At least one observation is required")

    current = [item for item in observations if item.camera_id == current_camera]
    if current:
        best_current = max(current, key=lambda item: item.quality)
        if not best_current.touches_edge:
            return best_current
        interior = [item for item in observations if not item.touches_edge]
        if interior:
            return max(interior, key=lambda item: item.quality)
        return best_current
    return max(observations, key=lambda item: item.quality)


def resolve_historical_global_id(
    camera_id: str,
    record: dict[str, Any],
    frame_width: int,
    frame_height: int,
    overrides: dict[tuple[str, int], int],
    apply_edge_remap_guard: bool,
    edge_margin_ratio: float = 0.01,
) -> tuple[int | None, str | None]:
    global_id = record.get("global_id")
    if global_id is None:
        return None, None
    global_id = int(global_id)
    if not apply_edge_remap_guard:
        return global_id, None

    local_track_id = int(record["local_track_id"])
    key = (camera_id, local_track_id)
    match = record.get("match") or {}
    reason = str(match.get("reason") or "")
    if reason.startswith("appearance_remap"):
        if touches_frame_edge(
            record["bbox"],
            frame_width,
            frame_height,
            edge_margin_ratio,
        ):
            previous_global_id = match.get("previous_global_id")
            if previous_global_id is not None:
                overrides[key] = int(previous_global_id)
                return int(previous_global_id), "edge_remap_suppressed"
        else:
            overrides.pop(key, None)

    if key in overrides:
        return overrides[key], "edge_remap_suppressed_hold"
    return global_id, None


def read_frame(state: CaptureState, frame_id: int) -> np.ndarray | None:
    capture = state.capture
    if state.last_frame_id is None or frame_id <= state.last_frame_id:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
    else:
        gap = frame_id - state.last_frame_id
        if gap <= 20:
            ok = True
            for _ in range(max(0, gap - 1)):
                ok = capture.grab()
                if not ok:
                    break
            ok, frame = capture.read() if ok else (False, None)
        else:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
    if not ok or frame is None:
        return None
    state.last_frame_id = frame_id
    return frame


def padded_crop(frame: np.ndarray, bbox: list[float], padding_ratio: float) -> np.ndarray | None:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    x1 = max(0, int(round(x1 - box_width * padding_ratio)))
    y1 = max(0, int(round(y1 - box_height * padding_ratio)))
    x2 = min(frame_width, int(round(x2 + box_width * padding_ratio)))
    y2 = min(frame_height, int(round(y2 + box_height * padding_ratio)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def id_color(global_id: int) -> tuple[int, int, int]:
    return (
        64 + (global_id * 53) % 160,
        64 + (global_id * 97) % 160,
        64 + (global_id * 149) % 160,
    )


def render_crop_frame(
    crop: np.ndarray,
    width: int,
    height: int,
    header_height: int,
    global_id: int,
    camera_id: str,
    timestamp: float,
    record: dict[str, Any],
    policy_note: str | None,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    body_height = max(1, height - header_height)
    crop_height, crop_width = crop.shape[:2]
    scale = min(width / max(1, crop_width), body_height / max(1, crop_height))
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (resized_width, resized_height), interpolation=interpolation)
    offset_x = (width - resized_width) // 2
    offset_y = header_height + (body_height - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized

    color = id_color(global_id)
    cv2.rectangle(canvas, (0, 0), (width - 1, header_height - 1), color, -1)
    cv2.putText(
        canvas,
        f"ID {global_id}  |  {camera_id}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    similarity = reid_similarity(record.get("match") or {})
    similarity_text = "-" if similarity is None else f"{similarity:.2f}"
    confidence = float(record.get("confidence", 0.0))
    cv2.putText(
        canvas,
        f"t={timestamp:.1f}s  Det={confidence:.2f}  ReID={similarity_text}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    reason = policy_note or short_reason((record.get("match") or {}).get("reason"))
    cv2.putText(
        canvas,
        reason,
        (12, 79),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return canvas


def reid_similarity(match: dict[str, Any]) -> float | None:
    reason = str(match.get("reason") or "")
    if reason.startswith("appearance"):
        distance = match.get("best_distance")
    else:
        distance = match.get("assigned_distance")
        if distance is None:
            distance = match.get("best_distance")
    if distance is None:
        return None
    return max(0.0, min(1.0, 1.0 - float(distance)))


def short_reason(reason: Any) -> str:
    value = str(reason or "tracked")
    replacements = {
        "existing_local_track_verified": "local track verified",
        "existing_local_track_drift_hold": "local track held",
        "existing_local_track_hold_no_gallery_update": "held; gallery unchanged",
        "low_quality_existing_track_hold": "low-quality track held",
        "appearance_match_verified": "cross/local appearance match",
        "new_identity": "new identity",
    }
    return replacements.get(value, value.replace("_", " "))


def render_transition_card(
    width: int,
    height: int,
    global_id: int,
    from_camera: str,
    to_camera: str,
    gap_seconds: float,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    color = id_color(global_id)
    cv2.putText(
        canvas,
        f"ID {global_id}",
        (width // 2 - 65, height // 2 - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{from_camera}  ->  {to_camera}",
        (28, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"source gap {gap_seconds:.1f} s",
        (55, height // 2 + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return canvas


def create_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open identity video writer: {path}")
    return writer


def export_identity_videos(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir or args.events.parent / "identity_videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_ids = {int(value) for value in args.global_ids.split(",") if value.strip()}
    recordings = discover_recordings(args.recordings_root, args.session, args.file_name)
    if not recordings:
        raise SystemExit("No source recordings found.")

    captures: dict[str, CaptureState] = {}
    for camera_id, path in recordings.items():
        capture = cv2.VideoCapture(str(path))
        if capture.isOpened():
            captures[camera_id] = CaptureState(capture)

    states: dict[int, IdentityVideoState] = {}
    historical_overrides: dict[tuple[str, int], int] = {}
    processed_frames = 0
    skipped_frames = 0
    last_progress = 0
    try:
        with args.events.open() as event_file:
            parsed_events = (
                json.loads(line)
                for line in event_file
                if line.strip()
            )
            for timestamp, grouped_events in itertools.groupby(
                parsed_events,
                key=lambda event: float(event.get("timestamp", 0.0)),
            ):
                observations_by_id: dict[int, list[IdentityObservation]] = {}
                for event in grouped_events:
                    records = [
                        record
                        for record in event.get("tracks", [])
                        if record.get("global_id") is not None
                    ]
                    if not records:
                        continue
                    camera_id = str(event["camera_id"])
                    capture_state = captures.get(camera_id)
                    if capture_state is None:
                        skipped_frames += 1
                        continue
                    frame = read_frame(capture_state, int(event["frame_id"]))
                    if frame is None:
                        skipped_frames += 1
                        continue
                    processed_frames += 1
                    frame_height, frame_width = frame.shape[:2]

                    for record in records:
                        global_id, policy_note = resolve_historical_global_id(
                            camera_id,
                            record,
                            frame_width,
                            frame_height,
                            historical_overrides,
                            args.apply_edge_remap_guard,
                            args.edge_margin_ratio,
                        )
                        if global_id is None or (allowed_ids and global_id not in allowed_ids):
                            continue
                        edge, quality = observation_quality(
                            record,
                            frame_width,
                            frame_height,
                            args.edge_margin_ratio,
                        )
                        observations_by_id.setdefault(global_id, []).append(
                            IdentityObservation(
                                camera_id=camera_id,
                                timestamp=timestamp,
                                frame=frame,
                                record=record,
                                policy_note=policy_note,
                                touches_edge=edge,
                                quality=quality,
                            )
                        )

                for global_id, observations in sorted(observations_by_id.items()):
                    existing_state = states.get(global_id)
                    observation = select_identity_observation(
                        observations,
                        existing_state.last_camera if existing_state is not None else None,
                    )
                    camera_id = observation.camera_id
                    record = observation.record
                    policy_note = observation.policy_note
                    crop = padded_crop(
                        observation.frame,
                        record["bbox"],
                        args.padding_ratio,
                    )
                    if crop is None:
                        continue

                    state = existing_state
                    if state is None:
                        path = output_dir / f"id_{global_id:04d}.mp4"
                        state = IdentityVideoState(
                            global_id=global_id,
                            path=path,
                            writer=create_writer(path, args.fps, args.width, args.height),
                        )
                        states[global_id] = state

                    if state.last_camera is not None and camera_id != state.last_camera:
                        gap_seconds = max(0.0, timestamp - float(state.last_time or timestamp))
                        card = render_transition_card(
                            args.width,
                            args.height,
                            global_id,
                            state.last_camera,
                            camera_id,
                            gap_seconds,
                        )
                        card_frames = max(1, int(round(args.transition_seconds * args.fps)))
                        for _ in range(card_frames):
                            state.writer.write(card)
                        state.transition_frames += card_frames
                        state.transitions.append(
                            {
                                "from_camera": state.last_camera,
                                "to_camera": camera_id,
                                "gap_seconds": gap_seconds,
                                "at_time": timestamp,
                            }
                        )

                    rendered = render_crop_frame(
                        crop,
                        args.width,
                        args.height,
                        args.header_height,
                        global_id,
                        camera_id,
                        timestamp,
                        record,
                        policy_note,
                    )
                    state.writer.write(rendered)
                    state.observations += 1
                    state.first_time = timestamp if state.first_time is None else state.first_time
                    state.last_time = timestamp
                    state.last_camera = camera_id
                    if camera_id not in state.cameras:
                        state.cameras.append(camera_id)

                if processed_frames - last_progress >= 500:
                    last_progress = processed_frames
                    print(
                        f"source_frames={processed_frames} identity_videos={len(states)} "
                        f"crop_frames={sum(state.observations for state in states.values())}"
                    )
    finally:
        for state in states.values():
            state.writer.release()
        for capture_state in captures.values():
            capture_state.capture.release()

    removed = []
    for global_id, state in list(states.items()):
        if state.observations < args.min_observations:
            state.path.unlink(missing_ok=True)
            removed.append(global_id)
            del states[global_id]

    identities = []
    for global_id, state in sorted(states.items()):
        identities.append(
            {
                "global_id": global_id,
                "path": str(state.path),
                "observations": state.observations,
                "transition_frames": state.transition_frames,
                "video_frames": state.observations + state.transition_frames,
                "video_duration_seconds": (state.observations + state.transition_frames) / args.fps,
                "source_start_time": state.first_time,
                "source_end_time": state.last_time,
                "cameras": state.cameras,
                "transitions": state.transitions,
            }
        )

    manifest = {
        "source_events": str(args.events),
        "source_recordings": {camera: str(path) for camera, path in sorted(recordings.items())},
        "parameters": {
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            "padding_ratio": args.padding_ratio,
            "transition_seconds": args.transition_seconds,
            "min_observations": args.min_observations,
            "apply_edge_remap_guard": args.apply_edge_remap_guard,
        },
        "processed_source_frames": processed_frames,
        "skipped_source_frames": skipped_frames,
        "removed_short_identity_ids": removed,
        "identity_count": len(identities),
        "identities": identities,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    print(f"Wrote {len(identities)} identity videos to {output_dir}")
    print(f"Wrote: {manifest_path}")
    return manifest


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.header_height < 0:
        raise SystemExit("Video dimensions must be positive.")
    if args.header_height >= args.height:
        raise SystemExit("Header height must be smaller than video height.")
    if args.fps <= 0:
        raise SystemExit("FPS must be positive.")
    export_identity_videos(args)


if __name__ == "__main__":
    main()
