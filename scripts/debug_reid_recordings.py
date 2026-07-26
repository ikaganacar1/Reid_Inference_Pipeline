#!/usr/bin/env python3
"""Offline replay/debug tool for factory CCTV ReID recordings."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import yaml

from src.detector import YOLOPersonDetector
from src.realtime.identity_assignment import GlobalIdentityAssigner
from src.realtime.protocol import write_jpeg
from src.reid_client import create_reid_client
from src.tracker import ReIDTracker
from src.yoloe_detector import YOLOEPersonDetector


DEFAULT_RECORDINGS_ROOT = Path("recordings")
DEFAULT_EXCLUDE_CHANNELS = {"2601", "3001", "401", "2201", "2101", "1201", "1701"}
MAX_EMBEDDING_DIAGNOSTIC_SAMPLES = 1000
LOCAL_GENERALIZED_ONNX = Path(
    "TwinProject_models/reid_generalized_yolo11n/"
    "generalized_reid_swin_epoch119.onnx"
)


@dataclass
class TrackState:
    track_id: int
    bbox: np.ndarray
    lost_frames: int = 0


class SimpleIoUTracker:
    """Small deterministic tracker for offline identity debugging.

    This is not a replacement for BoTSORT. It gives stable local IDs for replay
    when BoxMOT is not installed on the analysis machine.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 15):
        self.iou_threshold = float(iou_threshold)
        self.max_lost_frames = int(max_lost_frames)
        self.next_track_id = 1
        self.tracks: list[TrackState] = []

    def update(self, detections: np.ndarray) -> np.ndarray:
        if len(detections) == 0:
            for track in self.tracks:
                track.lost_frames += 1
            self.tracks = [track for track in self.tracks if track.lost_frames <= self.max_lost_frames]
            return np.empty((0, 8), dtype=np.float32)

        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))
        matches: list[tuple[int, int]] = []

        candidates = []
        for track_idx, track in enumerate(self.tracks):
            for det_idx, detection in enumerate(detections):
                candidates.append((bbox_iou(track.bbox, detection[:4]), track_idx, det_idx))
        for iou, track_idx, det_idx in sorted(candidates, reverse=True):
            if iou < self.iou_threshold:
                break
            if track_idx not in unmatched_tracks or det_idx not in unmatched_detections:
                continue
            unmatched_tracks.remove(track_idx)
            unmatched_detections.remove(det_idx)
            matches.append((track_idx, det_idx))

        rows = []
        for track_idx, det_idx in matches:
            detection = detections[det_idx]
            track = self.tracks[track_idx]
            track.bbox = detection[:4].astype(np.float32)
            track.lost_frames = 0
            rows.append(make_track_row(detection, track.track_id, det_idx))

        for det_idx in sorted(unmatched_detections):
            detection = detections[det_idx]
            track = TrackState(self.next_track_id, detection[:4].astype(np.float32), lost_frames=0)
            self.next_track_id += 1
            self.tracks.append(track)
            rows.append(make_track_row(detection, track.track_id, det_idx))

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].lost_frames += 1
        self.tracks = [track for track in self.tracks if track.lost_frames <= self.max_lost_frames]

        if not rows:
            return np.empty((0, 8), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)


def make_track_row(detection: np.ndarray, track_id: int, det_idx: int) -> list[float]:
    return [
        float(detection[0]),
        float(detection[1]),
        float(detection[2]),
        float(detection[3]),
        float(track_id),
        float(detection[4]),
        float(detection[5]),
        float(det_idx),
    ]


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-root", type=Path, default=DEFAULT_RECORDINGS_ROOT)
    parser.add_argument("--session", default="session")
    parser.add_argument("--file-name", default="recording.mkv")
    parser.add_argument("--exclude", default=",".join(sorted(DEFAULT_EXCLUDE_CHANNELS)))
    parser.add_argument("--channels", default="", help="Optional comma-separated channel allow-list.")
    parser.add_argument("--limit-cameras", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument(
        "--replay-fps",
        type=float,
        default=0.0,
        help=(
            "Synchronized wall-clock ticks per second. Default 0 uses the "
            "slowest source FPS. Sources slower than this rate are never inferred twice."
        ),
    )
    parser.add_argument("--sync-replay", action="store_true", default=True)
    parser.add_argument("--sequential", dest="sync_replay", action="store_false")
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--detector", choices=["yolo", "yoloe"], default="yolo")
    parser.add_argument("--yolo-model", type=Path, default=Path("yolo26m.pt"))
    parser.add_argument("--yolo-device", default="cuda:0")
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.5)
    parser.add_argument("--yolo-prompts", default="person", help="Comma-separated YOLOE text prompts.")

    parser.add_argument("--reid-model", type=Path, default=None)
    parser.add_argument("--reid-input-height", type=int, default=256)
    parser.add_argument("--reid-input-width", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--reid-batch-size", type=int, default=8)
    parser.add_argument("--reid-provider", default="CUDAExecutionProvider")

    parser.add_argument("--tracker-iou", type=float, default=0.3)
    parser.add_argument("--tracker-buffer", type=int, default=15)
    parser.add_argument(
        "--botsort-track-buffer",
        type=int,
        default=0,
        help="Override BoTSORT lost-track frames when replay FPS is sampled.",
    )
    parser.add_argument(
        "--tracker",
        choices=["botsort", "iou"],
        default="botsort",
        help="Use the production BoTSORT path or the lightweight diagnostic IoU tracker.",
    )
    parser.add_argument("--save-annotated-every", type=int, default=5)
    parser.add_argument("--save-annotated-video", action="store_true", default=True)
    parser.add_argument("--no-save-annotated-video", dest="save_annotated_video", action="store_false")
    parser.add_argument(
        "--annotated-video-fps",
        type=float,
        default=0.0,
        help="Output video FPS. Default 0 preserves sampled-time speed: source_fps / stride.",
    )
    parser.add_argument(
        "--annotated-video-width",
        type=int,
        default=1280,
        help="Resize annotated videos to this width. Use 0 for original resolution.",
    )
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument(
        "--allow-all-camera-overlap",
        action="store_true",
        help="Allow one global ID in any cameras at the same timestamp.",
    )
    parser.add_argument(
        "--overlapping-camera-pairs",
        default="",
        help="Comma-separated camera pairs, for example ch201:ch301,ch301:ch501.",
    )
    parser.add_argument(
        "--adjacent-camera-pairs",
        default="",
        help="Comma-separated fast-handoff pairs, for example ch501:ch601.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path("experiments") / f"offline_reid_debug_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    recordings = discover_recordings(args)
    if not recordings:
        raise SystemExit("No recordings matched the requested session/file/exclude filters.")

    print(f"recordings: {len(recordings)}")
    for channel, path in recordings:
        print(f"  ch{channel}: {path}")
    print(f"output: {output_dir}")

    detector = create_detector(args)
    reid_config = load_reid_config(args)
    realtime_config = load_realtime_config()
    prime_config = dict(realtime_config.get("prime", {}))
    prime_config["output_dir"] = str(output_dir)
    prime_config["debug_reid"] = True
    prime_config["save_debug_crops"] = bool(args.save_crops)
    prime_config["debug_crop_interval_frames"] = 1 if args.save_crops else int(
        prime_config.get("debug_crop_interval_frames", 10)
    )
    if args.allow_all_camera_overlap:
        prime_config["allow_all_camera_overlap"] = True
    if args.overlapping_camera_pairs:
        prime_config["overlapping_camera_pairs"] = args.overlapping_camera_pairs
    if args.adjacent_camera_pairs:
        prime_config["adjacent_camera_pairs"] = args.adjacent_camera_pairs

    reid_client = create_reid_client(reid_config)
    assigner = GlobalIdentityAssigner(prime_config, output_dir)
    print(
        "camera overlap policy: "
        f"all={assigner.camera_topology.allow_all_overlaps} "
        f"pairs={assigner.camera_topology.as_pairs()} "
        f"adjacent={assigner.camera_topology.as_adjacent_pairs()}"
    )
    trackers = {channel: create_tracker(args) for channel, _ in recordings}

    counters: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    global_counts: Counter[int] = Counter()
    by_camera: dict[str, Counter[str]] = defaultdict(Counter)
    embedding_rows: list[np.ndarray] = []
    embedding_meta: list[dict[str, Any]] = []

    event_path = output_dir / "offline_tracks.jsonl"
    annotated_dir = output_dir / "annotated"
    video_dir = output_dir / "annotated_videos"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    if args.save_annotated_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    with event_path.open("w") as event_file:
        if args.sync_replay:
            process_recordings_synchronized(
                recordings,
                args,
                detector,
                reid_client,
                assigner,
                trackers,
                event_file,
                annotated_dir,
                video_dir,
                counters,
                reason_counts,
                global_counts,
                by_camera,
                embedding_rows,
                embedding_meta,
            )
        else:
            for channel, video_path in recordings:
                process_recording(
                    channel,
                    video_path,
                    args,
                    detector,
                    reid_client,
                    assigner,
                    trackers[channel],
                    event_file,
                    annotated_dir,
                    video_dir,
                    counters,
                    reason_counts,
                    global_counts,
                    by_camera,
                    embedding_rows,
                    embedding_meta,
                )

    summary = build_summary(counters, reason_counts, global_counts, by_camera, embedding_rows, embedding_meta)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nsummary")
    print(json.dumps(summary, indent=2)[:6000])
    print(f"\nWrote: {output_dir / 'summary.json'}")
    print(f"Wrote: {event_path}")


def discover_recordings(args: argparse.Namespace) -> list[tuple[str, Path]]:
    exclude = {item.strip() for item in args.exclude.split(",") if item.strip()}
    allow = {item.strip() for item in args.channels.split(",") if item.strip()}
    session_dir = args.recordings_root / args.session
    nested_paths = list(session_dir.glob(f"cam*_ch*/{args.file_name}"))
    flat_paths = list(session_dir.glob("channel_*.mkv"))
    paths = sorted(set(nested_paths + flat_paths))
    recordings = []
    for path in paths:
        channel = extract_channel(path)
        if not channel:
            continue
        if channel in exclude:
            continue
        if allow and channel not in allow:
            continue
        recordings.append((channel, path))
    if args.limit_cameras > 0:
        recordings = recordings[: args.limit_cameras]
    return recordings


def extract_channel(path: Path) -> str | None:
    match = re.search(r"_ch(\d+)$", path.parent.name)
    if match:
        return match.group(1)
    match = re.match(r"channel_(\d+)(?:_|$)", path.stem)
    return match.group(1) if match else None


def create_detector(args: argparse.Namespace) -> Any:
    if args.detector == "yoloe":
        return YOLOEPersonDetector(load_yoloe_config(args))
    return YOLOPersonDetector(load_yolo_config(args))


def load_yolo_config(args: argparse.Namespace) -> dict[str, Any]:
    with Path("configs/yolo_config.yaml").open() as f:
        config = yaml.safe_load(f)
    config["model"]["path"] = str(args.yolo_model)
    config["model"]["device"] = args.yolo_device
    config["detection"]["imgsz"] = int(args.yolo_imgsz)
    config["detection"]["conf_threshold"] = float(args.yolo_conf)
    device = str(args.yolo_device).lower()
    config.setdefault("inference", {})["half"] = device.startswith("cuda") or device.isdigit()
    return config


def load_yoloe_config(args: argparse.Namespace) -> dict[str, Any]:
    with Path("configs/yoloe_config.yaml").open() as f:
        config = yaml.safe_load(f)
    config["model"]["path"] = str(args.yolo_model)
    config["model"]["device"] = args.yolo_device
    config["model"]["text_prompts"] = [
        item.strip() for item in args.yolo_prompts.split(",") if item.strip()
    ] or ["person"]
    config["detection"]["imgsz"] = int(args.yolo_imgsz)
    config["detection"]["conf_threshold"] = float(args.yolo_conf)
    config.setdefault("inference", {})["half"] = False
    return config


def load_reid_config(args: argparse.Namespace) -> dict[str, Any]:
    with Path("configs/reid_config.yaml").open() as f:
        config = yaml.safe_load(f)
    model_path = args.reid_model
    if model_path is None:
        configured = Path(config["model"]["onnx_path"])
        model_path = configured if configured.exists() else LOCAL_GENERALIZED_ONNX
    config["backend"] = "onnxruntime_direct"
    config["model"]["onnx_path"] = str(model_path)
    config["model"]["input_shape"] = [int(args.reid_input_height), int(args.reid_input_width)]
    config["model"]["embedding_dim"] = int(args.embedding_dim)
    config.setdefault("onnxruntime", {})["providers"] = [args.reid_provider]
    config.setdefault("inference", {})["max_batch_size"] = int(args.reid_batch_size)
    return config


def load_realtime_config() -> dict[str, Any]:
    with Path("configs/realtime_config.yaml").open() as f:
        return yaml.safe_load(f)


def create_tracker(args: argparse.Namespace) -> Any:
    if args.tracker == "iou":
        return SimpleIoUTracker(args.tracker_iou, args.tracker_buffer)
    with Path("configs/tracker_config.yaml").open() as f:
        config = yaml.safe_load(f)
    if args.botsort_track_buffer > 0:
        config["botsort"]["track_buffer"] = int(args.botsort_track_buffer)
    return ReIDTracker(config)


def update_tracker(
    tracker: Any,
    detections: np.ndarray,
    frame: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    if isinstance(tracker, SimpleIoUTracker):
        return tracker.update(detections)
    return tracker.update(detections, frame, embeddings)


def synchronized_replay_timing(
    fps_by_camera: dict[str, float],
    duration_by_camera: dict[str, float],
    start_frame: int,
    max_frames: int,
    stride: int,
    replay_fps: float = 0.0,
) -> tuple[float, float, float, float]:
    """Build one wall-clock timeline for recordings with different frame rates."""
    timeline_fps = replay_fps if replay_fps > 0 else min(fps_by_camera.values())
    start_time = float(start_frame) / timeline_fps
    stop_time = min(duration_by_camera.values())
    if max_frames > 0:
        stop_time = min(stop_time, start_time + float(max_frames) / timeline_fps)
    step_seconds = max(1, stride) / timeline_fps
    return timeline_fps, start_time, stop_time, step_seconds


def source_frame_at(timestamp: float, source_fps: float) -> int:
    return int(timestamp * source_fps + 0.5)


def process_recordings_synchronized(
    recordings: list[tuple[str, Path]],
    args: argparse.Namespace,
    detector: Any,
    reid_client: Any,
    assigner: GlobalIdentityAssigner,
    trackers: dict[str, Any],
    event_file: Any,
    annotated_dir: Path,
    video_dir: Path,
    counters: Counter[str],
    reason_counts: Counter[str],
    global_counts: Counter[int],
    by_camera: dict[str, Counter[str]],
    embedding_rows: list[np.ndarray],
    embedding_meta: list[dict[str, Any]],
) -> None:
    """Replay all camera files by frame time, like the realtime system sees them."""
    captures: dict[str, cv2.VideoCapture] = {}
    fps_by_camera: dict[str, float] = {}
    duration_by_camera: dict[str, float] = {}
    writers: dict[str, cv2.VideoWriter] = {}
    writer_paths: dict[str, Path] = {}
    last_source_frame_by_camera: dict[str, int] = {}

    try:
        for channel, video_path in recordings:
            camera_id = f"ch{channel}"
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"WARNING: failed to open {video_path}")
                continue
            captures[camera_id] = cap
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_by_camera[camera_id] = fps
            duration_by_camera[camera_id] = total_frames / fps

        if not captures:
            return

        timeline_fps, start_time, stop_time, step_seconds = synchronized_replay_timing(
            fps_by_camera,
            duration_by_camera,
            args.start_frame,
            args.max_frames,
            args.stride,
            args.replay_fps,
        )
        replay_fps = 1.0 / step_seconds
        camera_items = [
            (f"ch{channel}", channel, video_path)
            for channel, video_path in recordings
            if f"ch{channel}" in captures
        ]

        time_slice_index = 0
        event_timestamp = start_time
        while event_timestamp < stop_time:
            counters["time_slices"] += 1
            assigned_ids_by_camera: dict[str, set[int]] = {}
            observations = []
            all_crops: list[np.ndarray] = []
            for camera_id, channel, _ in camera_items:
                cap = captures[camera_id]
                fps = fps_by_camera[camera_id]
                source_frame_id = source_frame_at(event_timestamp, fps)
                last_source_frame_id = last_source_frame_by_camera.get(camera_id)
                if source_frame_id == last_source_frame_id:
                    counters["duplicate_source_frames_skipped"] += 1
                    continue
                if (
                    last_source_frame_id is None
                    or source_frame_id <= last_source_frame_id
                    or source_frame_id - last_source_frame_id > 20
                ):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_id)
                    ok, frame = cap.read()
                else:
                    ok = True
                    for _ in range(source_frame_id - last_source_frame_id - 1):
                        ok = cap.grab()
                        if not ok:
                            break
                    ok, frame = cap.read() if ok else (False, None)
                if not ok or frame is None:
                    continue
                last_source_frame_by_camera[camera_id] = source_frame_id

                detect_result = detector.detect(frame)
                detections, crops = detect_result[0], detect_result[1]
                start = len(all_crops)
                all_crops.extend(crops)
                observations.append(
                    (
                        camera_id,
                        channel,
                        source_frame_id,
                        frame,
                        detections,
                        crops,
                        start,
                        len(all_crops),
                    )
                )

            all_embeddings = reid_client.infer(all_crops, max_batch_size=args.reid_batch_size)
            for (
                camera_id,
                channel,
                source_frame_id,
                frame,
                detections,
                crops,
                start,
                end,
            ) in observations:
                embeddings = all_embeddings[start:end]
                tracks = update_tracker(trackers[channel], detections, frame, embeddings)
                blocked_global_ids = {
                    global_id
                    for other_camera, assigned_ids in assigned_ids_by_camera.items()
                    if not assigner.cameras_may_overlap(camera_id, other_camera)
                    for global_id in assigned_ids
                }
                records = assigner.assign_tracks(
                    camera_id,
                    source_frame_id,
                    frame.shape[1],
                    frame.shape[0],
                    tracks,
                    embeddings,
                    crops,
                    timestamp=event_timestamp,
                    blocked_global_ids=blocked_global_ids,
                )

                update_debug_counters(
                    camera_id,
                    channel,
                    source_frame_id,
                    detections,
                    crops,
                    embeddings,
                    records,
                    assigner,
                    args,
                    counters,
                    reason_counts,
                    global_counts,
                    by_camera,
                    embedding_rows,
                    embedding_meta,
                )

                assigned_ids_by_camera[camera_id] = {
                    int(record["global_id"])
                    for record in records
                    if record.get("global_id") is not None
                }

                event_file.write(
                    json.dumps(
                        {
                            "camera_id": camera_id,
                            "channel": channel,
                            "frame_id": source_frame_id,
                            "timestamp": event_timestamp,
                            "detections": len(detections),
                            "tracks": records,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

                annotated = frame.copy()
                draw_records(annotated, records)

                processed = by_camera[camera_id]["frames"] - 1
                if args.save_annotated_every > 0 and processed % args.save_annotated_every == 0:
                    out_path = annotated_dir / f"{camera_id}_frame_{source_frame_id:08d}.jpg"
                    write_jpeg(out_path, resize_for_output(annotated, 1280))

                if args.save_annotated_video:
                    video_frame = resize_for_output(annotated, args.annotated_video_width)
                    if camera_id not in writers:
                        video_fps = args.annotated_video_fps
                        if video_fps <= 0:
                            video_fps = min(fps_by_camera[camera_id], replay_fps)
                        video_path_out = video_dir / f"{camera_id}_annotated.mp4"
                        writers[camera_id] = cv2.VideoWriter(
                            str(video_path_out),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            float(video_fps),
                            (video_frame.shape[1], video_frame.shape[0]),
                        )
                        if not writers[camera_id].isOpened():
                            raise RuntimeError(f"Failed to open annotated video writer: {video_path_out}")
                        writer_paths[camera_id] = video_path_out
                        print(f"Writing annotated video: {video_path_out} fps={video_fps:.2f}")
                    writers[camera_id].write(video_frame)

            if counters["time_slices"] % 10 == 0:
                print(
                    f"sync time={event_timestamp:.2f}s time_slices={counters['time_slices']} "
                    f"frames={counters['frames']} gallery={len(assigner.gallery.entries)}"
                )
            time_slice_index += 1
            event_timestamp = start_time + time_slice_index * step_seconds
    finally:
        for cap in captures.values():
            cap.release()
        for camera_id, writer in writers.items():
            writer.release()
            print(f"Wrote annotated video: {writer_paths.get(camera_id)}")


def update_debug_counters(
    camera_id: str,
    channel: str,
    frame_id: int,
    detections: np.ndarray,
    crops: list[np.ndarray],
    embeddings: np.ndarray,
    records: list[dict[str, Any]],
    assigner: GlobalIdentityAssigner,
    args: argparse.Namespace,
    counters: Counter[str],
    reason_counts: Counter[str],
    global_counts: Counter[int],
    by_camera: dict[str, Counter[str]],
    embedding_rows: list[np.ndarray],
    embedding_meta: list[dict[str, Any]],
) -> None:
    counters["frames"] += 1
    counters["detections"] += len(detections)
    counters["tracks"] += len(records)
    by_camera[camera_id]["frames"] += 1
    by_camera[camera_id]["detections"] += len(detections)
    by_camera[camera_id]["tracks"] += len(records)

    for record in records:
        reason = ((record.get("match") or {}).get("reason")) or "no_embedding"
        reason_counts[reason] += 1
        by_camera[camera_id][reason] += 1
        if record.get("global_id") is not None:
            global_counts[int(record["global_id"])] += 1

    records_by_detection = {
        int(record["detection_index"]): record
        for record in records
        if int(record.get("detection_index", -1)) >= 0
    }
    for det_idx, embedding in enumerate(embeddings):
        record = records_by_detection.get(det_idx)
        embedding_rows.append(np.asarray(embedding, dtype=np.float32))
        crop_path = None
        if args.save_crops and det_idx < len(crops):
            crop_dir = Path(assigner.debug_dir) / camera_id
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_dir / f"frame_{frame_id:08d}_det_{det_idx:02d}.jpg"
            write_jpeg(crop_path, crops[det_idx])
        embedding_meta.append(
            {
                "camera_id": camera_id,
                "channel": channel,
                "frame_id": frame_id,
                "det_idx": det_idx,
                "local_track_id": record.get("local_track_id") if record else None,
                "global_id": record.get("global_id") if record else None,
                "match_reason": ((record.get("match") or {}).get("reason")) if record else None,
                "confidence": float(detections[det_idx][4]) if det_idx < len(detections) else None,
                "bbox": detections[det_idx][:4].astype(float).tolist() if det_idx < len(detections) else None,
                "crop_path": str(crop_path) if crop_path is not None else None,
            }
        )


def process_recording(
    channel: str,
    video_path: Path,
    args: argparse.Namespace,
    detector: YOLOPersonDetector,
    reid_client: Any,
    assigner: GlobalIdentityAssigner,
    tracker: Any,
    event_file: Any,
    annotated_dir: Path,
    video_dir: Path,
    counters: Counter[str],
    reason_counts: Counter[str],
    global_counts: Counter[int],
    by_camera: dict[str, Counter[str]],
    embedding_rows: list[np.ndarray],
    embedding_meta: list[dict[str, Any]],
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"WARNING: failed to open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    stop_frame = min(total_frames, args.start_frame + args.max_frames) if args.max_frames > 0 else total_frames
    processed = 0
    camera_id = f"ch{channel}"
    writer = None
    video_path_out = None

    try:
        for frame_id in range(args.start_frame, stop_frame, max(1, args.stride)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                break

            detect_result = detector.detect(frame)
            detections, crops = detect_result[0], detect_result[1]
            embeddings = reid_client.infer(crops, max_batch_size=args.reid_batch_size)
            tracks = update_tracker(tracker, detections, frame, embeddings)
            records = assigner.assign_tracks(
                camera_id,
                frame_id,
                frame.shape[1],
                frame.shape[0],
                tracks,
                embeddings,
                crops,
                timestamp=float(frame_id) / max(fps, 1.0),
            )

            counters["frames"] += 1
            counters["detections"] += len(detections)
            counters["tracks"] += len(records)
            by_camera[camera_id]["frames"] += 1
            by_camera[camera_id]["detections"] += len(detections)
            by_camera[camera_id]["tracks"] += len(records)

            for record in records:
                reason = ((record.get("match") or {}).get("reason")) or "no_embedding"
                reason_counts[reason] += 1
                by_camera[camera_id][reason] += 1
                if record.get("global_id") is not None:
                    global_counts[int(record["global_id"])] += 1

            for det_idx, embedding in enumerate(embeddings):
                embedding_rows.append(np.asarray(embedding, dtype=np.float32))
                crop_path = None
                if args.save_crops and det_idx < len(crops):
                    crop_dir = Path(assigner.debug_dir) / camera_id
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = crop_dir / f"frame_{frame_id:08d}_det_{det_idx:02d}.jpg"
                    write_jpeg(crop_path, crops[det_idx])
                embedding_meta.append(
                    {
                        "camera_id": camera_id,
                        "channel": channel,
                        "frame_id": frame_id,
                        "det_idx": det_idx,
                        "confidence": float(detections[det_idx][4]) if det_idx < len(detections) else None,
                        "bbox": detections[det_idx][:4].astype(float).tolist() if det_idx < len(detections) else None,
                        "crop_path": str(crop_path) if crop_path is not None else None,
                    }
                )

            event_file.write(
                json.dumps(
                    {
                        "camera_id": camera_id,
                        "channel": channel,
                        "frame_id": frame_id,
                        "detections": len(detections),
                        "tracks": records,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

            annotated = frame.copy()
            draw_records(annotated, records)

            if args.save_annotated_every > 0 and processed % args.save_annotated_every == 0:
                out_path = annotated_dir / f"{camera_id}_frame_{frame_id:08d}.jpg"
                write_jpeg(out_path, resize_for_output(annotated, 1280))

            if args.save_annotated_video:
                video_frame = resize_for_output(annotated, args.annotated_video_width)
                if writer is None:
                    video_fps = args.annotated_video_fps
                    if video_fps <= 0:
                        video_fps = max(0.1, fps / max(1, args.stride))
                    video_path_out = video_dir / f"{camera_id}_annotated.mp4"
                    writer = cv2.VideoWriter(
                        str(video_path_out),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(video_fps),
                        (video_frame.shape[1], video_frame.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open annotated video writer: {video_path_out}")
                    print(f"Writing annotated video: {video_path_out} fps={video_fps:.2f}")
                writer.write(video_frame)

            processed += 1
            if processed % 10 == 0:
                print(
                    f"{camera_id} processed={processed} frame={frame_id} "
                    f"det={len(detections)} tracks={len(records)} gallery={len(assigner.gallery.entries)}"
                )
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Wrote annotated video: {video_path_out}")


def draw_records(frame: np.ndarray, records: list[dict[str, Any]]) -> None:
    for record in records:
        x1, y1, x2, y2 = [int(value) for value in record["bbox"]]
        gid = record.get("global_id")
        reid_conf = reid_confidence_from_match(record.get("match") or {})
        reid_label = "-" if reid_conf is None else f"{reid_conf:.2f}"
        reason = short_reason((record.get("match") or {}).get("reason"))
        label = (
            f"ID:{gid if gid is not None else '-'} "
            f"Det:{float(record['confidence']):.2f} ReID:{reid_label}"
        )
        detail = f"{reason} L:{record['local_track_id']}"
        color = color_for_id(gid or record["local_track_id"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, [label, detail], x1, y1, color)


def reid_confidence_from_match(match: dict[str, Any]) -> float | None:
    """Convert the distance used by identity assignment to a readable similarity."""
    distance = match.get("assigned_distance")
    if distance is None:
        distance = match.get("best_distance")
    if distance is None:
        distance = match.get("candidate_distance")
    if distance is None:
        return None
    return max(0.0, min(1.0, 1.0 - float(distance)))


def short_reason(reason: str | None) -> str:
    mapping = {
        "appearance_match_verified": "appearance",
        "appearance_remap": "remap",
        "appearance_remap_frame_collision": "remap_collision",
        "existing_local_track_verified": "verified",
        "existing_local_track_hold_no_gallery_update": "hold_no_update",
        "existing_local_track_drift_hold": "drift_hold",
        "low_quality_existing_track_hold": "lowq_hold",
        "low_quality_no_identity": "lowq_no_id",
        "new_identity": "new",
        "new_identity_frame_collision": "new_collision",
        "pending_new_identity": "pending_new",
        "pending_frame_collision": "pending_collision",
        "pending_identity_drift": "pending_drift",
    }
    return mapping.get(reason or "", reason or "unknown")


def draw_label(frame: np.ndarray, lines: list[str], x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    width = max((size[0] for size in sizes), default=0) + 8
    line_height = max((size[1] for size in sizes), default=12) + 7
    height = line_height * len(lines) + 4
    top = max(0, y - height)
    left = max(0, x)
    cv2.rectangle(frame, (left, top), (left + width, top + height), color, -1)
    for index, line in enumerate(lines):
        text_y = top + 4 + line_height * (index + 1) - 5
        cv2.putText(
            frame,
            line,
            (left + 4, text_y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def resize_for_output(frame: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0 or frame.shape[1] == target_width:
        return frame
    scale = float(target_width) / float(frame.shape[1])
    target_height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def color_for_id(identity: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(identity) * 9973)
    color = rng.integers(60, 235, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def build_summary(
    counters: Counter[str],
    reason_counts: Counter[str],
    global_counts: Counter[int],
    by_camera: dict[str, Counter[str]],
    embedding_rows: list[np.ndarray],
    embedding_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "counts": dict(counters),
        "match_reasons": dict(reason_counts),
        "global_id_count": len(global_counts),
        "top_global_ids": {str(key): value for key, value in global_counts.most_common(20)},
        "by_camera": {camera: dict(values) for camera, values in sorted(by_camera.items())},
    }
    if embedding_rows:
        embeddings = np.asarray(embedding_rows, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1)
        normalized = embeddings / (norms[:, None] + 1e-12)
        sample_count = min(len(normalized), MAX_EMBEDDING_DIAGNOSTIC_SAMPLES)
        sample_indices = np.linspace(0, len(normalized) - 1, sample_count, dtype=int)
        sampled = normalized[sample_indices]
        sampled_meta = [embedding_meta[index] for index in sample_indices]
        distances = 1.0 - sampled @ sampled.T
        tri = distances[np.triu_indices(sample_count, 1)]
        summary["embedding_norm"] = {
            "min": float(norms.min()),
            "mean": float(norms.mean()),
            "max": float(norms.max()),
        }
        summary["embedding_pair_diagnostics"] = {
            "total_embeddings": len(embedding_rows),
            "sampled_embeddings": sample_count,
            "sampling": "all" if sample_count == len(normalized) else "evenly_spaced",
        }
        if len(tri):
            summary["distance_percentiles"] = {
                str(percentile): float(np.percentile(tri, percentile))
                for percentile in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            }
            summary["closest_pairs"] = closest_embedding_pairs(distances, sampled_meta, limit=20)
        track_diagnostics = build_track_prototype_diagnostics(normalized, embedding_meta)
        summary.update(track_diagnostics)
    return summary


def build_track_prototype_diagnostics(
    normalized_embeddings: np.ndarray,
    embedding_meta: list[dict[str, Any]],
    limit: int = 40,
) -> dict[str, Any]:
    """Summarize temporally averaged local tracks for cross-camera diagnosis."""
    grouped_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, meta in enumerate(embedding_meta):
        local_track_id = meta.get("local_track_id")
        if local_track_id is None:
            continue
        grouped_indices[(str(meta["camera_id"]), int(local_track_id))].append(index)

    prototypes = []
    prototype_meta = []
    for (camera_id, local_track_id), indices in sorted(grouped_indices.items()):
        prototype = np.asarray(normalized_embeddings[indices], dtype=np.float32).mean(axis=0)
        norm = float(np.linalg.norm(prototype))
        if norm <= 0:
            continue
        prototypes.append(prototype / norm)
        rows = [embedding_meta[index] for index in indices]
        global_ids = sorted(
            {int(row["global_id"]) for row in rows if row.get("global_id") is not None}
        )
        prototype_meta.append(
            {
                "camera_id": camera_id,
                "local_track_id": local_track_id,
                "global_ids": global_ids,
                "observations": len(indices),
                "first_frame": min(int(row["frame_id"]) for row in rows),
                "last_frame": max(int(row["frame_id"]) for row in rows),
            }
        )

    pairs = []
    for first in range(len(prototypes)):
        for second in range(first + 1, len(prototypes)):
            if prototype_meta[first]["camera_id"] == prototype_meta[second]["camera_id"]:
                continue
            pairs.append(
                {
                    "distance": 1.0 - float(np.dot(prototypes[first], prototypes[second])),
                    "a": prototype_meta[first],
                    "b": prototype_meta[second],
                }
            )
    pairs.sort(key=lambda pair: pair["distance"])
    return {
        "track_prototype_diagnostics": {
            "local_track_count": len(prototypes),
            "cross_camera_pair_count": len(pairs),
        },
        "closest_cross_camera_track_prototypes": pairs[: max(0, int(limit))],
    }


def closest_embedding_pairs(
    distances: np.ndarray,
    embedding_meta: list[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    count = len(embedding_meta)
    pair_count = count * (count - 1) // 2
    if pair_count == 0 or limit <= 0:
        return []

    upper_triangle = np.triu(np.ones((count, count), dtype=bool), k=1)
    candidates = np.where(upper_triangle, distances, np.inf).reshape(-1)
    result_count = min(int(limit), pair_count)
    flat_indices = np.argpartition(candidates, result_count - 1)[:result_count]
    flat_indices = flat_indices[np.argsort(candidates[flat_indices])]

    result = []
    for flat_index in flat_indices:
        i, j = np.unravel_index(int(flat_index), distances.shape)
        result.append(
            {
                "distance": float(distances[i, j]),
                "a": embedding_meta[i],
                "b": embedding_meta[j],
            }
        )
    return result


if __name__ == "__main__":
    main()
