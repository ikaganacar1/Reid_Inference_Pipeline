#!/usr/bin/env python3
"""Summarize realtime ReID debug events."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "events",
        nargs="?",
        default="outputs/realtime/reid_debug/events.jsonl",
        help="Path to events.jsonl produced by the realtime prime server.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=5000,
        help="Analyze only the most recent N events. Use 0 for all events.",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=20,
        help="Print this many recent new-identity events.",
    )
    parser.add_argument(
        "--min-cross-camera-gap-seconds",
        type=float,
        default=0.0,
        help=(
            "Warn when the same global ID changes cameras faster than this many "
            "seconds. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-transition-warnings",
        type=int,
        default=20,
        help="Maximum number of fast cross-camera transitions to print.",
    )
    return parser.parse_args()


def load_events(path: Path, tail: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing debug event file: {path}")

    if tail > 0:
        events: deque[dict[str, Any]] = deque(maxlen=tail)
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return list(events)

    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def format_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_fast_cross_camera_transitions(
    events: list[dict[str, Any]],
    min_gap_seconds: float,
    max_rows: int,
) -> None:
    if min_gap_seconds <= 0:
        return

    by_global: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        global_id = event.get("global_id")
        if global_id is not None:
            by_global[int(global_id)].append(event)

    transitions: list[dict[str, Any]] = []
    for global_id, global_events in by_global.items():
        ordered = sorted(global_events, key=lambda item: float(item.get("time", 0.0)))
        camera_runs: list[dict[str, Any]] = []
        for event in ordered:
            camera_id = event["camera_id"]
            timestamp = float(event.get("time", 0.0))
            frame_id = int(event.get("frame_id", -1))
            if not camera_runs or camera_runs[-1]["camera_id"] != camera_id:
                camera_runs.append(
                    {
                        "global_id": global_id,
                        "camera_id": camera_id,
                        "start_time": timestamp,
                        "end_time": timestamp,
                        "start_frame": frame_id,
                        "end_frame": frame_id,
                    }
                )
            else:
                camera_runs[-1]["end_time"] = timestamp
                camera_runs[-1]["end_frame"] = frame_id

        for previous, current in zip(camera_runs, camera_runs[1:]):
            gap = float(current["start_time"]) - float(previous["end_time"])
            if gap < min_gap_seconds:
                transitions.append(
                    {
                        "global_id": global_id,
                        "from_camera": previous["camera_id"],
                        "to_camera": current["camera_id"],
                        "from_frame": previous["end_frame"],
                        "to_frame": current["start_frame"],
                        "gap": gap,
                    }
                )

    transitions.sort(key=lambda item: (item["gap"], item["global_id"], item["from_frame"]))
    print()
    print(f"fast_cross_camera_transitions(<{min_gap_seconds:.2f}s): {len(transitions)}")
    for item in transitions[:max_rows]:
        print(
            "  "
            f"gid={item['global_id']} "
            f"{item['from_camera']}->{item['to_camera']} "
            f"gap={item['gap']:.2f}s "
            f"frames={item['from_frame']}->{item['to_frame']}"
        )


def main() -> None:
    args = parse_args()
    events = load_events(Path(args.events), args.tail)
    if not events:
        print("No ReID debug events found.")
        return

    by_camera = Counter(event["camera_id"] for event in events)
    by_global = Counter(event["global_id"] for event in events if event.get("global_id") is not None)
    pending_events = sum(1 for event in events if event.get("global_id") is None)
    reasons = Counter((event.get("match") or {}).get("reason", "unknown") for event in events)
    quality_reasons = Counter()
    distances_by_reason: dict[str, list[float]] = defaultdict(list)

    for event in events:
        match = event.get("match") or {}
        distance = match.get("best_distance")
        if distance is not None:
            distances_by_reason[match.get("reason", "unknown")].append(float(distance))
        quality = match.get("quality") or {}
        for reason in quality.get("reasons", []):
            quality_reasons[reason] += 1

    print(f"events: {len(events)}")
    print(f"cameras: {dict(sorted(by_camera.items()))}")
    print(f"global_ids: {len(by_global)} -> {dict(by_global.most_common(10))}")
    print(f"pending_without_id: {pending_events}")
    print(f"reasons: {dict(reasons)}")
    if quality_reasons:
        print(f"quality_reasons: {dict(quality_reasons)}")
    print()

    for reason, values in sorted(distances_by_reason.items()):
        if values:
            print(
                f"{reason}: count={len(values)} "
                f"avg={mean(values):.3f} min={min(values):.3f} max={max(values):.3f}"
            )

    print_fast_cross_camera_transitions(
        events,
        min_gap_seconds=args.min_cross_camera_gap_seconds,
        max_rows=args.max_transition_warnings,
    )

    new_events = [
        event
        for event in events
        if (event.get("match") or {}).get("reason") == "new_identity"
    ][-args.recent :]
    if new_events:
        print("\nrecent new identities:")
        for event in new_events:
            match = event.get("match") or {}
            print(
                "  "
                f"cam={event['camera_id']} frame={event['frame_id']} "
                f"gid={event['global_id']} local={event['local_track_id']} "
                f"best={format_float(match.get('best_distance'))} "
                f"candidate={match.get('candidate_global_id')} "
                f"crop={event.get('crop_path') or '-'}"
            )

    pending = [
        event
        for event in events
        if (event.get("match") or {}).get("reason") == "pending_new_identity"
    ][-args.recent :]
    if pending:
        print("\nrecent pending identities:")
        for event in pending:
            match = event.get("match") or {}
            print(
                "  "
                f"cam={event['camera_id']} frame={event['frame_id']} "
                f"local={event['local_track_id']} "
                f"pending={match.get('pending_frames')}/{match.get('required_frames')} "
                f"best={format_float(match.get('best_distance'))} "
                f"candidate={match.get('candidate_global_id')} "
                f"crop={event.get('crop_path') or '-'}"
            )


if __name__ == "__main__":
    main()
