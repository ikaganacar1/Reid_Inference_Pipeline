#!/usr/bin/env python3
"""Audit global identity journeys in ReID debug events."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.realtime.camera_topology import (  # noqa: E402
    CameraTopology,
    parse_adjacent_camera_pairs,
    parse_overlapping_camera_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, help="Path to reid_debug/events.jsonl.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path. Defaults to <experiment>/cross_camera_analysis.json.",
    )
    parser.add_argument(
        "--visit-gap-seconds",
        type=float,
        default=1.0,
        help="Split a camera visit after this much time without an assigned observation.",
    )
    parser.add_argument(
        "--min-travel-seconds",
        type=float,
        default=2.0,
        help="Flag cross-camera transitions faster than this value.",
    )
    parser.add_argument(
        "--close-rejection-distance",
        type=float,
        default=0.30,
        help="Report new identities whose rejected best candidate is this close or closer.",
    )
    parser.add_argument(
        "--required-match-margin",
        type=float,
        default=0.08,
        help="Margin used to classify close rejections as appearance ambiguity.",
    )
    parser.add_argument("--allow-all-camera-overlap", action="store_true")
    parser.add_argument(
        "--overlapping-camera-pairs",
        default="",
        help="Comma-separated pairs such as ch201:ch301,ch301:ch501.",
    )
    parser.add_argument(
        "--adjacent-camera-pairs",
        default="",
        help="Comma-separated fast-handoff pairs such as ch501:ch601.",
    )
    return parser.parse_args()


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing ReID event file: {path}")
    with path.open() as event_file:
        return [json.loads(line) for line in event_file if line.strip()]


def event_time(event: dict[str, Any]) -> float:
    return float(event.get("time", event.get("timestamp", 0.0)))


def _split_timed_events(
    events: Iterable[dict[str, Any]],
    max_gap_seconds: float,
) -> list[list[dict[str, Any]]]:
    ordered = sorted(events, key=lambda item: (event_time(item), int(item.get("frame_id", -1))))
    if not ordered:
        return []

    chunks = [[ordered[0]]]
    for event in ordered[1:]:
        if event_time(event) - event_time(chunks[-1][-1]) > max_gap_seconds:
            chunks.append([])
        chunks[-1].append(event)
    return chunks


def _visit_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    first = events[0]
    last = events[-1]
    first_match = first.get("match") or {}
    return {
        "global_id": int(first["global_id"]),
        "camera_id": str(first["camera_id"]),
        "local_track_ids": sorted({int(event["local_track_id"]) for event in events}),
        "start_time": event_time(first),
        "end_time": event_time(last),
        "duration_seconds": max(0.0, event_time(last) - event_time(first)),
        "start_frame": int(first.get("frame_id", -1)),
        "end_frame": int(last.get("frame_id", -1)),
        "observations": len(events),
        "entry_reason": first_match.get("reason"),
        "entry_candidate_global_id": first_match.get("candidate_global_id"),
        "entry_best_distance": first_match.get("best_distance"),
        "entry_second_distance": first_match.get("second_distance"),
    }


def build_camera_visits(
    events: list[dict[str, Any]],
    max_gap_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("global_id") is None:
            continue
        grouped[(int(event["global_id"]), str(event["camera_id"]))].append(event)

    visits = []
    for grouped_events in grouped.values():
        for chunk in _split_timed_events(grouped_events, max_gap_seconds):
            visits.append(_visit_from_events(chunk))
    return sorted(
        visits,
        key=lambda visit: (visit["global_id"], visit["start_time"], visit["camera_id"]),
    )


def build_local_visits(
    events: list[dict[str, Any]],
    max_gap_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("global_id") is None:
            continue
        key = (
            int(event["global_id"]),
            str(event["camera_id"]),
            int(event["local_track_id"]),
        )
        grouped[key].append(event)

    visits = []
    for grouped_events in grouped.values():
        for chunk in _split_timed_events(grouped_events, max_gap_seconds):
            visit = _visit_from_events(chunk)
            visit["local_track_id"] = visit.pop("local_track_ids")[0]
            visits.append(visit)
    return sorted(
        visits,
        key=lambda visit: (
            visit["global_id"],
            visit["camera_id"],
            visit["start_time"],
            visit["local_track_id"],
        ),
    )


def build_transitions(visits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_global: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for visit in visits:
        by_global[int(visit["global_id"])].append(visit)

    transitions = []
    for global_id, identity_visits in by_global.items():
        ordered = sorted(identity_visits, key=lambda item: (item["start_time"], item["camera_id"]))
        previous = ordered[0]
        for current in ordered[1:]:
            if current["camera_id"] == previous["camera_id"]:
                previous = current
                continue
            transitions.append(
                {
                    "global_id": global_id,
                    "from_camera": previous["camera_id"],
                    "to_camera": current["camera_id"],
                    "from_end_time": previous["end_time"],
                    "to_start_time": current["start_time"],
                    "gap_seconds": current["start_time"] - previous["end_time"],
                    "from_end_frame": previous["end_frame"],
                    "to_start_frame": current["start_frame"],
                    "to_entry_reason": current["entry_reason"],
                    "to_best_distance": current["entry_best_distance"],
                    "to_second_distance": current["entry_second_distance"],
                }
            )
            previous = current
    return sorted(transitions, key=lambda item: (item["to_start_time"], item["global_id"]))


def exact_time_conflicts(
    events: list[dict[str, Any]],
    topology: CameraTopology,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("global_id") is None:
            continue
        grouped[(int(event["global_id"]), round(event_time(event), 6))].append(event)

    cross_camera = []
    allowed_cross_camera = []
    same_camera_duplicates = []
    for (global_id, timestamp), grouped_events in sorted(grouped.items()):
        by_camera: dict[str, set[int]] = defaultdict(set)
        for event in grouped_events:
            by_camera[str(event["camera_id"])].add(int(event["local_track_id"]))

        if len(by_camera) > 1:
            cameras = sorted(by_camera)
            overlapping_pairs = [
                [first, second]
                for first, second in combinations(cameras, 2)
                if topology.may_overlap(first, second)
            ]
            invalid_pairs = [
                [first, second]
                for first, second in combinations(cameras, 2)
                if not topology.may_overlap(first, second)
            ]
            base_event = {
                "global_id": global_id,
                "time": timestamp,
                "cameras": cameras,
                "local_tracks": {
                    camera: sorted(local_ids) for camera, local_ids in sorted(by_camera.items())
                },
            }
            if overlapping_pairs:
                allowed_cross_camera.append(
                    {**base_event, "overlapping_camera_pairs": overlapping_pairs}
                )
            if invalid_pairs:
                cross_camera.append({**base_event, "invalid_camera_pairs": invalid_pairs})
        for camera_id, local_ids in sorted(by_camera.items()):
            if len(local_ids) > 1:
                same_camera_duplicates.append(
                    {
                        "global_id": global_id,
                        "time": timestamp,
                        "camera_id": camera_id,
                        "local_track_ids": sorted(local_ids),
                    }
                )
    return cross_camera, allowed_cross_camera, same_camera_duplicates


def build_local_recoveries(local_visits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for visit in local_visits:
        grouped[(int(visit["global_id"]), str(visit["camera_id"]))].append(visit)

    recoveries = []
    for (global_id, camera_id), visits in grouped.items():
        ordered = sorted(visits, key=lambda item: (item["start_time"], item["local_track_id"]))
        for previous, current in zip(ordered, ordered[1:]):
            if previous["local_track_id"] == current["local_track_id"]:
                continue
            if current["start_time"] <= previous["end_time"]:
                continue
            recoveries.append(
                {
                    "global_id": global_id,
                    "camera_id": camera_id,
                    "from_local_track_id": previous["local_track_id"],
                    "to_local_track_id": current["local_track_id"],
                    "gap_seconds": current["start_time"] - previous["end_time"],
                    "to_entry_reason": current["entry_reason"],
                    "to_best_distance": current["entry_best_distance"],
                }
            )
    return sorted(recoveries, key=lambda item: (item["global_id"], item["camera_id"]))


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    match = event.get("match") or {}
    return {
        "time": event_time(event),
        "camera_id": event.get("camera_id"),
        "frame_id": event.get("frame_id"),
        "local_track_id": event.get("local_track_id"),
        "global_id": event.get("global_id"),
        "reason": match.get("reason"),
        "previous_global_id": match.get("previous_global_id"),
        "candidate_global_id": match.get("candidate_global_id"),
        "best_distance": match.get("best_distance"),
        "second_distance": match.get("second_distance"),
        "assigned_distance": match.get("assigned_distance"),
        "remap_suppressed_reason": match.get("remap_suppressed_reason"),
        "blocked_candidate_global_id": match.get("blocked_candidate_global_id"),
        "blocked_candidate_distance": match.get("blocked_candidate_distance"),
        "blocked_candidate_is_better": bool(match.get("blocked_candidate_is_better", False)),
        "ambiguity_gap": (
            float(match["second_distance"]) - float(match["best_distance"])
            if match.get("best_distance") is not None and match.get("second_distance") is not None
            else None
        ),
    }


def rejection_cause(match: dict[str, Any], required_match_margin: float) -> str:
    if match.get("blocked_candidate_is_better"):
        return "simultaneous_better_candidate"

    best_distance = match.get("best_distance")
    threshold = match.get("threshold")
    if best_distance is not None and threshold is not None and float(best_distance) > float(threshold):
        return "distance_threshold"

    second_distance = match.get("second_distance")
    if best_distance is not None and second_distance is not None:
        if float(second_distance) - float(best_distance) < required_match_margin:
            return "ambiguity_margin"
    return "pending_consensus_or_other"


def analyze_events(
    events: list[dict[str, Any]],
    visit_gap_seconds: float = 1.0,
    min_travel_seconds: float = 2.0,
    close_rejection_distance: float = 0.30,
    required_match_margin: float = 0.08,
    allow_all_camera_overlap: bool = False,
    overlapping_camera_pairs: Any = None,
    adjacent_camera_pairs: Any = None,
) -> dict[str, Any]:
    topology = CameraTopology(
        allow_all_overlaps=allow_all_camera_overlap,
        overlapping_pairs=parse_overlapping_camera_pairs(overlapping_camera_pairs),
        adjacent_pairs=parse_adjacent_camera_pairs(adjacent_camera_pairs),
    )
    camera_visits = build_camera_visits(events, visit_gap_seconds)
    local_visits = build_local_visits(events, visit_gap_seconds)
    transitions = build_transitions(camera_visits)
    (
        cross_camera_conflicts,
        allowed_cross_camera,
        same_camera_duplicates,
    ) = exact_time_conflicts(events, topology)
    local_recoveries = build_local_recoveries(local_visits)

    by_global_visits: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for visit in camera_visits:
        by_global_visits[int(visit["global_id"])].append(visit)
    by_global_transitions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for transition in transitions:
        by_global_transitions[int(transition["global_id"])].append(transition)

    journeys = []
    for global_id, visits in sorted(by_global_visits.items()):
        cameras = sorted({visit["camera_id"] for visit in visits})
        if len(cameras) < 2:
            continue
        journeys.append(
            {
                "global_id": global_id,
                "cameras": cameras,
                "visits": sorted(visits, key=lambda item: item["start_time"]),
                "transitions": by_global_transitions.get(global_id, []),
            }
        )

    close_rejections = []
    remaps = []
    remap_suppressions = []
    for event in events:
        match = event.get("match") or {}
        if match.get("reason") == "appearance_remap":
            remaps.append(compact_event(event))
        if match.get("remap_suppressed_reason"):
            remap_suppressions.append(compact_event(event))
        if (
            match.get("reason") == "new_identity"
            and match.get("candidate_global_id") is not None
            and match.get("best_distance") is not None
            and float(match["best_distance"]) <= close_rejection_distance
        ):
            compact = compact_event(event)
            compact["rejection_cause"] = rejection_cause(match, required_match_margin)
            close_rejections.append(compact)

    close_rejections.sort(key=lambda item: float(item["best_distance"]))
    fast_transitions = [
        transition
        for transition in transitions
        if float(transition["gap_seconds"]) < min_travel_seconds
    ]
    allowed_overlap_transitions = [
        transition
        for transition in fast_transitions
        if topology.may_overlap(transition["from_camera"], transition["to_camera"])
    ]
    allowed_adjacent_transitions = [
        transition
        for transition in fast_transitions
        if float(transition["gap_seconds"]) >= 0.0
        and topology.are_adjacent(transition["from_camera"], transition["to_camera"])
    ]
    fast_nonoverlap_transitions = [
        transition
        for transition in fast_transitions
        if not topology.may_overlap(transition["from_camera"], transition["to_camera"])
        and not (
            float(transition["gap_seconds"]) >= 0.0
            and topology.are_adjacent(transition["from_camera"], transition["to_camera"])
        )
    ]
    assigned_events = [event for event in events if event.get("global_id") is not None]
    reason_counts = Counter((event.get("match") or {}).get("reason", "unknown") for event in events)

    return {
        "parameters": {
            "visit_gap_seconds": visit_gap_seconds,
            "min_travel_seconds": min_travel_seconds,
            "close_rejection_distance": close_rejection_distance,
            "required_match_margin": required_match_margin,
            "allow_all_camera_overlap": topology.allow_all_overlaps,
            "overlapping_camera_pairs": topology.as_pairs(),
            "adjacent_camera_pairs": topology.as_adjacent_pairs(),
        },
        "summary": {
            "events": len(events),
            "assigned_events": len(assigned_events),
            "unassigned_events": len(events) - len(assigned_events),
            "cameras_with_events_count": len({str(event["camera_id"]) for event in events}),
            "global_identity_count": len({int(event["global_id"]) for event in assigned_events}),
            "camera_visit_count": len(camera_visits),
            "cross_camera_identity_count": len(journeys),
            "cross_camera_transition_count": len(transitions),
            "fast_or_overlapping_transition_count": len(fast_transitions),
            "allowed_overlap_transition_count": len(allowed_overlap_transitions),
            "allowed_adjacent_transition_count": len(allowed_adjacent_transitions),
            "fast_nonoverlap_transition_count": len(fast_nonoverlap_transitions),
            "simultaneous_cross_camera_conflict_count": len(cross_camera_conflicts),
            "simultaneous_allowed_overlap_count": len(allowed_cross_camera),
            "same_camera_duplicate_count": len(same_camera_duplicates),
            "same_camera_local_recovery_count": len(local_recoveries),
            "appearance_remap_count": len(remaps),
            "remap_suppression_count": len(remap_suppressions),
            "close_rejected_new_identity_count": len(close_rejections),
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "journeys": journeys,
        "transitions": transitions,
        "fast_or_overlapping_transitions": fast_transitions,
        "allowed_overlap_transitions": allowed_overlap_transitions,
        "allowed_adjacent_transitions": allowed_adjacent_transitions,
        "fast_nonoverlap_transitions": fast_nonoverlap_transitions,
        "simultaneous_cross_camera_conflicts": cross_camera_conflicts,
        "simultaneous_allowed_overlaps": allowed_cross_camera,
        "same_camera_duplicates": same_camera_duplicates,
        "same_camera_local_recoveries": local_recoveries,
        "appearance_remaps": remaps,
        "remap_suppressions": remap_suppressions,
        "close_rejected_new_identities": close_rejections,
    }


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("Cross-camera ReID audit")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    for journey in report["journeys"]:
        print(f"\nID {journey['global_id']} cameras={','.join(journey['cameras'])}")
        for visit in journey["visits"]:
            print(
                f"  {visit['camera_id']} {visit['start_time']:.1f}-{visit['end_time']:.1f}s "
                f"locals={visit['local_track_ids']} entry={visit['entry_reason']}"
            )
        for transition in journey["transitions"]:
            print(
                f"    {transition['from_camera']}->{transition['to_camera']} "
                f"gap={transition['gap_seconds']:.1f}s "
                f"distance={format_optional_float(transition['to_best_distance'])}"
            )

    if report["close_rejected_new_identities"]:
        print("\nClose rejected candidates")
        for event in report["close_rejected_new_identities"]:
            print(
                f"  new ID {event['global_id']} at {event['camera_id']} {event['time']:.1f}s "
                f"candidate={event['candidate_global_id']} "
                f"distance={format_optional_float(event['best_distance'])} "
                f"margin={format_optional_float(event['ambiguity_gap'])} "
                f"cause={event['rejection_cause']}"
            )


def format_optional_float(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def default_output_path(events_path: Path) -> Path:
    if events_path.parent.name == "reid_debug":
        return events_path.parent.parent / "cross_camera_analysis.json"
    return events_path.with_name("cross_camera_analysis.json")


def main() -> None:
    args = parse_args()
    events = load_events(args.events)
    report = analyze_events(
        events,
        visit_gap_seconds=args.visit_gap_seconds,
        min_travel_seconds=args.min_travel_seconds,
        close_rejection_distance=args.close_rejection_distance,
        required_match_margin=args.required_match_margin,
        allow_all_camera_overlap=args.allow_all_camera_overlap,
        overlapping_camera_pairs=args.overlapping_camera_pairs,
        adjacent_camera_pairs=args.adjacent_camera_pairs,
    )
    output_path = args.output or default_output_path(args.events)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as report_file:
        json.dump(report, report_file, indent=2)
    print_report(report)
    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
