#!/usr/bin/env python3
"""Create transition-aware ReID contact sheets from a completed debug run."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CAMERA_COLORS = {
    "ch201": (180, 122, 75),
    "ch301": (91, 155, 213),
    "ch401": (120, 174, 96),
    "ch501": (172, 113, 190),
    "ch601": (77, 173, 208),
    "ch701": (166, 141, 86),
    "ch901": (151, 91, 188),
}

REASON_LABELS = {
    "new_identity": "new identity",
    "appearance_match_verified_active_overlap": "active overlap",
    "appearance_match_verified_pending_average": "temporal average",
    "appearance_match_verified_adjacent_handoff": "adjacent handoff",
    "appearance_match_verified_overlap_lightness": "overlap lightness",
    "low_quality_existing_track_hold": "low-quality hold",
    "existing_local_track_verified": "local continuation",
}


@dataclass
class Source:
    path: Path
    capture: cv2.VideoCapture
    width: int
    height: int


@dataclass
class Visit:
    global_id: int
    camera_id: str
    events: list[dict[str, Any]]

    @property
    def start_time(self) -> float:
        return event_time(self.events[0])

    @property
    def end_time(self) -> float:
        return event_time(self.events[-1])

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def entry_match(self) -> dict[str, Any]:
        return self.events[0].get("match") or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "events",
        type=Path,
        help="Flattened ReID event log, normally reid_debug/events.jsonl.",
    )
    parser.add_argument(
        "--identity-manifest",
        type=Path,
        default=None,
        help="Identity-video manifest containing source_recordings.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--visit-gap-seconds", type=float, default=1.0)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=288)
    parser.add_argument("--tile-height", type=int, default=420)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


def event_time(event: dict[str, Any]) -> float:
    return float(event.get("time", event.get("timestamp", 0.0)))


def default_experiment_dir(events_path: Path) -> Path:
    if events_path.parent.name == "reid_debug":
        return events_path.parent.parent
    return events_path.parent


def load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open() as event_file:
        for line in event_file:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("global_id") is not None:
                events.append(event)
    return events


def load_sources(manifest_path: Path) -> dict[str, Source]:
    with manifest_path.open() as manifest_file:
        manifest = json.load(manifest_file)

    sources = {}
    for camera_id, raw_path in sorted(manifest["source_recordings"].items()):
        path = Path(raw_path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source recording: {path}")
        sources[camera_id] = Source(
            path=path,
            capture=capture,
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    return sources


def build_visits(
    events: list[dict[str, Any]],
    max_gap_seconds: float,
) -> dict[int, list[Visit]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        key = (int(event["global_id"]), str(event["camera_id"]))
        grouped[key].append(event)

    by_identity: dict[int, list[Visit]] = defaultdict(list)
    for (global_id, camera_id), camera_events in grouped.items():
        ordered = sorted(
            camera_events,
            key=lambda item: (event_time(item), int(item.get("frame_id", -1))),
        )
        chunks = [[ordered[0]]]
        for event in ordered[1:]:
            if event_time(event) - event_time(chunks[-1][-1]) > max_gap_seconds:
                chunks.append([])
            chunks[-1].append(event)
        by_identity[global_id].extend(
            Visit(global_id=global_id, camera_id=camera_id, events=chunk)
            for chunk in chunks
        )

    for visits in by_identity.values():
        visits.sort(key=lambda item: (item.start_time, item.camera_id))
    return dict(sorted(by_identity.items()))


def touches_edge(event: dict[str, Any], source: Source) -> bool:
    x1, y1, x2, y2 = [float(value) for value in event["bbox"]]
    margin_x = source.width * 0.01
    margin_y = source.height * 0.01
    return (
        x1 <= margin_x
        or y1 <= margin_y
        or x2 >= source.width - margin_x
        or y2 >= source.height - margin_y
    )


def event_score(event: dict[str, Any], visit: Visit, source: Source) -> float:
    x1, y1, x2, y2 = [float(value) for value in event["bbox"]]
    visible_width = max(0.0, min(source.width, x2) - max(0.0, x1))
    visible_height = max(0.0, min(source.height, y2) - max(0.0, y1))
    height_ratio = visible_height / max(1.0, source.height)
    area_ratio = visible_width * visible_height / max(1.0, source.width * source.height)
    midpoint = (visit.start_time + visit.end_time) / 2.0
    half_duration = max(0.5, visit.duration / 2.0)
    midpoint_penalty = abs(event_time(event) - midpoint) / half_duration
    return (
        5.0 * float(not touches_edge(event, source))
        + 1.5 * float(event.get("confidence", 0.0))
        + min(height_ratio, 0.9)
        + min(area_ratio * 5.0, 1.0)
        - 0.2 * midpoint_penalty
    )


def read_representative(visit: Visit, source: Source) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = sorted(
        visit.events,
        key=lambda event: event_score(event, visit, source),
        reverse=True,
    )
    for event in candidates:
        source.capture.set(cv2.CAP_PROP_POS_FRAMES, int(event["frame_id"]))
        ok, frame = source.capture.read()
        if not ok or frame is None:
            continue
        crop = padded_crop(frame, event["bbox"])
        if crop is not None:
            return crop, event
    raise RuntimeError(
        f"Could not read a representative frame for ID {visit.global_id} "
        f"{visit.camera_id} at {visit.start_time:.2f}s"
    )


def padded_crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    x1 = max(0, int(math.floor(x1 - width * 0.08)))
    y1 = max(0, int(math.floor(y1 - height * 0.08)))
    x2 = min(frame_width, int(math.ceil(x2 + width * 0.08)))
    y2 = min(frame_height, int(math.ceil(y2 + height * 0.08)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def camera_color(camera_id: str) -> tuple[int, int, int]:
    if camera_id in CAMERA_COLORS:
        return CAMERA_COLORS[camera_id]
    seed = sum(ord(character) for character in camera_id)
    return (
        80 + (seed * 31) % 130,
        80 + (seed * 47) % 130,
        80 + (seed * 59) % 130,
    )


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.full((height, width, 3), 17, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def reason_label(reason: Any) -> str:
    raw_reason = str(reason or "unknown")
    return REASON_LABELS.get(raw_reason, raw_reason.replace("_", " "))


def render_visit_tile(
    visit: Visit,
    visit_index: int,
    crop: np.ndarray,
    event: dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    header_height = 68
    footer_height = 92
    body_height = height - header_height - footer_height
    color = camera_color(visit.camera_id)
    tile = np.full((height, width, 3), 20, dtype=np.uint8)
    tile[:header_height] = color
    tile[header_height : header_height + body_height] = fit_image(
        crop,
        width,
        body_height,
    )

    put_text(
        tile,
        f"ID {visit.global_id}  |  VISIT {visit_index:02d}",
        (10, 25),
        0.55,
        (255, 255, 255),
        2,
    )
    put_text(
        tile,
        f"{visit.camera_id}  {visit.start_time:.1f}-{visit.end_time:.1f}s",
        (10, 52),
        0.53,
        (255, 255, 255),
        1,
    )

    footer_y = header_height + body_height
    match = visit.entry_match
    distance = match.get("best_distance")
    distance_text = "-" if distance is None else f"{float(distance):.3f}"
    put_text(
        tile,
        f"sample {event_time(event):.1f}s  det={float(event.get('confidence', 0.0)):.2f}",
        (10, footer_y + 23),
        0.43,
    )
    put_text(
        tile,
        f"entry: {reason_label(match.get('reason'))}",
        (10, footer_y + 48),
        0.40,
    )
    put_text(
        tile,
        f"ReID d={distance_text}  observations={len(visit.events)}",
        (10, footer_y + 73),
        0.40,
    )
    cv2.rectangle(tile, (1, 1), (width - 2, height - 2), color, 3)
    return tile


def transition_kind(previous: Visit, current: Visit) -> tuple[str, float]:
    gap = current.start_time - previous.end_time
    if gap < 0:
        return "OVERLAP", gap
    if previous.camera_id == current.camera_id:
        return "REAPPEAR", gap
    if gap <= 2.0:
        return "HANDOFF", gap
    return "REAPPEAR", gap


def render_transition_tile(
    previous: Visit,
    current: Visit,
    width: int,
    height: int,
) -> np.ndarray:
    tile = np.full((height, width, 3), 16, dtype=np.uint8)
    kind, gap = transition_kind(previous, current)
    kind_color = {
        "OVERLAP": (73, 170, 230),
        "HANDOFF": (102, 194, 102),
        "REAPPEAR": (180, 145, 91),
    }[kind]
    put_text(tile, kind, (20, 55), 0.8, kind_color, 2)
    put_text(
        tile,
        f"{previous.camera_id}  ->  {current.camera_id}",
        (20, height // 2 - 15),
        0.72,
        (245, 245, 245),
        2,
    )
    if gap < 0:
        timing = f"simultaneous for {-gap:.1f}s"
    else:
        timing = f"gap {gap:.1f}s"
    put_text(tile, timing, (20, height // 2 + 22), 0.53, (190, 190, 190), 1)

    match = current.entry_match
    distance = match.get("best_distance")
    distance_text = "-" if distance is None else f"{float(distance):.3f}"
    put_text(
        tile,
        f"entry: {reason_label(match.get('reason'))}",
        (20, height - 72),
        0.43,
        (205, 205, 205),
    )
    put_text(
        tile,
        f"ReID distance: {distance_text}",
        (20, height - 42),
        0.43,
        (205, 205, 205),
    )
    cv2.arrowedLine(
        tile,
        (45, height // 2 + 65),
        (width - 45, height // 2 + 65),
        kind_color,
        3,
        cv2.LINE_AA,
        tipLength=0.08,
    )
    cv2.rectangle(tile, (1, 1), (width - 2, height - 2), kind_color, 2)
    return tile


def render_identity_sheet(
    global_id: int,
    visits: list[Visit],
    sources: dict[str, Source],
    columns: int,
    tile_width: int,
    tile_height: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    tiles = []
    visit_rows = []
    previous = None
    for visit_index, visit in enumerate(visits, start=1):
        if previous is not None:
            tiles.append(
                render_transition_tile(
                    previous,
                    visit,
                    tile_width,
                    tile_height,
                )
            )
        crop, event = read_representative(visit, sources[visit.camera_id])
        tiles.append(
            render_visit_tile(
                visit,
                visit_index,
                crop,
                event,
                tile_width,
                tile_height,
            )
        )
        kind, gap = (None, None)
        if previous is not None:
            kind, gap = transition_kind(previous, visit)
        visit_rows.append(
            {
                "visit_index": visit_index,
                "camera_id": visit.camera_id,
                "start_time": visit.start_time,
                "end_time": visit.end_time,
                "observations": len(visit.events),
                "representative_time": event_time(event),
                "representative_frame": int(event["frame_id"]),
                "transition_kind": kind,
                "transition_gap_seconds": gap,
                "entry_reason": visit.entry_match.get("reason"),
                "entry_best_distance": visit.entry_match.get("best_distance"),
            }
        )
        previous = visit

    gap = 8
    title_height = 112
    rows = math.ceil(len(tiles) / columns)
    width = columns * tile_width + (columns + 1) * gap
    height = title_height + rows * tile_height + (rows + 1) * gap
    sheet = np.full((height, width, 3), 238, dtype=np.uint8)
    cameras = sorted({visit.camera_id for visit in visits})
    cross_camera = sum(
        left.camera_id != right.camera_id
        for left, right in zip(visits, visits[1:], strict=False)
    )
    put_text(sheet, f"GLOBAL ID {global_id} - TRANSITION CONTACT SHEET", (16, 38), 0.9, (35, 35, 35), 2)
    put_text(
        sheet,
        f"cameras: {', '.join(cameras)}   visits: {len(visits)}   cross-camera entries: {cross_camera}",
        (16, 72),
        0.55,
        (55, 55, 55),
        1,
    )
    put_text(
        sheet,
        "Orange=overlap, green=handoff <=2s, blue=reappearance. "
        "All IDs and transitions are pipeline output.",
        (16, 98),
        0.43,
        (70, 70, 70),
        1,
    )

    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        x = gap + column * (tile_width + gap)
        y = title_height + gap + row * (tile_height + gap)
        sheet[y : y + tile_height, x : x + tile_width] = tile
    return sheet, visit_rows


def best_camera_event(
    events: list[dict[str, Any]],
    global_id: int,
    camera_id: str,
    source: Source,
) -> dict[str, Any]:
    candidates = [
        event
        for event in events
        if int(event["global_id"]) == global_id and str(event["camera_id"]) == camera_id
    ]
    synthetic_visit = Visit(global_id, camera_id, candidates)
    return max(candidates, key=lambda event: event_score(event, synthetic_visit, source))


def render_overview(
    events: list[dict[str, Any]],
    visits_by_identity: dict[int, list[Visit]],
    sources: dict[str, Source],
) -> np.ndarray:
    camera_ids = sorted(sources)
    identity_ids = sorted(visits_by_identity)
    label_width = 170
    tile_width = 210
    tile_height = 300
    header_height = 92
    gap = 8
    width = label_width + len(camera_ids) * (tile_width + gap) + gap
    height = header_height + len(identity_ids) * (tile_height + gap) + gap
    overview = np.full((height, width, 3), 240, dtype=np.uint8)

    put_text(
        overview,
        "FINAL REID CAMERA-TRANSITION OVERVIEW",
        (16, 38),
        0.95,
        (35, 35, 35),
        2,
    )
    put_text(
        overview,
        "One representative crop per assigned global ID and camera",
        (16, 70),
        0.55,
        (65, 65, 65),
        1,
    )

    for column, camera_id in enumerate(camera_ids):
        x = label_width + column * (tile_width + gap)
        color = camera_color(camera_id)
        cv2.rectangle(overview, (x, 8), (x + tile_width, header_height - 8), color, -1)
        put_text(overview, camera_id, (x + 54, 58), 0.7, (255, 255, 255), 2)

    for row, global_id in enumerate(identity_ids):
        y = header_height + row * (tile_height + gap)
        cameras = sorted({visit.camera_id for visit in visits_by_identity[global_id]})
        put_text(overview, f"ID {global_id}", (18, y + 62), 0.85, (35, 35, 35), 2)
        put_text(
            overview,
            f"{len(cameras)} camera{'s' if len(cameras) != 1 else ''}",
            (18, y + 92),
            0.48,
            (70, 70, 70),
            1,
        )
        for column, camera_id in enumerate(camera_ids):
            x = label_width + column * (tile_width + gap)
            if camera_id not in cameras:
                cv2.rectangle(
                    overview,
                    (x, y),
                    (x + tile_width, y + tile_height),
                    (220, 220, 220),
                    -1,
                )
                put_text(
                    overview,
                    "not observed",
                    (x + 43, y + tile_height // 2),
                    0.48,
                    (145, 145, 145),
                    1,
                )
                continue
            event = best_camera_event(events, global_id, camera_id, sources[camera_id])
            sources[camera_id].capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(event["frame_id"]),
            )
            ok, frame = sources[camera_id].capture.read()
            crop = padded_crop(frame, event["bbox"]) if ok and frame is not None else None
            if crop is None:
                continue
            body = fit_image(crop, tile_width, tile_height - 42)
            overview[y + 42 : y + tile_height, x : x + tile_width] = body
            color = camera_color(camera_id)
            cv2.rectangle(overview, (x, y), (x + tile_width, y + 42), color, -1)
            put_text(
                overview,
                f"{camera_id}  t={event_time(event):.1f}s",
                (x + 8, y + 27),
                0.46,
                (255, 255, 255),
                1,
            )
            cv2.rectangle(
                overview,
                (x, y),
                (x + tile_width, y + tile_height),
                color,
                2,
            )
    return overview


def compact_camera_sequence(visits: list[Visit]) -> list[Visit]:
    compacted = []
    for visit in visits:
        if compacted and compacted[-1].camera_id == visit.camera_id:
            previous = compacted[-1]
            compacted[-1] = Visit(
                global_id=visit.global_id,
                camera_id=visit.camera_id,
                events=previous.events + visit.events,
            )
        else:
            compacted.append(visit)
    return compacted


def render_transition_graph(visits_by_identity: dict[int, list[Visit]]) -> np.ndarray:
    max_nodes_per_line = 8
    label_width = 140
    node_width = 132
    node_height = 72
    horizontal_gap = 55
    line_height = 112
    identity_gap = 18
    header_height = 128
    margin = 18

    sequences = {
        global_id: compact_camera_sequence(visits)
        for global_id, visits in visits_by_identity.items()
    }
    line_counts = {
        global_id: max(1, math.ceil(len(sequence) / max_nodes_per_line))
        for global_id, sequence in sequences.items()
    }
    width = (
        label_width
        + max_nodes_per_line * node_width
        + (max_nodes_per_line - 1) * horizontal_gap
        + 2 * margin
    )
    height = (
        header_height
        + sum(line_counts.values()) * line_height
        + len(sequences) * identity_gap
        + margin
    )
    graph = np.full((height, width, 3), 242, dtype=np.uint8)

    put_text(
        graph,
        "FINAL REID TRANSITION GRAPH",
        (margin, 38),
        0.95,
        (35, 35, 35),
        2,
    )
    put_text(
        graph,
        "Nodes are chronological camera entries. Consecutive same-camera recoveries are collapsed.",
        (margin, 70),
        0.53,
        (65, 65, 65),
        1,
    )
    put_text(
        graph,
        "Edge labels: O=overlapping visibility, H=handoff <=2s, R=reappearance.",
        (margin, 98),
        0.48,
        (65, 65, 65),
        1,
    )

    y = header_height
    for global_id, sequence in sequences.items():
        lines = line_counts[global_id]
        group_height = lines * line_height
        put_text(
            graph,
            f"ID {global_id}",
            (margin, y + 34),
            0.75,
            (35, 35, 35),
            2,
        )
        put_text(
            graph,
            f"{len(sequence)} nodes",
            (margin, y + 62),
            0.43,
            (80, 80, 80),
            1,
        )

        positions = []
        for index, visit in enumerate(sequence):
            row, column = divmod(index, max_nodes_per_line)
            x = margin + label_width + column * (node_width + horizontal_gap)
            node_y = y + row * line_height
            positions.append((x, node_y))
            color = camera_color(visit.camera_id)
            cv2.rectangle(
                graph,
                (x, node_y),
                (x + node_width, node_y + node_height),
                color,
                -1,
            )
            put_text(
                graph,
                f"{index + 1:02d}  {visit.camera_id}",
                (x + 9, node_y + 29),
                0.49,
                (255, 255, 255),
                2,
            )
            put_text(
                graph,
                f"t={visit.start_time:.1f}s",
                (x + 9, node_y + 55),
                0.43,
                (255, 255, 255),
                1,
            )

        for index in range(1, len(sequence)):
            previous = sequence[index - 1]
            current = sequence[index]
            previous_x, previous_y = positions[index - 1]
            current_x, current_y = positions[index]
            kind, gap = transition_kind(previous, current)
            edge_color = {
                "OVERLAP": (73, 170, 230),
                "HANDOFF": (102, 194, 102),
                "REAPPEAR": (180, 145, 91),
            }[kind]
            edge_label = {
                "OVERLAP": "O",
                "HANDOFF": "H",
                "REAPPEAR": "R",
            }[kind]
            timing = f"{-gap:.1f}s" if gap < 0 else f"{gap:.1f}s"

            if previous_y == current_y:
                start = (previous_x + node_width + 4, previous_y + node_height // 2)
                end = (current_x - 5, current_y + node_height // 2)
                cv2.arrowedLine(
                    graph,
                    start,
                    end,
                    edge_color,
                    2,
                    cv2.LINE_AA,
                    tipLength=0.15,
                )
                put_text(
                    graph,
                    f"{edge_label} {timing}",
                    (start[0] + 3, start[1] - 10),
                    0.34,
                    edge_color,
                    1,
                )
            else:
                start = (
                    previous_x + node_width // 2,
                    previous_y + node_height + 2,
                )
                end = (current_x + node_width // 2, current_y - 4)
                bend_y = start[1] + 14
                cv2.line(
                    graph,
                    start,
                    (start[0], bend_y),
                    edge_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.line(
                    graph,
                    (start[0], bend_y),
                    (end[0], bend_y),
                    edge_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.arrowedLine(
                    graph,
                    (end[0], bend_y),
                    end,
                    edge_color,
                    2,
                    cv2.LINE_AA,
                    tipLength=0.15,
                )
                put_text(
                    graph,
                    f"{edge_label} {timing}",
                    (min(start[0], end[0]) + 8, bend_y - 5),
                    0.34,
                    edge_color,
                    1,
                )

        cv2.line(
            graph,
            (margin, y + group_height - 5),
            (width - margin, y + group_height - 5),
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        y += group_height + identity_gap
    return graph


def write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise RuntimeError(f"Failed to write image: {path}")


def main() -> None:
    args = parse_args()
    experiment_dir = default_experiment_dir(args.events)
    manifest_path = (
        args.identity_manifest
        if args.identity_manifest is not None
        else experiment_dir / "identity_videos" / "manifest.json"
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else experiment_dir / "final_evaluation" / "transition_contact_sheets"
    )

    events = load_events(args.events)
    visits_by_identity = build_visits(events, args.visit_gap_seconds)
    sources = load_sources(manifest_path)
    report = {
        "source_events": str(args.events),
        "source_manifest": str(manifest_path),
        "visit_gap_seconds": args.visit_gap_seconds,
        "identity_count": len(visits_by_identity),
        "identities": [],
    }

    try:
        for global_id, visits in visits_by_identity.items():
            sheet, visit_rows = render_identity_sheet(
                global_id,
                visits,
                sources,
                max(1, args.columns),
                args.tile_width,
                args.tile_height,
            )
            path = output_dir / f"id_{global_id:04d}_transition_contact_sheet.jpg"
            write_jpeg(path, sheet, args.jpeg_quality)
            report["identities"].append(
                {
                    "global_id": global_id,
                    "path": str(path),
                    "cameras": sorted({visit.camera_id for visit in visits}),
                    "visits": visit_rows,
                }
            )
            print(f"Wrote: {path}")

        overview = render_overview(events, visits_by_identity, sources)
        overview_path = output_dir / "all_identities_camera_overview.jpg"
        write_jpeg(overview_path, overview, args.jpeg_quality)
        report["overview_path"] = str(overview_path)

        transition_graph = render_transition_graph(visits_by_identity)
        transition_graph_path = output_dir / "all_identities_transition_graph.jpg"
        write_jpeg(transition_graph_path, transition_graph, args.jpeg_quality)
        report["transition_graph_path"] = str(transition_graph_path)

        manifest_output = output_dir / "manifest.json"
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        with manifest_output.open("w") as manifest_file:
            json.dump(report, manifest_file, indent=2)
        print(f"Wrote: {overview_path}")
        print(f"Wrote: {transition_graph_path}")
        print(f"Wrote: {manifest_output}")
    finally:
        for source in sources.values():
            source.capture.release()


if __name__ == "__main__":
    main()
