"""
Multi-Camera Burglar Detection
Sequential cross-camera ReID: Burglar_comes_in (cam1) → Burglar_runs (cam2)

The two videos are treated as sequential footage from different rooms.
A single tracker instance persists across both videos — the burglar keeps
the same track ID via appearance-based ReID when re-entering camera 2.

Output: side-by-side video where cam1 plays then freezes, then cam2 activates.
"""

import subprocess
import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import YOLOPersonDetector
from src.reid_client import TritonReIDClient
from src.tracker import ReIDTracker
from src.utils.config_loader import load_all_configs


# ─── Config ────────────────────────────────────────────────────────────────────
CAM1_PATH = Path("test_videos/Burglar_comes_in_id0.mp4")
CAM2_PATH = Path("test_videos/Burglar_runs_id0.mp4")
OUTPUT_PATH     = Path("outputs/multicam_burglar_output.mp4")
OUTPUT_H264     = Path("outputs/multicam_burglar_output_h264.mp4")

CAM1_LABEL = "CAM 1 · Entrance"
CAM2_LABEL = "CAM 2 · Inner Room"

# Frames of empty detections to feed between videos (simulates time gap,
# causes tracker to mark track as "lost" so appearance-ReID re-identifies in cam2)
GAP_FRAMES = 60

# Output resolution — each camera pane
PANE_W, PANE_H = 960, 540
DIVIDER = 4          # px divider between panes
OUT_W = PANE_W * 2 + DIVIDER
OUT_H = PANE_H + 120  # extra bar at bottom for timeline / status

# ─── Colors ────────────────────────────────────────────────────────────────────
RED     = (0,   0,   220)
GREEN   = (0,   200, 0  )
YELLOW  = (0,   210, 240)
WHITE   = (255, 255, 255)
BLACK   = (0,   0,   0  )
GRAY    = (60,  60,  60 )
ORANGE  = (0,   140, 255)
CYAN    = (220, 200, 0  )

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX

TRACK_COLORS = [
    (255,  80,  80),   # id 1 – blue-ish
    ( 80, 200,  80),   # id 2 – green
    ( 80,  80, 255),   # id 3 – red
    (255, 200,  80),   # id 4 – cyan
    (255,  80, 255),   # id 5 – magenta
    ( 80, 255, 255),   # id 6 – yellow
]


def track_color(track_id: int):
    return TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]


# ─── Drawing helpers ───────────────────────────────────────────────────────────

def draw_track_box(frame, track, alert_ids=None):
    """Draw a bounding box + ID label for a single track."""
    if len(track) == 8:
        x1, y1, x2, y2, tid, conf, cls, _ = track
    else:
        x1, y1, x2, y2, tid, conf, cls = track
    x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)

    color = track_color(tid)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID:{tid}"
    (tw, th), _ = cv2.getTextSize(label, FONT_BOLD, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 3), FONT_BOLD, 0.55, BLACK, 1, cv2.LINE_AA)


def draw_hud(frame, cam_label, is_active, cam_w):
    """Overlay minimal camera label on a pane."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (cam_w, 36), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    label_color = WHITE if is_active else GRAY
    cv2.putText(frame, cam_label, (10, 25), FONT_BOLD, 0.68, label_color, 1, cv2.LINE_AA)


def make_status_bar(out_w, status_h, global_time_s,
                    cam1_active, cam2_active,
                    cam1_ids, cam2_ids, cross_ids,
                    cam1_total_s, cam2_total_s):
    """Create the bottom status bar image."""
    bar = np.zeros((status_h, out_w, 3), dtype=np.uint8)
    bar[:] = (20, 20, 20)

    # Timeline line
    tl_y = status_h // 2 + 10
    total_s = cam1_total_s + cam2_total_s
    tl_x0, tl_x1 = 20, out_w - 20
    tl_len = tl_x1 - tl_x0

    cv2.line(bar, (tl_x0, tl_y), (tl_x1, tl_y), GRAY, 2)

    # Cam1 segment
    cam1_end_x = tl_x0 + int(tl_len * cam1_total_s / total_s)
    cv2.line(bar, (tl_x0, tl_y), (cam1_end_x, tl_y),
             GREEN if cam1_active else (60, 100, 60), 4)

    # Cam2 segment
    cv2.line(bar, (cam1_end_x, tl_y), (tl_x1, tl_y),
             ORANGE if cam2_active else (60, 60, 100), 4)

    # Segment labels
    cv2.putText(bar, CAM1_LABEL, (tl_x0, tl_y - 12), FONT, 0.40, WHITE, 1)
    cv2.putText(bar, CAM2_LABEL, (cam1_end_x + 6, tl_y - 12), FONT, 0.40, WHITE, 1)

    # Playhead dot
    elapsed_frac = min(global_time_s / total_s, 1.0)
    ph_x = tl_x0 + int(tl_len * elapsed_frac)
    cv2.circle(bar, (ph_x, tl_y), 7, WHITE, -1)

    # Timestamp
    mins = int(global_time_s) // 60
    secs = int(global_time_s) % 60
    ts = f"  {mins:02d}:{secs:02d}"
    cv2.putText(bar, ts, (ph_x + 10, tl_y + 5), FONT, 0.45, WHITE, 1)

    return bar


def resize_to_pane(frame):
    return cv2.resize(frame, (PANE_W, PANE_H), interpolation=cv2.INTER_AREA)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" Multi-Camera Burglar Detection Pipeline")
    print("=" * 65)

    # Load configs
    configs = load_all_configs()

    # ── Init components ──
    print("\n[1/3] Initializing YOLO detector…")
    detector = YOLOPersonDetector(configs['yolo'])

    print("[2/3] Initializing Triton ReID client…")
    reid_client = TritonReIDClient(configs['reid'])

    print("[3/3] Initializing BoTSORT tracker (shared across cameras)…")
    tracker = ReIDTracker(configs['tracker'])

    # ── Open videos ──
    cap1 = cv2.VideoCapture(str(CAM1_PATH))
    cap2 = cv2.VideoCapture(str(CAM2_PATH))

    if not cap1.isOpened():
        raise FileNotFoundError(f"Cannot open {CAM1_PATH}")
    if not cap2.isOpened():
        raise FileNotFoundError(f"Cannot open {CAM2_PATH}")

    fps   = cap1.get(cv2.CAP_PROP_FPS)
    total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    cam1_total_s = total1 / fps
    cam2_total_s = total2 / fps

    print(f"\nCAM1: {CAM1_PATH.name}  ({total1} frames @ {fps:.2f} fps = {cam1_total_s:.1f}s)")
    print(f"CAM2: {CAM2_PATH.name}  ({total2} frames @ {fps:.2f} fps = {cam2_total_s:.1f}s)")
    print(f"Output: {OUTPUT_PATH}  ({OUT_W}×{OUT_H})")

    # ── Setup writer ──
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (OUT_W, OUT_H))
    status_h = OUT_H - PANE_H

    # State tracking
    cam1_ids_seen: set = set()
    cam2_ids_seen: set = set()
    cross_ids:     set = set()           # IDs seen in BOTH cameras (ReID success)
    cam1_frames = []                     # Processed cam1 panes (for freeze)
    cam2_first_frame = None              # Frozen cam2 preview pane
    last_cam1_pane = None

    # ─── PHASE 1: Process CAM1 ─────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f" PHASE 1: Processing {CAM1_LABEL}")
    print(f"{'─'*65}")

    ret, cam2_first_raw = cap2.read()
    cam2_first_pane = resize_to_pane(cam2_first_raw) if ret else \
        np.zeros((PANE_H, PANE_W, 3), dtype=np.uint8)
    # Dim cam2 preview during phase 1 (it's not active yet)
    cam2_first_pane = (cam2_first_pane.astype(np.float32) * 0.45).astype(np.uint8)
    draw_hud(cam2_first_pane, CAM2_LABEL, False, PANE_W)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind

    frame_idx = 0
    t0 = time.time()

    while cap1.isOpened():
        ret, frame = cap1.read()
        if not ret:
            break

        # Detect + embed + track
        detections, crops = detector.detect(frame)
        if len(crops) > 0:
            embeddings = reid_client.infer(crops)
        else:
            embeddings = np.empty((0, 0))
        tracks = tracker.update(detections, frame, embeddings)

        track_ids = [int(t[4]) for t in tracks] if len(tracks) > 0 else []
        cam1_ids_seen.update(track_ids)

        # Draw tracks on cam1 frame
        vis = frame.copy()
        for t in tracks:
            draw_track_box(vis, t)

        pane1 = resize_to_pane(vis)
        draw_hud(pane1, CAM1_LABEL, True, PANE_W)

        last_cam1_pane = pane1.copy()

        # Compose output frame
        divider_col = np.zeros((PANE_H, DIVIDER, 3), dtype=np.uint8)
        combined = np.hstack([pane1, divider_col, cam2_first_pane])

        status = make_status_bar(OUT_W, status_h,
                                 frame_idx / fps,
                                 True, False,
                                 cam1_ids_seen, cam2_ids_seen, cross_ids,
                                 cam1_total_s, cam2_total_s)

        full = np.vstack([combined, status])
        out.write(full)

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            ids_str = str(sorted(cam1_ids_seen)) if cam1_ids_seen else "none yet"
            print(f"  CAM1 frame {frame_idx}/{total1}  |  tracks: {len(tracks)}  |  IDs: {ids_str}  |  {elapsed:.1f}s elapsed")

    cap1.release()
    print(f"\nCAM1 done. Intruder IDs seen: {sorted(cam1_ids_seen)}")

    # ─── GAP: Feed empty frames to mark tracks as lost ─────────────────────
    print(f"\nFeeding {GAP_FRAMES} gap frames to put tracks in 'lost' state…")
    dummy_frame = np.zeros((PANE_H, PANE_W, 3), dtype=np.uint8)
    for _ in range(GAP_FRAMES):
        tracker.update(np.empty((0, 6)), dummy_frame, np.empty((0, 0)))

    # ─── PHASE 2: Process CAM2 ─────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f" PHASE 2: Processing {CAM2_LABEL}")
    print(f"{'─'*65}")

    cam2_frame_idx = 0
    t1 = time.time()
    frozen_cam1 = last_cam1_pane.copy() if last_cam1_pane is not None else \
        np.zeros((PANE_H, PANE_W, 3), dtype=np.uint8)

    # Dim the frozen cam1 (no extra text — dimming signals it's inactive)
    frozen_cam1_display = (frozen_cam1.astype(np.float32) * 0.45).astype(np.uint8)
    draw_hud(frozen_cam1_display, CAM1_LABEL, False, PANE_W)

    while cap2.isOpened():
        ret, frame = cap2.read()
        if not ret:
            break

        detections, crops = detector.detect(frame)
        if len(crops) > 0:
            embeddings = reid_client.infer(crops)
        else:
            embeddings = np.empty((0, 0))
        tracks = tracker.update(detections, frame, embeddings)

        track_ids = [int(t[4]) for t in tracks] if len(tracks) > 0 else []
        cam2_ids_seen.update(track_ids)

        # Check cross-camera ReID: any ID seen in BOTH cameras?
        cross_ids = cam1_ids_seen & cam2_ids_seen

        # Draw tracks on cam2 frame
        vis = frame.copy()
        for t in tracks:
            draw_track_box(vis, t, alert_ids=cross_ids)

        pane2 = resize_to_pane(vis)
        global_time = cam1_total_s + cam2_frame_idx / fps
        draw_hud(pane2, CAM2_LABEL, True, PANE_W)

        # Compose
        divider_col = np.zeros((PANE_H, DIVIDER, 3), dtype=np.uint8)
        combined = np.hstack([frozen_cam1_display, divider_col, pane2])

        status = make_status_bar(OUT_W, status_h,
                                 global_time,
                                 False, True,
                                 cam1_ids_seen, cam2_ids_seen, cross_ids,
                                 cam1_total_s, cam2_total_s)

        full = np.vstack([combined, status])
        out.write(full)

        cam2_frame_idx += 1
        if cam2_frame_idx % 100 == 0:
            elapsed = time.time() - t1
            ids_str = str(sorted(cam2_ids_seen)) if cam2_ids_seen else "none yet"
            cross_str = str(sorted(cross_ids)) if cross_ids else "none yet"
            print(f"  CAM2 frame {cam2_frame_idx}/{total2}  |  tracks: {len(tracks)}  |  IDs: {ids_str}  |  cross-cam: {cross_str}  |  {elapsed:.1f}s elapsed")

    cap2.release()

    # Cleanup
    out.release()

    # Re-encode to H.264 for universal compatibility
    print(f"\nRe-encoding to H.264 for compatibility…")
    OUTPUT_H264.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(OUTPUT_PATH),
        "-c:v", "libopenh264", "-b:v", "4M", "-pix_fmt", "yuv420p",
        str(OUTPUT_H264)
    ], check=True, capture_output=True)
    OUTPUT_PATH.unlink()   # remove raw mp4v file

    total_time = time.time() - t0
    print(f"\n{'='*65}")
    print(f" Done!  Total time: {total_time:.1f}s")
    print(f" Output: {OUTPUT_H264}")
    print(f"\n ReID Summary:")
    print(f"   CAM1 intruder IDs:  {sorted(cam1_ids_seen)}")
    print(f"   CAM2 intruder IDs:  {sorted(cam2_ids_seen)}")
    if cross_ids:
        print(f"   ✓ Cross-camera ReID SUCCESS — same person re-identified: IDs {sorted(cross_ids)}")
    else:
        print(f"   ✗ No cross-camera ReID match (IDs differ across cameras — appearance may have changed)")
        print(f"     Consider lowering appearance_thresh in tracker_config.yaml")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
