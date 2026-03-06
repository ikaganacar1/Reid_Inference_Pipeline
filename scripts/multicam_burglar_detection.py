"""
Multi-Camera Burglar Detection
Sequential cross-camera ReID: Burglar_comes_in (cam1) → Burglar_runs (cam2)

YOLOE-26x prompting strategy:
  CAM1 → text prompt "person": detect all people, track, save burglar reference crop
  CAM2 → visual prompt (burglar crop from CAM1): YOLOE finds only THAT specific person
         Segmentation masks are visualized as colored overlays.

Usage:
    python scripts/multicam_burglar_detection.py                    # YOLO11n
    python scripts/multicam_burglar_detection.py --detector yoloe   # YOLOE (text→visual)
    python scripts/multicam_burglar_detection.py --compare          # both + report
"""

import argparse
import json
import subprocess
import sys
import time
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import YOLOPersonDetector
from src.yoloe_detector import YOLOEPersonDetector
from src.reid_client import TritonReIDClient
from src.tracker import ReIDTracker
from src.utils.config_loader import load_all_configs


# ─── Paths ─────────────────────────────────────────────────────────────────────
CAM1_PATH  = Path("test_videos/Burglar_comes_in_id0.mp4")
CAM2_PATH  = Path("test_videos/Burglar_runs_id0.mp4")
CAM1_LABEL = "CAM 1 · Entrance"
CAM2_LABEL = "CAM 2 · Inner Room"
GAP_FRAMES = 60

# ─── Layout ────────────────────────────────────────────────────────────────────
PANE_W, PANE_H = 960, 540
DIVIDER = 4
OUT_W   = PANE_W * 2 + DIVIDER
OUT_H   = PANE_H + 100

# ─── Colors ────────────────────────────────────────────────────────────────────
GREEN  = (0,   200, 0  )
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0  )
GRAY   = (60,  60,  60 )
ORANGE = (0,   140, 255)
CYAN   = (230, 210, 0  )

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

TRACK_COLORS = [
    (255,  80,  80),
    ( 80, 200,  80),
    ( 80,  80, 255),
    (255, 200,  80),
    (255,  80, 255),
    ( 80, 255, 255),
]

MASK_ALPHA = 0.40   # segmentation overlay transparency


def track_color(tid: int):
    return TRACK_COLORS[int(tid) % len(TRACK_COLORS)]


# ─── Drawing ───────────────────────────────────────────────────────────────────

def draw_mask_overlay(frame: np.ndarray, mask: np.ndarray, color: tuple, alpha=MASK_ALPHA):
    """Blend a binary mask as a colored semi-transparent overlay."""
    overlay = frame.copy()
    overlay[mask > 0] = color
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # Contour for crisp border
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, 2)


def draw_track_box(frame, track, mask=None):
    if len(track) == 8:
        x1, y1, x2, y2, tid, conf, cls, _ = track
    else:
        x1, y1, x2, y2, tid, conf, cls = track
    x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)
    color = track_color(tid)

    # Segmentation mask overlay (if available)
    if mask is not None:
        draw_mask_overlay(frame, mask, color)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{tid}"
    (tw, th), _ = cv2.getTextSize(label, FONT_BOLD, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 3), FONT_BOLD, 0.55, BLACK, 1, cv2.LINE_AA)


def draw_hud(frame, cam_label, is_active, cam_w, mode_tag=""):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (cam_w, 36), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    color = WHITE if is_active else GRAY
    cv2.putText(frame, cam_label, (10, 25), FONT_BOLD, 0.68, color, 1, cv2.LINE_AA)
    if mode_tag and is_active:
        cv2.putText(frame, mode_tag, (cam_w - len(mode_tag)*9 - 10, 25),
                    FONT, 0.40, CYAN, 1, cv2.LINE_AA)


def make_timeline(out_w, bar_h, global_time_s, cam1_active,
                  cam1_total_s, cam2_total_s, detector_label):
    bar = np.zeros((bar_h, out_w, 3), dtype=np.uint8)
    bar[:] = (20, 20, 20)

    total_s    = cam1_total_s + cam2_total_s
    tl_x0, tl_x1 = 20, out_w - 20
    tl_len     = tl_x1 - tl_x0
    tl_y       = bar_h // 2 + 12
    cam1_end_x = tl_x0 + int(tl_len * cam1_total_s / total_s)

    cv2.line(bar, (tl_x0, tl_y), (tl_x1, tl_y), GRAY, 2)
    cv2.line(bar, (tl_x0, tl_y), (cam1_end_x, tl_y),
             GREEN if cam1_active else (40, 80, 40), 4)
    cv2.line(bar, (cam1_end_x, tl_y), (tl_x1, tl_y),
             ORANGE if not cam1_active else (40, 40, 80), 4)

    cv2.putText(bar, CAM1_LABEL, (tl_x0, tl_y - 10), FONT, 0.38, WHITE, 1)
    cv2.putText(bar, CAM2_LABEL, (cam1_end_x + 5, tl_y - 10), FONT, 0.38, WHITE, 1)

    frac = min(global_time_s / total_s, 1.0)
    ph_x = tl_x0 + int(tl_len * frac)
    cv2.circle(bar, (ph_x, tl_y), 6, WHITE, -1)
    mins, secs = int(global_time_s) // 60, int(global_time_s) % 60
    cv2.putText(bar, f"{mins:02d}:{secs:02d}", (ph_x + 8, tl_y + 4), FONT, 0.40, WHITE, 1)
    cv2.putText(bar, detector_label, (out_w - len(detector_label)*8 - 10, 20),
                FONT_BOLD, 0.48, ORANGE, 1, cv2.LINE_AA)
    return bar


def resize_pane(frame):
    return cv2.resize(frame, (PANE_W, PANE_H), interpolation=cv2.INTER_AREA)


def h264_encode(src: Path, dst: Path):
    """Re-encode mp4v → H.264 using miniconda ffmpeg."""
    ffmpeg = next(
        (p for p in ["/home/ika/miniconda3/bin/ffmpeg",
                     "/usr/local/bin/ffmpeg", "ffmpeg"]
         if Path(p).exists() or p == "ffmpeg"), "ffmpeg")
    subprocess.run([
        ffmpeg, "-y", "-i", str(src),
        "-c:v", "libopenh264", "-b:v", "4M", "-pix_fmt", "yuv420p",
        str(dst)
    ], check=True)
    src.unlink()


# ─── Reference frame selection ─────────────────────────────────────────────────

def best_reference(candidate_frames: list) -> tuple:
    """
    From a list of (frame, det_xyxy, conf) tuples collected during CAM1,
    return the (frame, bbox) with the largest high-confidence detection.
    This gives YOLOE's visual prompt the clearest view of the burglar.
    """
    best_score = -1
    best_frame, best_bbox = None, None
    for frame, bbox, conf in candidate_frames:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        score = conf * area
        if score > best_score:
            best_score = score
            best_frame = frame
            best_bbox  = bbox
    return best_frame, best_bbox


# ─── Core pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(detector, reid_client, tracker, output_path, detector_label,
                 use_visual_prompt=False) -> dict:
    """Process both videos. Returns metrics dict."""

    cap1 = cv2.VideoCapture(str(CAM1_PATH))
    cap2 = cv2.VideoCapture(str(CAM2_PATH))
    assert cap1.isOpened() and cap2.isOpened()

    fps    = cap1.get(cv2.CAP_PROP_FPS)
    total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    cam1_s = total1 / fps
    cam2_s = total2 / fps

    tmp_path = output_path.with_suffix('.raw.mp4')
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    out      = cv2.VideoWriter(str(tmp_path), fourcc, fps, (OUT_W, OUT_H))
    bar_h    = OUT_H - PANE_H

    cam1_ids: set = set()
    cam2_ids: set = set()
    cross_ids: set = set()
    det_times_cam1, det_times_cam2 = [], []
    det_counts_cam1, det_counts_cam2 = [], []

    # Reference candidates for visual prompt
    ref_candidates = []   # (frame, bbox_xyxy, conf)

    # Frozen cam2 preview pane (shown during cam1 phase)
    ret, cam2_preview_raw = cap2.read()
    cam2_preview = resize_pane(cam2_preview_raw) if ret else np.zeros((PANE_H, PANE_W, 3), np.uint8)
    cam2_preview = (cam2_preview.astype(np.float32) * 0.45).astype(np.uint8)
    draw_hud(cam2_preview, CAM2_LABEL, False, PANE_W)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    t0 = time.time()

    # ── Phase 1: CAM1  (text prompt) ───────────────────────────────────────
    print(f"\n[{detector_label}] Phase 1: {CAM1_LABEL}  (text prompt: 'person')")
    frame_idx = 0
    last_pane1 = None
    is_yoloe = isinstance(detector, YOLOEPersonDetector)

    while cap1.isOpened():
        ret, frame = cap1.read()
        if not ret:
            break

        t_det = time.time()
        result = detector.detect(frame)
        det_times_cam1.append(time.time() - t_det)

        # Handle both detector interfaces (YOLO returns 2 values, YOLOE returns 3)
        if is_yoloe:
            detections, crops, masks = result
        else:
            detections, crops = result
            masks = [None] * len(detections)

        det_counts_cam1.append(len(detections))
        embeddings = reid_client.infer(crops) if len(crops) > 0 else np.empty((0, 0))
        tracks     = tracker.update(detections, frame, embeddings)

        track_ids = [int(t[4]) for t in tracks] if len(tracks) > 0 else []
        cam1_ids.update(track_ids)

        # Collect reference candidates: save frame+bbox for each confirmed detection
        if use_visual_prompt and len(detections) > 0:
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, _ = det
                ref_candidates.append((frame.copy(), np.array([x1, y1, x2, y2]), float(conf)))

        # Visualize
        vis = frame.copy()
        for i, t in enumerate(tracks):
            tid = int(t[4])
            # Match track to detection index for mask lookup
            mask = None
            if masks and i < len(masks):
                mask = masks[i]
            draw_track_box(vis, t, mask=mask)

        pane1 = resize_pane(vis)
        draw_hud(pane1, CAM1_LABEL, True, PANE_W,
                 mode_tag="TEXT PROMPT" if is_yoloe else "")
        last_pane1 = pane1.copy()

        divider  = np.zeros((PANE_H, DIVIDER, 3), np.uint8)
        timeline = make_timeline(OUT_W, bar_h, frame_idx / fps, True,
                                 cam1_s, cam2_s, detector_label)
        out.write(np.vstack([np.hstack([pane1, divider, cam2_preview]), timeline]))

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}/{total1}  dets={len(detections)}  tracks={len(tracks)}")

    cap1.release()
    print(f"  CAM1 done. IDs seen: {sorted(cam1_ids)}")

    # ── Switch YOLOE to visual prompt mode for CAM2 ────────────────────────
    if use_visual_prompt and is_yoloe and ref_candidates:
        ref_frame, ref_bbox = best_reference(ref_candidates)
        print(f"\n[{detector_label}] Switching to VISUAL PROMPT mode")
        print(f"  Reference bbox: {ref_bbox.astype(int).tolist()}")
        detector.set_visual_prompt(ref_frame, ref_bbox.reshape(1, 4))
    elif is_yoloe:
        print(f"\n[{detector_label}] No reference collected — keeping text prompt for CAM2")

    # ── Gap ─────────────────────────────────────────────────────────────────
    dummy = np.zeros((PANE_H, PANE_W, 3), np.uint8)
    for _ in range(GAP_FRAMES):
        tracker.update(np.empty((0, 6)), dummy, np.empty((0, 0)))

    frozen1 = (last_pane1.astype(np.float32) * 0.45).astype(np.uint8) if last_pane1 is not None \
        else np.zeros((PANE_H, PANE_W, 3), np.uint8)
    draw_hud(frozen1, CAM1_LABEL, False, PANE_W)

    # ── Phase 2: CAM2  (visual prompt if YOLOE, else text/class) ──────────
    mode_tag = "VISUAL PROMPT" if (use_visual_prompt and is_yoloe) else \
               ("TEXT PROMPT" if is_yoloe else "")
    print(f"\n[{detector_label}] Phase 2: {CAM2_LABEL}  ({mode_tag or 'class filter'})")
    cam2_idx = 0

    while cap2.isOpened():
        ret, frame = cap2.read()
        if not ret:
            break

        t_det = time.time()
        result = detector.detect(frame)
        det_times_cam2.append(time.time() - t_det)

        if is_yoloe:
            detections, crops, masks = result
        else:
            detections, crops = result
            masks = [None] * len(detections)

        det_counts_cam2.append(len(detections))
        embeddings = reid_client.infer(crops) if len(crops) > 0 else np.empty((0, 0))
        tracks     = tracker.update(detections, frame, embeddings)

        track_ids = [int(t[4]) for t in tracks] if len(tracks) > 0 else []
        cam2_ids.update(track_ids)
        cross_ids = cam1_ids & cam2_ids

        vis = frame.copy()
        for i, t in enumerate(tracks):
            mask = masks[i] if masks and i < len(masks) else None
            draw_track_box(vis, t, mask=mask)

        pane2 = resize_pane(vis)
        draw_hud(pane2, CAM2_LABEL, True, PANE_W, mode_tag=mode_tag)

        global_t = cam1_s + cam2_idx / fps
        divider  = np.zeros((PANE_H, DIVIDER, 3), np.uint8)
        timeline = make_timeline(OUT_W, bar_h, global_t, False,
                                 cam1_s, cam2_s, detector_label)
        out.write(np.vstack([np.hstack([frozen1, divider, pane2]), timeline]))

        cam2_idx += 1
        if cam2_idx % 100 == 0:
            print(f"  frame {cam2_idx}/{total2}  dets={len(detections)}  "
                  f"tracks={len(tracks)}  cross={sorted(cross_ids)}")

    cap2.release()
    out.release()
    total_time = time.time() - t0

    print(f"\n[{detector_label}] Re-encoding to H.264…")
    h264_encode(tmp_path, output_path)

    metrics = {
        "detector":          detector_label,
        "total_time_s":      round(total_time, 2),
        "overall_fps":       round((total1 + total2) / total_time, 2),
        "cam1_avg_det_ms":   round(np.mean(det_times_cam1) * 1000, 2) if det_times_cam1 else 0,
        "cam2_avg_det_ms":   round(np.mean(det_times_cam2) * 1000, 2) if det_times_cam2 else 0,
        "cam1_total_dets":   int(sum(det_counts_cam1)),
        "cam2_total_dets":   int(sum(det_counts_cam2)),
        "cam1_ids":          sorted(cam1_ids),
        "cam2_ids":          sorted(cam2_ids),
        "cross_camera_reid": sorted(cross_ids),
        "reid_success":      len(cross_ids) > 0,
        "output":            str(output_path),
    }
    print(f"  Done in {total_time:.1f}s  |  Cross-camera ReID: "
          f"{'SUCCESS ✓' if metrics['reid_success'] else 'FAILED ✗'}")
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["yolo11n", "yoloe"], default="yolo11n")
    parser.add_argument("--compare", action="store_true",
                        help="Run both detectors and print comparison report")
    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True)
    configs = load_all_configs()

    detectors_to_run = ["yolo11n", "yoloe"] if args.compare else [args.detector]
    all_metrics = []

    for det_name in detectors_to_run:
        print(f"\n{'='*65}")
        print(f" Multi-Camera Burglar Detection  [{det_name.upper()}]")
        print(f"{'='*65}")

        if det_name == "yolo11n":
            detector          = YOLOPersonDetector(configs['yolo'])
            label             = "YOLO11n"
            out_path          = Path("outputs/multicam_burglar_yolo11n.mp4")
            use_visual_prompt = False
        else:
            detector          = YOLOEPersonDetector(configs['yoloe'])
            label             = "YOLOE-26x"
            out_path          = Path("outputs/multicam_burglar_yoloe26.mp4")
            use_visual_prompt = True   # CAM1=text, CAM2=visual prompt

        reid_client = TritonReIDClient(configs['reid'])
        tracker     = ReIDTracker(configs['tracker'])

        metrics = run_pipeline(detector, reid_client, tracker,
                               out_path, label, use_visual_prompt)
        all_metrics.append(metrics)

        json_path = out_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics: {json_path}")

    # ── Comparison report ──────────────────────────────────────────────────
    if args.compare and len(all_metrics) == 2:
        a, b = all_metrics
        print(f"\n{'='*65}")
        print(f" COMPARISON REPORT  (YOLO11n  vs  YOLOE-26x)")
        print(f"{'='*65}")
        rows = [
            ("Overall FPS",             f"{a['overall_fps']:.2f}",    f"{b['overall_fps']:.2f}"),
            ("CAM1 avg detection (ms)", f"{a['cam1_avg_det_ms']:.2f}", f"{b['cam1_avg_det_ms']:.2f}"),
            ("CAM2 avg detection (ms)", f"{a['cam2_avg_det_ms']:.2f}", f"{b['cam2_avg_det_ms']:.2f}"),
            ("CAM1 total detections",   str(a['cam1_total_dets']),     str(b['cam1_total_dets'])),
            ("CAM2 total detections",   str(a['cam2_total_dets']),     str(b['cam2_total_dets'])),
            ("Cross-camera ReID",       str(a['reid_success']),        str(b['reid_success'])),
            ("IDs in CAM1",             str(a['cam1_ids']),            str(b['cam1_ids'])),
            ("IDs in CAM2",             str(a['cam2_ids']),            str(b['cam2_ids'])),
            ("Total time (s)",          f"{a['total_time_s']:.2f}",    f"{b['total_time_s']:.2f}"),
        ]
        print(f"  {'Metric':<33} {'YOLO11n':>10}   {'YOLOE-26x':>10}")
        print(f"  {'-'*59}")
        for name, va, vb in rows:
            print(f"  {name:<33} {va:>10}   {vb:>10}")
        print(f"{'='*65}")
        print()
        print("  YOLOE-26x notes:")
        print("  · CAM1: text prompt 'person' via CLIP embedding")
        print("  · CAM2: visual prompt from burglar reference crop saved in CAM1")
        print("          → YOLOE searches for THAT specific person, not any person")
        print("  · Segmentation masks drawn as colored overlays on detected persons")
        print(f"{'='*65}")

        report = {"yolo11n": a, "yoloe26": b}
        rp = Path("outputs/detector_comparison.json")
        with open(rp, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  Full report: {rp}")
        print(f"  Videos: {a['output']}  |  {b['output']}")


if __name__ == "__main__":
    main()
