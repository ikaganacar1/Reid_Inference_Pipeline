"""
Multi-Camera Burglar Detection
Sequential cross-camera ReID: Burglar_comes_in (cam1) → Burglar_runs (cam2)

YOLOE-26x detects with two text prompts: ["person", "intruder"]
CLIP classifies each box as one or the other.
Swin Base ReID (Triton) handles cross-camera re-identification.

Usage:
    python scripts/multicam_burglar_detection.py                    # YOLO11n
    python scripts/multicam_burglar_detection.py --detector yoloe   # YOLOE-26x
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

PANE_W, PANE_H = 960, 540
DIVIDER = 4
OUT_W   = PANE_W * 2 + DIVIDER
OUT_H   = PANE_H + 100

# Colors
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0  )
GRAY   = (60,  60,  60 )
GREEN  = (0,   200, 0  )
ORANGE = (0,   140, 255)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# Class colors: index matches text_prompts order
# cls=0 → "person"   → blue-green
# cls=1 → "intruder" → red
CLASS_COLORS = {
    0: (200, 180, 60),   # person   — teal
    1: (50,  50,  220),  # intruder — red
}
MASK_ALPHA = 0.35


def class_color(cls_idx: int) -> tuple:
    return CLASS_COLORS.get(int(cls_idx), (180, 180, 180))


# ─── Drawing ───────────────────────────────────────────────────────────────────

def draw_mask(frame, mask, color):
    overlay = frame.copy()
    overlay[mask > 0] = color
    cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0, frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, 2)


def draw_track(frame, track, class_names, mask=None):
    if len(track) == 8:
        x1, y1, x2, y2, tid, conf, cls, _ = track
    else:
        x1, y1, x2, y2, tid, conf, cls = track
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    tid, cls = int(tid), int(cls)

    color = class_color(cls)
    label = f"{class_names[cls] if cls < len(class_names) else 'person'}  ID:{tid}"

    if mask is not None:
        draw_mask(frame, mask, color)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, FONT_BOLD, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 3), FONT_BOLD, 0.55, BLACK, 1, cv2.LINE_AA)


def draw_hud(frame, label, is_active, cam_w):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (cam_w, 36), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, label, (10, 25), FONT_BOLD, 0.68,
                WHITE if is_active else GRAY, 1, cv2.LINE_AA)


def make_timeline(out_w, bar_h, t, cam1_active, cam1_s, cam2_s, det_label):
    bar = np.zeros((bar_h, out_w, 3), np.uint8)
    bar[:] = (20, 20, 20)
    total  = cam1_s + cam2_s
    x0, x1 = 20, out_w - 20
    tl_len = x1 - x0
    tl_y   = bar_h // 2 + 12
    c1x    = x0 + int(tl_len * cam1_s / total)

    cv2.line(bar, (x0, tl_y), (x1, tl_y), GRAY, 2)
    cv2.line(bar, (x0, tl_y), (c1x, tl_y), GREEN if cam1_active else (40, 80, 40), 4)
    cv2.line(bar, (c1x, tl_y), (x1, tl_y), ORANGE if not cam1_active else (40, 40, 80), 4)
    cv2.putText(bar, CAM1_LABEL, (x0, tl_y - 10), FONT, 0.38, WHITE, 1)
    cv2.putText(bar, CAM2_LABEL, (c1x + 5, tl_y - 10), FONT, 0.38, WHITE, 1)

    ph = x0 + int(tl_len * min(t / total, 1.0))
    cv2.circle(bar, (ph, tl_y), 6, WHITE, -1)
    cv2.putText(bar, f"{int(t)//60:02d}:{int(t)%60:02d}", (ph + 8, tl_y + 4), FONT, 0.40, WHITE, 1)
    cv2.putText(bar, det_label, (out_w - len(det_label) * 8 - 10, 20),
                FONT_BOLD, 0.48, ORANGE, 1, cv2.LINE_AA)
    return bar


def resize_pane(f):
    return cv2.resize(f, (PANE_W, PANE_H), interpolation=cv2.INTER_AREA)


def h264_encode(src: Path, dst: Path):
    ffmpeg = next((p for p in ["/home/ika/miniconda3/bin/ffmpeg",
                                "/usr/local/bin/ffmpeg", "ffmpeg"]
                   if Path(p).exists() or p == "ffmpeg"), "ffmpeg")
    subprocess.run([ffmpeg, "-y", "-i", str(src),
                    "-c:v", "libopenh264", "-b:v", "4M", "-pix_fmt", "yuv420p",
                    str(dst)], check=True)
    src.unlink()


# ─── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(detector, reid_client, tracker, output_path, det_label) -> dict:
    is_yoloe    = isinstance(detector, YOLOEPersonDetector)
    class_names = detector.text_prompts if is_yoloe else ["person"]

    cap1 = cv2.VideoCapture(str(CAM1_PATH))
    cap2 = cv2.VideoCapture(str(CAM2_PATH))

    fps    = cap1.get(cv2.CAP_PROP_FPS)
    total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    cam1_s = total1 / fps
    cam2_s = total2 / fps

    tmp    = output_path.with_suffix('.raw.mp4')
    out    = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (OUT_W, OUT_H))
    bar_h  = OUT_H - PANE_H

    cam1_ids: set = set()
    cam2_ids: set = set()
    cross_ids: set = set()
    det_times1, det_times2 = [], []
    det_counts1, det_counts2 = [], []

    # Dimmed cam2 preview shown during phase 1
    ret, prev = cap2.read()
    cam2_prev = (resize_pane(prev).astype(np.float32) * 0.45).astype(np.uint8) if ret \
        else np.zeros((PANE_H, PANE_W, 3), np.uint8)
    draw_hud(cam2_prev, CAM2_LABEL, False, PANE_W)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    t0 = time.time()

    # ── Phase 1: CAM1 ──────────────────────────────────────────────────────
    print(f"\n[{det_label}] Phase 1: {CAM1_LABEL}")
    fidx, last1 = 0, None

    while cap1.isOpened():
        ret, frame = cap1.read()
        if not ret:
            break

        t_d = time.time()
        if is_yoloe:
            dets, crops, masks = detector.detect(frame)
        else:
            dets, crops = detector.detect(frame)
            masks = [None] * len(dets)
        det_times1.append(time.time() - t_d)
        det_counts1.append(len(dets))

        embs   = reid_client.infer(crops) if len(crops) > 0 else np.empty((0, 0))
        tracks = tracker.update(dets, frame, embs)
        cam1_ids.update(int(t[4]) for t in tracks)

        vis = frame.copy()
        for i, t in enumerate(tracks):
            draw_track(vis, t, class_names, mask=masks[i] if i < len(masks) else None)

        pane1 = resize_pane(vis)
        draw_hud(pane1, CAM1_LABEL, True, PANE_W)
        last1 = pane1.copy()

        div      = np.zeros((PANE_H, DIVIDER, 3), np.uint8)
        timeline = make_timeline(OUT_W, bar_h, fidx / fps, True, cam1_s, cam2_s, det_label)
        out.write(np.vstack([np.hstack([pane1, div, cam2_prev]), timeline]))
        fidx += 1

        if fidx % 100 == 0:
            print(f"  frame {fidx}/{total1}  dets={len(dets)}  tracks={len(tracks)}")

    cap1.release()
    print(f"  CAM1 done. IDs: {sorted(cam1_ids)}")

    # Gap
    dummy = np.zeros((PANE_H, PANE_W, 3), np.uint8)
    for _ in range(GAP_FRAMES):
        tracker.update(np.empty((0, 6)), dummy, np.empty((0, 0)))

    frozen1 = (last1.astype(np.float32) * 0.45).astype(np.uint8) if last1 is not None \
        else np.zeros((PANE_H, PANE_W, 3), np.uint8)
    draw_hud(frozen1, CAM1_LABEL, False, PANE_W)

    # ── Phase 2: CAM2 ──────────────────────────────────────────────────────
    print(f"\n[{det_label}] Phase 2: {CAM2_LABEL}")
    cidx = 0

    while cap2.isOpened():
        ret, frame = cap2.read()
        if not ret:
            break

        t_d = time.time()
        if is_yoloe:
            dets, crops, masks = detector.detect(frame)
        else:
            dets, crops = detector.detect(frame)
            masks = [None] * len(dets)
        det_times2.append(time.time() - t_d)
        det_counts2.append(len(dets))

        embs   = reid_client.infer(crops) if len(crops) > 0 else np.empty((0, 0))
        tracks = tracker.update(dets, frame, embs)
        cam2_ids.update(int(t[4]) for t in tracks)
        cross_ids = cam1_ids & cam2_ids

        vis = frame.copy()
        for i, t in enumerate(tracks):
            draw_track(vis, t, class_names, mask=masks[i] if i < len(masks) else None)

        pane2 = resize_pane(vis)
        draw_hud(pane2, CAM2_LABEL, True, PANE_W)

        gt       = cam1_s + cidx / fps
        div      = np.zeros((PANE_H, DIVIDER, 3), np.uint8)
        timeline = make_timeline(OUT_W, bar_h, gt, False, cam1_s, cam2_s, det_label)
        out.write(np.vstack([np.hstack([frozen1, div, pane2]), timeline]))
        cidx += 1

        if cidx % 100 == 0:
            print(f"  frame {cidx}/{total2}  dets={len(dets)}  "
                  f"tracks={len(tracks)}  cross={sorted(cross_ids)}")

    cap2.release()
    out.release()
    total_t = time.time() - t0

    print(f"\n[{det_label}] Re-encoding to H.264…")
    h264_encode(tmp, output_path)

    m = {
        "detector":          det_label,
        "total_time_s":      round(total_t, 2),
        "overall_fps":       round((total1 + total2) / total_t, 2),
        "cam1_avg_det_ms":   round(np.mean(det_times1) * 1000, 2) if det_times1 else 0,
        "cam2_avg_det_ms":   round(np.mean(det_times2) * 1000, 2) if det_times2 else 0,
        "cam1_total_dets":   int(sum(det_counts1)),
        "cam2_total_dets":   int(sum(det_counts2)),
        "cam1_ids":          sorted(cam1_ids),
        "cam2_ids":          sorted(cam2_ids),
        "cross_camera_reid": sorted(cross_ids),
        "reid_success":      len(cross_ids) > 0,
        "output":            str(output_path),
    }
    print(f"  Done {total_t:.1f}s  |  ReID: {'SUCCESS ✓' if m['reid_success'] else 'FAILED ✗'}")
    return m


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["yolo11n", "yoloe"], default="yolo11n")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True)
    configs = load_all_configs()
    to_run  = ["yolo11n", "yoloe"] if args.compare else [args.detector]
    results = []

    for name in to_run:
        print(f"\n{'='*60}")
        print(f" [{name.upper()}]  Multi-Camera Burglar Detection")
        print(f"{'='*60}")

        if name == "yolo11n":
            det      = YOLOPersonDetector(configs['yolo'])
            label    = "YOLO11n"
            out_path = Path("outputs/multicam_burglar_yolo11n.mp4")
        else:
            det      = YOLOEPersonDetector(configs['yoloe'])
            label    = f"YOLOE-26x  {configs['yoloe']['model']['text_prompts']}"
            out_path = Path("outputs/multicam_burglar_yoloe26.mp4")

        reid    = TritonReIDClient(configs['reid'])
        tracker = ReIDTracker(configs['tracker'])
        m       = run_pipeline(det, reid, tracker, out_path, label)
        results.append(m)

        jp = out_path.with_suffix('.json')
        json.dump(m, open(jp, 'w'), indent=2)
        print(f"  Metrics: {jp}")

    if args.compare and len(results) == 2:
        a, b = results
        print(f"\n{'='*60}")
        print(f"  {'Metric':<32} {'YOLO11n':>9}  {'YOLOE-26x':>9}")
        print(f"  {'-'*55}")
        rows = [
            ("Overall FPS",             f"{a['overall_fps']:.1f}",    f"{b['overall_fps']:.1f}"),
            ("Avg detection ms (CAM1)", f"{a['cam1_avg_det_ms']:.1f}", f"{b['cam1_avg_det_ms']:.1f}"),
            ("Avg detection ms (CAM2)", f"{a['cam2_avg_det_ms']:.1f}", f"{b['cam2_avg_det_ms']:.1f}"),
            ("CAM1 total detections",   str(a['cam1_total_dets']),     str(b['cam1_total_dets'])),
            ("CAM2 total detections",   str(a['cam2_total_dets']),     str(b['cam2_total_dets'])),
            ("Cross-camera ReID",       str(a['reid_success']),        str(b['reid_success'])),
            ("Total time (s)",          f"{a['total_time_s']:.1f}",    f"{b['total_time_s']:.1f}"),
        ]
        for n, va, vb in rows:
            print(f"  {n:<32} {va:>9}  {vb:>9}")
        print(f"{'='*60}")
        json.dump({"yolo11n": a, "yoloe26": b},
                  open("outputs/detector_comparison.json", 'w'), indent=2)


if __name__ == "__main__":
    main()
