"""
Single-video inference with YOLOE-26x + Swin Base ReID + BoTSORT

Prompts are configured in configs/yoloe_config.yaml → model.text_prompts

Usage:
    python scripts/infer_video.py --video path/to/video.mp4
    python scripts/infer_video.py --video path/to/video.mp4 --output out.mp4
    python scripts/infer_video.py --video path/to/video.mp4 --no-reid
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import YOLOPersonDetector
from src.yoloe_detector import YOLOEPersonDetector
from src.reid_client import TritonReIDClient
from src.tracker import ReIDTracker
from src.utils.config_loader import load_all_configs


# ─── Colors ────────────────────────────────────────────────────────────────────
BLACK     = (0,   0,   0  )
WHITE     = (255, 255, 255)
GRAY      = (80,  80,  80 )
FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
MASK_ALPHA = 0.35

# Color per class index — extends automatically for more prompts
_PALETTE = [
    (200, 180,  60),   # cls 0
    ( 50,  50, 220),   # cls 1
    ( 50, 200,  50),   # cls 2
    (220, 100,  50),   # cls 3
    (200,  50, 200),   # cls 4
]

def cls_color(cls_idx: int) -> tuple:
    return _PALETTE[int(cls_idx) % len(_PALETTE)]


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

    color = cls_color(cls)
    name  = class_names[cls] if cls < len(class_names) else f"cls{cls}"
    label = f"{name}  ID:{tid}"

    if mask is not None:
        draw_mask(frame, mask, color)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, FONT_BOLD, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 3), FONT_BOLD, 0.55, BLACK, 1, cv2.LINE_AA)


def draw_hud(frame, prompts, frame_idx, fps, n_tracks):
    """Top-left overlay: prompt legend + frame stats."""
    line_h = 22
    n      = len(prompts) + 1          # legend rows + stats row
    pad    = 8
    box_h  = n * line_h + pad * 2
    box_w  = 220

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, name in enumerate(prompts):
        color = cls_color(i)
        y = pad + i * line_h + line_h - 4
        cv2.rectangle(frame, (pad, y - 12), (pad + 14, y + 2), color, -1)
        cv2.putText(frame, name, (pad + 20, y), FONT_BOLD, 0.50, WHITE, 1, cv2.LINE_AA)

    # Stats row
    y_stats = pad + len(prompts) * line_h + line_h - 4
    stats = f"frame {frame_idx}   tracks {n_tracks}"
    cv2.putText(frame, stats, (pad, y_stats), FONT, 0.40, GRAY, 1, cv2.LINE_AA)


def h264_encode(src: Path, dst: Path):
    ffmpeg = next((p for p in ["/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg", "ffmpeg"]
                   if Path(p).exists() or p == "ffmpeg"), "ffmpeg")
    for codec in ["h264_nvenc", "libx264", "libopenh264"]:
        try:
            subprocess.run([ffmpeg, "-y", "-i", str(src),
                            "-c:v", codec, "-b:v", "6M", "-pix_fmt", "yuv420p",
                            str(dst)], check=True, capture_output=True)
            src.unlink()
            return
        except subprocess.CalledProcessError:
            continue
    src.rename(dst)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Single-video inference")
    parser.add_argument("--video",   required=True, help="Input video path")
    parser.add_argument("--detector", choices=["yolo11n", "yoloe"], default="yolo11n")
    parser.add_argument("--output",  default=None,  help="Output video path (default: outputs/<name>_<det>.mp4)")
    parser.add_argument("--no-reid", action="store_true", help="Skip ReID — tracking by IoU only")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for quick testing")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        sys.exit(1)

    configs = load_all_configs()
    is_yoloe = args.detector == "yoloe"
    prompts = configs['yoloe']['model']['text_prompts'] if is_yoloe else ["person"]

    Path("outputs").mkdir(exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        suffix = "yoloe" if is_yoloe else "yolo11n"
        out_path = Path("outputs") / f"{video_path.stem}_{suffix}.mp4"
    tmp_path = out_path.with_suffix('.raw.mp4')

    print("=" * 55)
    print(f" {args.detector.upper()} Inference")
    print(f"  Video  : {video_path}")
    print(f"  Output : {out_path}")
    print(f"  Prompts: {prompts}")
    print(f"  ReID   : {'disabled' if args.no_reid else 'Swin Base via Triton'}")
    print("=" * 55)

    # ── Init ──────────────────────────────────────────────────────────────
    print("\nLoading models...")
    if is_yoloe:
        detector = YOLOEPersonDetector(configs['yoloe'])
    else:
        detector = YOLOPersonDetector(configs['yolo'])

    reid_client = None
    if not args.no_reid:
        reid_client = TritonReIDClient(configs['reid'])

    tracker = ReIDTracker(configs['tracker'])

    # ── Open video ────────────────────────────────────────────────────────
    cap     = cv2.VideoCapture(str(video_path))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)

    print(f"\n  {width}×{height} @ {fps:.2f} fps  ({total} frames)\n")

    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))

    # ── Process ───────────────────────────────────────────────────────────
    det_times, total_dets = [], 0
    t0 = time.time()

    for fidx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        t_d = time.time()
        if is_yoloe:
            dets, crops, masks = detector.detect(frame)
        else:
            dets, crops = detector.detect(frame)
            masks = [None] * len(dets)
        det_times.append(time.time() - t_d)
        total_dets += len(dets)

        if reid_client and len(crops) > 0:
            embs = reid_client.infer(crops)
        else:
            embs = np.empty((0, 0))

        tracks = tracker.update(dets, frame, embs)

        # Draw
        for i, t in enumerate(tracks):
            mask = masks[i] if i < len(masks) else None
            draw_track(frame, t, prompts, mask=mask)

        draw_hud(frame, prompts, fidx + 1, fps, len(tracks))
        writer.write(frame)

        if (fidx + 1) % 100 == 0:
            elapsed = time.time() - t0
            fps_so_far = (fidx + 1) / elapsed
            print(f"  frame {fidx+1}/{total}  |  dets={len(dets)}  tracks={len(tracks)}"
                  f"  |  {fps_so_far:.1f} fps")

    cap.release()
    writer.release()
    elapsed = time.time() - t0

    # ── Encode ────────────────────────────────────────────────────────────
    print(f"\nRe-encoding to H.264…")
    h264_encode(tmp_path, out_path)

    print(f"\n{'='*55}")
    print(f" Done in {elapsed:.1f}s  ({(fidx+1)/elapsed:.1f} fps)")
    print(f" Total detections : {total_dets}")
    print(f" Avg det time     : {np.mean(det_times)*1000:.1f} ms")
    print(f" Output           : {out_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
