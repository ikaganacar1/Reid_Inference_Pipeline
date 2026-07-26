# Realtime Multi-Orin Pipeline

The realtime subsystem is separate from the single-video runner. The current
two-device deployment is:

```text
camera cam1 -> prime-jetson worker (YOLO) --\
                                                   -> prime-jetson
camera cam2 -> worker-jetson worker (YOLO) --/         ONNX ReID on CUDA
                                                    one BoTSORT per camera
                                                    global ID gallery
                                                    MP4 recording + dashboard
```

It scales to more workers and cameras as long as every physical stream has a
globally unique `camera_id`.

## What Runs Where

Each camera worker runs person detection and sends a binary WebSocket packet
containing the compressed frame, `[x1, y1, x2, y2, confidence, class]`
detections, and compressed person crops. The prime batches crops arriving in a
short time window, runs the generalized Swin ONNX model on the CUDA execution
provider, tracks each camera independently, then assigns persistent global IDs.

The default path does **not** use Triton. `configs/reid_config.yaml` selects
`onnxruntime_direct`, and the generalized model requires RGB input normalized
with mean/std `[0.5, 0.5, 0.5]`.

## Device Profiles

The launchers deliberately use different profiles to prevent both devices from
publishing the same camera ID:

| Device | Realtime config | Camera ID | Detector config |
|---|---|---|---|
| `prime-jetson` | `configs/realtime_config.yaml` | `cam1` | `configs/yolo_config.yaml` |
| `worker-jetson` | `configs/realtime_config.worker.yaml` | `cam2` | `configs/yolo_config.worker.yaml` |

Both detector configs currently use `yolo26m.pt`, CUDA, FP16, 640-pixel input,
person class only, and confidence `0.50`. Deployment preflight requires the
staged checkpoint to match `deploy/model_manifest.yaml`; unattended startup
does not rely on an internet download or a device-specific TensorRT plan.

## Installation Boundaries

Cloning the repository and installing Python requirements does not install
JetPack, CUDA, TensorRT, GPU drivers, or a compatible Jetson PyTorch wheel.
Those platform components must already work on each Orin.

Prime requirements:

- compatible JetPack/CUDA and GPU-enabled ONNX Runtime for Jetson
- Python packages from `requirements_prime_jetson.txt`
- the generalized ONNX model at the configured path or `REID_MODEL_PATH`
- BoxMOT 16.x, PyTorch, OpenCV, Pillow, and aiohttp

Worker requirements:

- JetPack-compatible PyTorch and Ultralytics for the default `yolo26m.pt` path
- packages from `requirements_worker_jetson.txt`
- OpenCV camera support supplied by JetPack/L4T

Before realtime startup, verify the prime environment can report
`CUDAExecutionProvider`. The prime launcher supports `.venv-jetson` by default
and prepares the ONNX Runtime/CUDA library paths for the selected environment.
See [JETSON_DEPLOYMENT.md](JETSON_DEPLOYMENT.md) for the role installers,
fail-fast preflight, model staging, and systemd services.

## Start The System

On the prime Jetson, start the centralized server and its local camera control:

```bash
cd ~/Desktop/Reid_Inference_Pipeline
scripts/start_prime_dashboard.sh
scripts/start_worker_control.sh prime
```

On the worker Jetson, use the worker profile:

```bash
cd ~/Desktop/Reid_Inference_Pipeline
scripts/start_worker_control.sh worker
```

Open the dashboard from a LAN/Tailscale client:

```text
http://<prime-lan-ip>:8765/
```

The dashboard can start, stop, or scan/restart either configured worker node.
The same operations are available from each worker control API on port `8787`.

Stop commands:

```bash
scripts/stop_worker_control.sh
scripts/stop_prime_dashboard.sh
```

The control service automatically scans and starts its configured camera, then
retries if the camera or worker process is unavailable. The launch scripts use
`nohup` for manual operation and load
`~/.config/reid-pipeline/jetson.env` when it exists. For unattended operation,
install the prepared systemd user services described in
`docs/JETSON_DEPLOYMENT.md`.

## Camera Discovery And Reconnect

The scanner prefers `/dev/v4l/by-path/*video-index0`. The stable by-path symlink
is passed to OpenCV directly, so unplugging and reconnecting a camera in the same
USB port does not depend on `/dev/videoN` numbering. A running worker retries a
failed capture every `worker.reconnect_seconds`.

If no camera is present, the service reports a failure and retries. After a
camera is connected, auto-start discovers it; moving it to a different USB port
also triggers a worker restart with the new stable path. Discovery count must
match the configured global camera IDs, so an extra source cannot silently gain
a colliding fallback name. The prime permits only one active WebSocket owner for
each `camera_id`; duplicate IDs are rejected rather than merged into one tracker.

## Identity Rules

- BoTSORT IDs are local, short-lived implementation details and are not shown.
- The dashboard and videos show `ID:<global_id>` only.
- New identities require five good observations before becoming visible.
- Frame-edge fragments cannot seed a global identity. Existing tracks can
  still hold their ID while entering or leaving a frame.
- The gallery retains bounded per-camera-track appearance prototypes. A long
  observation in one camera therefore cannot erase earlier views.
- Every established visible ID is reserved before new-track matching, preventing
  output ordering from letting one track steal another person's ID.
- A configured overlap pair may use one ID simultaneously after a confirmed
  appearance match. Other camera pairs still reserve an active ID.
- A configured adjacent pair permits a fast sequential handoff after its short
  travel exclusion, but never permits simultaneous use.
- The in-memory gallery persists across all cameras for the lifetime of the
  prime process. Camera-local tracker state is reset after frame rollback,
  worker restart, or a long capture gap.

Map the physical topology before deployment. Overlap and adjacency are not the
same relation:

```yaml
prime:
  overlapping_camera_pairs:
    - ["cam1", "cam2"]  # one person may be visible in both
  adjacent_camera_pairs:
    - ["cam2", "cam3"]  # fast handoff, no simultaneous presence
```

Do not set `allow_all_camera_overlap` merely to avoid rejected matches. It
removes useful impossibility constraints and makes similar clothing harder to
disambiguate. `cross_camera_exclusion_seconds` applies to ordinary disjoint
pairs; adjacent pairs use `adjacent_camera_exclusion_seconds`.

## Dashboard And Metrics

The dashboard uses a dynamic grid: one view fills the area, two views use two
columns, and three or four views use a 2x2 layout. Cards disappear after
`prime.camera_offline_seconds`. Metrics include camera FPS, detections, tracks,
ReID time, tracking time, processing time, queue occupancy, dropped packets,
viewer count, and gallery size.

Viewer sends have a timeout. A slow browser is disconnected instead of applying
backpressure to inference and recording.

## Recording

Processed footage is written to a new session directory on every prime start:

```text
outputs/realtime/recordings/<YYYYMMDD_HHMMSS>/cam1_000001_processed.mp4
outputs/realtime/recordings/<YYYYMMDD_HHMMSS>/cam2_000001_processed.mp4
```

The current writer uses OpenCV `mp4v` at 10 FPS and finalizes a new segment every
`prime.recording_segment_seconds` (15 minutes by default). Segmentation limits
damage from an abrupt shutdown. Automatic deletion is intentionally disabled;
the site still needs a disk retention/archive policy. Hardware encoding is a
separate optimization. The prime checks storage every 10 seconds and pauses all
writers below the default 5 GiB free-space reserve. It resumes into new segments
after space is freed, while inference and live viewing remain active.

## Offline Reproduction

The synchronized factory-footage path uses the same detector, ReID client,
BoTSORT wrapper, and global identity assigner:

```bash
scripts/start_reid_debug.sh
```

Useful overrides:

```bash
START_FRAME=900 MAX_FRAMES=200 STRIDE=2 \
OUTPUT_DIR=experiments/reid_audit scripts/start_reid_debug.sh
```

For mixed-FPS files, replay is synchronized by wall-clock time rather than raw
frame index. A sampled run should scale BoTSORT's frame buffer to preserve its
wall-clock retention. For example, 2 FPS with a three-second retention uses:

```bash
STRIDE=10 BOTSORT_TRACK_BUFFER=6 \
OVERLAPPING_CAMERA_PAIRS=ch201:ch301,ch401:ch501 \
ADJACENT_CAMERA_PAIRS=ch501:ch601 \
scripts/start_reid_debug.sh
```

Flat sessions named `channel_<id>_*.mkv` and the older nested
`cam*_ch*/<filename>` layout are both discovered.

Every discovered recording is included unless `CHANNELS` or
`EXCLUDE_CHANNELS` is explicitly set. Set `RECORDINGS_ROOT`, `SESSION`, and
`FILE_NAME` for the local evaluation dataset. Annotated videos show global ID
and ReID distance/confidence information.

With `ANALYZE=1` (the default), the launcher also writes
`cross_camera_analysis.json`. It reports identity journeys, travel gaps,
simultaneous-camera conflicts, duplicate IDs, remaps, local-track recoveries,
allowed overlap/adjacent transitions, and close rejected matches. The summary
also includes averaged local-track prototype distances for diagnosing whether
a split comes from model appearance or gallery policy.

Export one compact chronological crop video per global identity after a replay:

```bash
python scripts/export_identity_videos.py \
  experiments/reid_audit/offline_tracks.jsonl \
  --output-dir experiments/reid_audit/identity_videos \
  --apply-edge-remap-guard
```

When an identity changes cameras, its crop video inserts a short transition
card and then continues with the crop from the next camera. The manifest records
the source time range, cameras, transitions, observation count, and output path.

## Operational Caveats

- The control endpoints are intended for a trusted LAN and currently have no
  authentication. Do not expose ports `8765` or `8787` to the public internet.
- Synchronize Orin clocks with NTP/PTP. Capture timestamps outside the configured
  skew bound fall back to prime receive time.
- The gallery is online state, not ground truth. Thresholds must be calibrated
  with labeled trajectories from the actual camera network.
- Lower detector confidence can recover distant people but also sends false
  crops into tracking/ReID. Keep the default at `0.50` until measured data
  justifies a change.
