# Jetson Deployment Preparation

This deployment uses two Orins and one camera per device:

```text
prime-jetson: cam1 YOLO worker + prime ONNX ReID/tracking/dashboard/recording
worker-jetson: cam2 YOLO worker
```

The active realtime path does not use Triton and does not require a ReID
TensorRT engine. The prime loads the generalized ONNX model directly with the
ONNX Runtime CUDA execution provider.

## Deployment Boundary

The repository installer deliberately does not install or upgrade JetPack,
CUDA, cuDNN, TensorRT, PyTorch, torchvision, or the NVIDIA camera stack. Those
packages form one JetPack compatibility unit and must already work before the
repository runtime is installed.

The realtime code requires Python 3.10 or newer. Use the Python version that
belongs to the installed JetPack release; do not replace the system CUDA stack
to satisfy a Python package.

Large models are ignored by Git. A fresh clone therefore also requires:

- `generalized_reid_swin_epoch119.onnx` on the prime
- `yolo26m.pt` on every camera worker
- a JetPack/Python-matched ONNX Runtime GPU aarch64 wheel on the prime

Do not use the x86 `onnxruntime-gpu` PyPI wheel on a Jetson. Select the aarch64
wheel for the exact JetPack and Python versions installed on that device.
The expected model sizes and SHA-256 hashes are pinned in
`deploy/model_manifest.yaml`; preflight rejects a partial or different model.

## 1. Stage Code And Models

Do not deploy from an uncommitted workstation tree. Commit the realtime source,
configs, requirements, tests, and deployment files first, then clone or pull
that commit on both devices.

Keep models outside Git, for example:

```text
~/TwinProject_models/reid_generalized_yolo11n/generalized_reid_swin_epoch119.onnx
~/TwinProject_models/reid_generalized_yolo11n/yolo26m.pt
```

## 2. Install The Python Runtime

The installer creates `.venv-jetson` with `--system-site-packages`. This keeps
JetPack's compatible CUDA-enabled PyTorch and OpenCV visible inside the venv.
Ultralytics and BoxMOT are installed without dependency resolution so pip
cannot replace those JetPack builds with generic packages.

Prime:

```bash
cd ~/Desktop/Reid_Inference_Pipeline
ORT_WHEEL=/absolute/path/to/onnxruntime_gpu-*-linux_aarch64.whl \
  scripts/install_jetson_runtime.sh prime
```

Worker:

```bash
cd ~/Desktop/Reid_Inference_Pipeline
scripts/install_jetson_runtime.sh worker
```

The prime installer fails when `CUDAExecutionProvider` is unavailable. It does
not silently accept CPU ReID.

## 3. Prepare Per-Device Environment

Create the private repository-local `.env`:

```bash
# Prime Jetson
scripts/reidctl.sh init prime

# Worker Jetson
scripts/reidctl.sh init worker
```

Edit `.env` on each device. It is ignored by Git and created with mode `0600`.
All normal runtime settings, config paths, camera assignments, model paths,
network endpoints, recording settings, and primary ReID thresholds are defined
there.

Prime example:

```dotenv
PIPELINE_ROLE=prime
LOCAL_CAMERA_ENABLED=true
CAMERA_IDS=cam1
REID_MODEL_PATH=~/TwinProject_models/reid_generalized_yolo11n/generalized_reid_swin_epoch119.onnx
YOLO_MODEL_PATH=~/TwinProject_models/reid_generalized_yolo11n/yolo26m.pt
PRIME_URL=ws://192.0.2.10:8765
REALTIME_OUTPUT_DIR=/mnt/recordings/reid
RECORDING_MOUNTPOINT=/mnt/recordings
RECORDING_MIN_FREE_GB=5
```

Worker example:

```dotenv
PIPELINE_ROLE=worker
CAMERA_IDS=cam2
YOLO_MODEL_PATH=~/TwinProject_models/reid_generalized_yolo11n/yolo26m.pt
PRIME_URL=ws://192.0.2.10:8765
```

`192.0.2.10` is a documentation-only address; replace it with the prime
device's LAN address.
Preflight verifies that the chosen directory is writable and has the configured
free-space reserve. When external storage is used, set `RECORDING_MOUNTPOINT`;
startup then fails rather than writing to the root filesystem if that mount is
missing or the output path is outside it.

## 4. Run Fail-Fast Preflight

First run the application smoke test. It validates `.env`, all selected YAML
files, typed overrides, packet serialization, a localhost WebSocket round trip,
global gallery behavior, and real model inference on the GPU:

```bash
scripts/reidctl.sh smoke --load-models
```

Then run the Jetson-specific model-integrity, CUDA, storage, clock, and camera
checks:

```bash
scripts/reidctl.sh preflight
```

Every required line must report `PASS`. In particular, verify:

- `torch_cuda`
- `onnxruntime_cuda_provider` on the prime
- `reid_inference` and `tracker_initialization` on the prime
- `yolo_inference` and `camera_capture` on camera devices
- `yolo_model_device` reports an actual CUDA device
- `reid_model_integrity` and `yolo_model_integrity`
- `recording_mountpoint` (when configured) and `recording_free_space`
- unique camera IDs (`cam1` and `cam2`)

## 5. Enable Boot Services

After both checks pass, render and enable services from the same `.env`:

```bash
# Run on each device with its own role.
scripts/install_jetson_services.sh prime --enable   # prime only
scripts/install_jetson_services.sh worker --enable  # worker only

# Required once if services must start before interactive login.
sudo loginctl enable-linger "$USER"
```

With `LOCAL_CAMERA_ENABLED=true`, the prime role installs two independent
services:

- `reid-prime.service`
- `reid-camera.service`

With `LOCAL_CAMERA_ENABLED=false`, only `reid-prime.service` is installed.

The worker role installs `reid-camera.service`. Camera control starts at boot,
scans for the stable `/dev/v4l/by-path/*video-index0` source, and retries when a
camera or worker process is unavailable. A dashboard stop action disables that
automatic restart until the next explicit start/restart action.

Source count is strict: `prime-jetson` must discover exactly the one source
mapped to `cam1`, and `worker-jetson` exactly the one source mapped to `cam2`.
This prevents a fallback name on one device from colliding with a camera ID on
the other device.

## 6. Operate And Diagnose

```bash
scripts/reidctl.sh status
scripts/reidctl.sh logs
scripts/reidctl.sh restart
scripts/reidctl.sh stop

systemctl --user status reid-prime.service reid-camera.service
journalctl --user -u reid-prime.service -f
journalctl --user -u reid-camera.service -f
curl -fsS http://127.0.0.1:8787/status | python3 -m json.tool
curl -fsS http://127.0.0.1:8765/status | python3 -m json.tool
```

Dashboard:

```text
http://<prime-lan-ip>:8765/
```

## Remaining Site Acceptance

Code preflight is necessary but not sufficient. Before unattended operation,
run a 30-60 minute two-camera acceptance test and confirm:

- sustained worker FPS near the configured 10 FPS
- no growing queue or dropped-packet count
- no thermal throttling or memory pressure
- valid recorded MP4 files after graceful and abrupt restarts
- camera unplug/replug recovery
- one known person retains one ID across the real camera route
- Orin clocks remain synchronized with NTP/PTP

Camera topology is not yet encoded. The current one-second global exclusion
prevents simultaneous reuse, but site-specific route and minimum travel times
are still required to improve recall safely among identical uniforms.

Recordings are finalized in 15-minute segments named like
`cam1_000001_processed.mp4`. Automatic deletion is deliberately disabled until
the site's disk-retention policy is defined. Recording pauses while free space
is below the default 5 GiB reserve and resumes after space is freed; inference
and the dashboard continue running.

## Compatibility References

- [Ultralytics NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- [NVIDIA JetPack setup](https://docs.nvidia.com/jetson/agx-orin-devkit/user-guide/latest/setup_jetpack.html)
- [NVIDIA PyTorch for Jetson](https://docs.nvidia.com/deeplearning/frameworks/pdf/Install-PyTorch-Jetson-Platform.pdf)
