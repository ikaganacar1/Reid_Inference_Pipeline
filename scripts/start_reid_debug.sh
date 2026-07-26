#!/usr/bin/env bash
set -euo pipefail

# Offline factory CCTV ReID debug starter.
#
# Defaults debug the current generalized ReID model on a short dense segment
# with the normal Ultralytics YOLO medium model on GPU:
#   scripts/start_reid_debug.sh
#
# Useful overrides:
#   REID_PRESET=tao1024 scripts/start_reid_debug.sh
#   YOLO_MODEL=yolo11n.pt scripts/start_reid_debug.sh
#   CHANNELS=1501,2901,501,301,701 START_FRAME=900 MAX_FRAMES=600 STRIDE=10 scripts/start_reid_debug.sh
#   SAVE_CROPS=1 OUTPUT_DIR=experiments/my_reid_debug scripts/start_reid_debug.sh
#   ANNOTATED_VIDEO_WIDTH=1920 ANNOTATED_VIDEO_FPS=10 scripts/start_reid_debug.sh
#   REPLAY_FPS=25 STRIDE=1 scripts/start_reid_debug.sh
#   MIN_CROSS_CAMERA_GAP_SECONDS=0 scripts/start_reid_debug.sh
#
# Extra args are passed directly to scripts/debug_reid_recordings.py.

cd "$(dirname "$0")/.."

CONDA_ENV="${CONDA_ENV:-tensorrt_blackwell}"
# Empty means every discovered recording except the exclusions defined by the
# replay tool. Set CHANNELS only when intentionally testing a smaller subset.
CHANNELS="${CHANNELS:-}"
EXCLUDE_CHANNELS="${EXCLUDE_CHANNELS-2601,3001,401,2201,2101,1201,1701}"
RECORDINGS_ROOT="${RECORDINGS_ROOT:-recordings}"
SESSION="${SESSION:-session}"
FILE_NAME="${FILE_NAME:-recording.mkv}"
START_FRAME="${START_FRAME:-0}"
MAX_FRAMES="${MAX_FRAMES:-600}"
STRIDE="${STRIDE:-2}"
REPLAY_FPS="${REPLAY_FPS:-0}"
DETECTOR="${DETECTOR:-yolo}"
YOLO_DEVICE="${YOLO_DEVICE:-cuda:0}"
YOLO_CONF="${YOLO_CONF:-0.50}"
YOLO_PROMPTS="${YOLO_PROMPTS:-person}"
REID_BATCH_SIZE="${REID_BATCH_SIZE:-8}"
REID_PROVIDER="${REID_PROVIDER:-CUDAExecutionProvider}"
SAVE_ANNOTATED_EVERY="${SAVE_ANNOTATED_EVERY:-10}"
SAVE_ANNOTATED_VIDEO="${SAVE_ANNOTATED_VIDEO:-1}"
# Zero lets the replay derive source_fps / stride and preserve wall-clock speed.
ANNOTATED_VIDEO_FPS="${ANNOTATED_VIDEO_FPS:-0}"
ANNOTATED_VIDEO_WIDTH="${ANNOTATED_VIDEO_WIDTH:-1280}"
SAVE_CROPS="${SAVE_CROPS:-0}"
ANALYZE="${ANALYZE:-1}"
MIN_CROSS_CAMERA_GAP_SECONDS="${MIN_CROSS_CAMERA_GAP_SECONDS:-2.0}"
ALLOW_ALL_CAMERA_OVERLAP="${ALLOW_ALL_CAMERA_OVERLAP:-0}"
OVERLAPPING_CAMERA_PAIRS="${OVERLAPPING_CAMERA_PAIRS:-}"
ADJACENT_CAMERA_PAIRS="${ADJACENT_CAMERA_PAIRS:-}"
REID_PRESET="${REID_PRESET:-generalized}"
TRACKER="${TRACKER:-botsort}"
BOTSORT_TRACK_BUFFER="${BOTSORT_TRACK_BUFFER:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/offline_reid_debug_$(date +%Y%m%d_%H%M%S)_${REID_PRESET}}"

if [ -z "${YOLO_MODEL:-}" ]; then
    case "${DETECTOR}" in
        yolo)
            YOLO_MODEL="yolo26m.pt"
            ;;
        yoloe)
            YOLO_MODEL="models/yoloe-26x-seg.pt"
            ;;
        *)
            echo "ERROR: Unknown DETECTOR='${DETECTOR}'. Use yoloe or yolo."
            exit 2
            ;;
    esac
fi

if [ -z "${YOLO_IMGSZ:-}" ]; then
    if [ "${DETECTOR}" = "yoloe" ]; then
        YOLO_IMGSZ="640"
    else
        YOLO_IMGSZ="640"
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found."
    exit 1
fi

source scripts/onnxruntime_cuda_env.sh
CONDA_PREFIX_PATH="$(conda_env_prefix "${CONDA_ENV}")"
CONDA_SITE_PACKAGES="$(conda_env_site_packages "${CONDA_ENV}")"
prepend_onnxruntime_cuda_ld_library_path "${CONDA_PREFIX_PATH}" "${CONDA_SITE_PACKAGES}"

case "${REID_PRESET}" in
    generalized)
        REID_MODEL="${REID_MODEL:-TwinProject_models/reid_generalized_yolo11n/generalized_reid_swin_epoch119.onnx}"
        REID_INPUT_HEIGHT="${REID_INPUT_HEIGHT:-256}"
        REID_INPUT_WIDTH="${REID_INPUT_WIDTH:-128}"
        EMBEDDING_DIM="${EMBEDDING_DIM:-1024}"
        ;;
    tao1024)
        REID_MODEL="${REID_MODEL:-models/reidentificationnet_transformer_vdeployable_v1.0/swin_base_market1501_aicity156_featuredim1024.onnx}"
        REID_INPUT_HEIGHT="${REID_INPUT_HEIGHT:-256}"
        REID_INPUT_WIDTH="${REID_INPUT_WIDTH:-128}"
        EMBEDDING_DIM="${EMBEDDING_DIM:-1024}"
        ;;
    tao256)
        REID_MODEL="${REID_MODEL:-models/reidentificationnet_transformer_vdeployable_v1.0/swin_base_market1501_aicity156_featuredim256.onnx}"
        REID_INPUT_HEIGHT="${REID_INPUT_HEIGHT:-256}"
        REID_INPUT_WIDTH="${REID_INPUT_WIDTH:-128}"
        EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
        ;;
    tiny256)
        REID_MODEL="${REID_MODEL:-models/reidentificationnet_transformer_vdeployable_v1.0/swin_tiny_market1501_aicity156_featuredim256.onnx}"
        REID_INPUT_HEIGHT="${REID_INPUT_HEIGHT:-256}"
        REID_INPUT_WIDTH="${REID_INPUT_WIDTH:-128}"
        EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
        ;;
    ltcc)
        REID_MODEL="${REID_MODEL:-models/lttc_0.1.4.49.onnx}"
        REID_INPUT_HEIGHT="${REID_INPUT_HEIGHT:-384}"
        REID_INPUT_WIDTH="${REID_INPUT_WIDTH:-192}"
        EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
        ;;
    *)
        echo "ERROR: Unknown REID_PRESET='${REID_PRESET}'. Use generalized, tao1024, tao256, tiny256, or ltcc."
        exit 2
        ;;
esac

debug_args=(
    scripts/debug_reid_recordings.py
    --recordings-root "${RECORDINGS_ROOT}"
    --session "${SESSION}"
    --file-name "${FILE_NAME}"
    --detector "${DETECTOR}"
    --channels "${CHANNELS}"
    --exclude "${EXCLUDE_CHANNELS}"
    --start-frame "${START_FRAME}"
    --max-frames "${MAX_FRAMES}"
    --stride "${STRIDE}"
    --replay-fps "${REPLAY_FPS}"
    --output-dir "${OUTPUT_DIR}"
    --yolo-model "${YOLO_MODEL}"
    --yolo-device "${YOLO_DEVICE}"
    --yolo-imgsz "${YOLO_IMGSZ}"
    --yolo-conf "${YOLO_CONF}"
    --yolo-prompts "${YOLO_PROMPTS}"
    --reid-model "${REID_MODEL}"
    --reid-input-height "${REID_INPUT_HEIGHT}"
    --reid-input-width "${REID_INPUT_WIDTH}"
    --embedding-dim "${EMBEDDING_DIM}"
    --reid-batch-size "${REID_BATCH_SIZE}"
    --reid-provider "${REID_PROVIDER}"
    --tracker "${TRACKER}"
    --botsort-track-buffer "${BOTSORT_TRACK_BUFFER}"
    --save-annotated-every "${SAVE_ANNOTATED_EVERY}"
    --annotated-video-fps "${ANNOTATED_VIDEO_FPS}"
    --annotated-video-width "${ANNOTATED_VIDEO_WIDTH}"
    --overlapping-camera-pairs "${OVERLAPPING_CAMERA_PAIRS}"
    --adjacent-camera-pairs "${ADJACENT_CAMERA_PAIRS}"
)

if [ "${ALLOW_ALL_CAMERA_OVERLAP}" = "1" ]; then
    debug_args+=(--allow-all-camera-overlap)
fi

if [ "${SAVE_ANNOTATED_VIDEO}" = "1" ]; then
    debug_args+=(--save-annotated-video)
else
    debug_args+=(--no-save-annotated-video)
fi

if [ "${SAVE_CROPS}" = "1" ]; then
    debug_args+=(--save-crops)
fi

echo "Starting offline ReID debug"
echo "  env: ${CONDA_ENV}"
echo "  conda_prefix: ${CONDA_PREFIX_PATH}"
echo "  detector: ${DETECTOR}"
echo "  yolo_model: ${YOLO_MODEL}"
echo "  yolo_device: ${YOLO_DEVICE}"
echo "  yolo_conf: ${YOLO_CONF}"
echo "  yolo_imgsz: ${YOLO_IMGSZ}"
echo "  preset: ${REID_PRESET}"
echo "  reid_model: ${REID_MODEL}"
echo "  reid_provider: ${REID_PROVIDER}"
echo "  tracker: ${TRACKER}"
echo "  botsort_track_buffer: ${BOTSORT_TRACK_BUFFER}"
echo "  recordings: ${RECORDINGS_ROOT}/${SESSION}"
if [ -n "${CHANNELS}" ]; then
    echo "  channels: ${CHANNELS}"
else
    echo "  channels: all discovered channels except configured exclusions"
fi
echo "  excluded channels: ${EXCLUDE_CHANNELS:-none}"
echo "  frame window: start=${START_FRAME} max=${MAX_FRAMES} stride=${STRIDE}"
echo "  replay_fps: ${REPLAY_FPS:-auto}"
echo "  annotated_video: ${SAVE_ANNOTATED_VIDEO} width=${ANNOTATED_VIDEO_WIDTH} fps=${ANNOTATED_VIDEO_FPS}"
echo "  min_cross_camera_gap_seconds: ${MIN_CROSS_CAMERA_GAP_SECONDS}"
echo "  overlap_all: ${ALLOW_ALL_CAMERA_OVERLAP}"
echo "  overlap_pairs: ${OVERLAPPING_CAMERA_PAIRS:-none}"
echo "  adjacent_pairs: ${ADJACENT_CAMERA_PAIRS:-none}"
echo "  output: ${OUTPUT_DIR}"

conda run --no-capture-output -n "${CONDA_ENV}" \
    env -u LD_PRELOAD LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" python "${debug_args[@]}" "$@"

events_path="${OUTPUT_DIR}/reid_debug/events.jsonl"
if [ "${ANALYZE}" = "1" ] && [ -f "${events_path}" ]; then
    echo
    echo "Analyzing debug events"
    conda run --no-capture-output -n "${CONDA_ENV}" env -u LD_PRELOAD LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
        python scripts/analyze_reid_debug.py "${events_path}" --tail 0 \
        --min-cross-camera-gap-seconds "${MIN_CROSS_CAMERA_GAP_SECONDS}"

    echo
    echo "Auditing cross-camera journeys"
    cross_camera_analysis_args=(
        scripts/analyze_cross_camera_reid.py
        "${events_path}"
        --min-travel-seconds "${MIN_CROSS_CAMERA_GAP_SECONDS}"
        --overlapping-camera-pairs "${OVERLAPPING_CAMERA_PAIRS}"
        --adjacent-camera-pairs "${ADJACENT_CAMERA_PAIRS}"
    )
    if [ "${ALLOW_ALL_CAMERA_OVERLAP}" = "1" ]; then
        cross_camera_analysis_args+=(--allow-all-camera-overlap)
    fi
    conda run --no-capture-output -n "${CONDA_ENV}" env -u LD_PRELOAD LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
        python "${cross_camera_analysis_args[@]}"
fi

echo
echo "Debug output: ${OUTPUT_DIR}"
