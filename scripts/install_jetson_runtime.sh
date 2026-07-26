#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROLE="${1:-}"
if [ "${ROLE}" != "prime" ] && [ "${ROLE}" != "worker" ]; then
    echo "Usage: $0 prime|worker"
    echo "Optional env: JETSON_VENV, PYTHON_BIN, ORT_WHEEL, ULTRALYTICS_VERSION, BOXMOT_VERSION"
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
JETSON_VENV="${JETSON_VENV:-$(pwd)/.venv-jetson}"
ULTRALYTICS_VERSION="${ULTRALYTICS_VERSION:-8.4.21}"
BOXMOT_VERSION="${BOXMOT_VERSION:-16.0.5}"

if [ "$(uname -m)" != "aarch64" ] && [ "${ALLOW_NON_JETSON:-0}" != "1" ]; then
    echo "ERROR: This installer is for Jetson aarch64 devices."
    echo "Set ALLOW_NON_JETSON=1 only when testing the installer itself."
    exit 1
fi
if [ ! -r /etc/nv_tegra_release ] && [ "${ALLOW_NON_JETSON:-0}" != "1" ]; then
    echo "ERROR: /etc/nv_tegra_release is missing; JetPack/L4T was not detected."
    exit 1
fi
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    echo "ERROR: This realtime pipeline requires Python 3.10 or newer."
    exit 1
fi

echo "Creating Jetson virtual environment: ${JETSON_VENV}"
"${PYTHON_BIN}" -m venv --system-site-packages "${JETSON_VENV}"
VENV_PYTHON="${JETSON_VENV}/bin/python"

"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${VENV_PYTHON}" -m pip install -r "requirements_${ROLE}_jetson.txt"
# These packages declare generic torch, torchvision and OpenCV dependencies.
# Resolve their other runtime dependencies above, then preserve JetPack's
# CUDA-enabled builds by installing the application packages without deps.
"${VENV_PYTHON}" -m pip install --no-deps \
    "ultralytics==${ULTRALYTICS_VERSION}" \
    'ultralytics-thop>=2.0.18'
if [ "${ROLE}" = "prime" ]; then
    "${VENV_PYTHON}" -m pip install --no-deps "boxmot==${BOXMOT_VERSION}"
fi

if [ "${ROLE}" = "prime" ]; then
    if [ -n "${ORT_WHEEL:-}" ]; then
        echo "Installing JetPack-matched ONNX Runtime wheel: ${ORT_WHEEL}"
        "${VENV_PYTHON}" -m pip install "${ORT_WHEEL}"
        # Some ONNX Runtime wheels loosen NumPy to an ABI-incompatible major.
        "${VENV_PYTHON}" -m pip install 'numpy>=1.23,<2.0'
    fi
    if ! "${VENV_PYTHON}" -c \
        'import onnxruntime as ort; assert "CUDAExecutionProvider" in ort.get_available_providers()' \
        >/dev/null 2>&1; then
        echo "ERROR: Prime requires a JetPack/Python-matched ONNX Runtime GPU wheel."
        echo "Set ORT_WHEEL=/path/to/onnxruntime_gpu-*-linux_aarch64.whl and rerun."
        exit 1
    fi
fi

"${VENV_PYTHON}" - <<'PY'
import cv2
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: JetPack PyTorch is installed but CUDA is unavailable")
print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
print(f"opencv={cv2.__version__}")
PY

echo
echo "Runtime installed locally. Models are intentionally not installed by pip."
echo "Run the role-specific deployment preflight after setting the model path environment."
if [ "${ROLE}" = "prime" ]; then
    echo "  scripts/with_onnxruntime_cuda_env.sh ${VENV_PYTHON} scripts/jetson_preflight.py --role prime --load-models"
else
    echo "  ${VENV_PYTHON} scripts/jetson_preflight.py --role worker --load-models --check-camera"
fi
