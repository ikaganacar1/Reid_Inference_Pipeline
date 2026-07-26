#!/bin/bash
# Triton Inference Server Startup Script

set -e

cd "$(dirname "$0")/.."
source scripts/runtime_env.sh
load_runtime_env

# Configuration
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.04-py3-igpu}"
CONTAINER_NAME="${TRITON_CONTAINER_NAME:-triton-reid-server}"
MODEL_REPO="${TRITON_MODEL_REPOSITORY:-$(pwd)/triton_models}"
MODEL_NAME="${TRITON_MODEL_NAME:-generalized_reid_swin}"
GPU_RUNTIME_ARGS=()
if [ -n "${TRITON_DOCKER_GPU_ARGS:-}" ]; then
    # Example for discrete GPU hosts:
    #   TRITON_DOCKER_GPU_ARGS="--gpus all" bash scripts/start_triton_server.sh
    read -r -a GPU_RUNTIME_ARGS <<< "$TRITON_DOCKER_GPU_ARGS"
else
    # Jetson/iGPU Triton images require the NVIDIA runtime directly.
    GPU_RUNTIME_ARGS=(--runtime=nvidia)
fi
HTTP_PORT="${TRITON_HTTP_PORT:-8100}"
GRPC_PORT="${TRITON_GRPC_PORT:-8101}"
METRICS_PORT="${TRITON_METRICS_PORT:-8102}"
MAX_RETRIES="${TRITON_STARTUP_RETRIES:-120}"

echo "=================================================="
echo "Starting Triton Inference Server"
echo "=================================================="
echo "Model repository: $MODEL_REPO"
echo "Triton image: $TRITON_IMAGE"
echo "GPU runtime args: ${GPU_RUNTIME_ARGS[*]}"
echo "HTTP port: $HTTP_PORT"
echo "gRPC port: $GRPC_PORT"
echo "Metrics port: $METRICS_PORT"
echo ""

# Check if model repository exists
if [ ! -d "$MODEL_REPO" ]; then
    echo "ERROR: Model repository not found: $MODEL_REPO"
    echo "Run scripts/setup_triton_model.py first"
    exit 1
fi

# Check if container is already running
if docker ps | grep -q $CONTAINER_NAME; then
    echo "Triton server is already running!"
    echo "To stop it, run: docker stop $CONTAINER_NAME"
    exit 0
fi

# Remove old container if exists
if docker ps -a | grep -q $CONTAINER_NAME; then
    echo "Removing old container..."
    docker rm $CONTAINER_NAME
fi

# Start Triton server
echo "Starting Triton server container..."
docker run -d \
    "${GPU_RUNTIME_ARGS[@]}" \
    --name $CONTAINER_NAME \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p $HTTP_PORT:8000 \
    -p $GRPC_PORT:8001 \
    -p $METRICS_PORT:8002 \
    -v "$MODEL_REPO":/models \
    $TRITON_IMAGE \
    tritonserver \
        --model-repository=/models \
        --log-verbose=1 \
        --strict-model-config=false \
        --model-control-mode=explicit \
        --load-model=$MODEL_NAME

echo "Container started: $CONTAINER_NAME"
echo ""

# Wait for server to be ready
echo "Waiting for Triton server to be ready..."
RETRY_COUNT=0

until curl -sf http://localhost:$HTTP_PORT/v2/health/ready >/dev/null; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: Triton server failed to start within timeout"
        echo "Check logs with: docker logs $CONTAINER_NAME"
        exit 1
    fi
    sleep 1
    echo -n "."
done

echo ""
echo "=================================================="
echo "Triton server is ready!"
echo "=================================================="

# Check model status
echo ""
echo "Model status:"
curl -s http://localhost:$HTTP_PORT/v2/models/$MODEL_NAME | python3 -m json.tool || echo "WARNING: Model $MODEL_NAME not loaded"

echo ""
echo "=================================================="
echo "Triton server endpoints:"
echo "  HTTP:    http://localhost:$HTTP_PORT"
echo "  gRPC:    localhost:$GRPC_PORT"
echo "  Metrics: http://localhost:$METRICS_PORT/metrics"
echo ""
echo "To stop server: docker stop $CONTAINER_NAME"
echo "To view logs:   docker logs -f $CONTAINER_NAME"
echo "=================================================="
