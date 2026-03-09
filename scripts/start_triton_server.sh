#!/bin/bash
# Triton Inference Server Startup Script

set -e

# Configuration
TRITON_IMAGE="nvcr.io/nvidia/tritonserver:25.04-py3"
CONTAINER_NAME="triton-reid-server"
MODEL_REPO="$(pwd)/triton_models"
HTTP_PORT=8100
GRPC_PORT=8101
METRICS_PORT=8102

echo "=================================================="
echo "Starting Triton Inference Server"
echo "=================================================="
echo "Model repository: $MODEL_REPO"
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
docker run --rm -d \
    --gpus all \
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
        --model-control-mode=poll \
        --repository-poll-secs=30

echo "Container started: $CONTAINER_NAME"
echo ""

# Wait for server to be ready
echo "Waiting for Triton server to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0

until curl -s http://localhost:$HTTP_PORT/v2/health/ready | grep -q "true"; do
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
curl -s http://localhost:$HTTP_PORT/v2/models/swin_base_reid | python3 -m json.tool || echo "WARNING: Model swin_base_reid not loaded"

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
