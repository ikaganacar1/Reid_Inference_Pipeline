#!/bin/bash

# ReID Pipeline - Quick Start Script
# This script sets up and starts the ReID Pipeline web interface

set -e

echo "=========================================="
echo "ReID Pipeline - Setup and Start"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if NVIDIA Docker runtime is available (optional)
if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA Docker runtime detected${NC}"
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ NVIDIA Docker runtime not detected. Worker will use CPU mode.${NC}"
    GPU_AVAILABLE=false
fi

# Create required directories
echo ""
echo "Creating required directories..."
mkdir -p uploads outputs models

echo -e "${GREEN}✓ Directories created${NC}"

# Check for model files
echo ""
echo "Checking for model files..."
if [ -f "models/yolo11n.pt" ]; then
    echo -e "${GREEN}✓ YOLO model found${NC}"
else
    echo -e "${YELLOW}⚠ YOLO model not found at models/yolo11n.pt${NC}"
    echo "  You can:"
    echo "  1. Place your YOLO model in models/yolo11n.pt"
    echo "  2. Download it: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt"
    echo ""
    read -p "Continue without YOLO model? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build Docker images
echo ""
echo "Building Docker images..."
echo "This may take 10-15 minutes on first run..."
echo ""

if ! docker compose build; then
    echo -e "${RED}Error: Failed to build Docker images${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker images built successfully${NC}"

# Start services
echo ""
echo "Starting services..."
echo ""

if ! docker compose up -d; then
    echo -e "${RED}Error: Failed to start services${NC}"
    exit 1
fi

# Wait for services to be healthy
echo ""
echo "Waiting for services to be ready..."
sleep 5

# Check service status
echo ""
echo "Service Status:"
docker compose ps

# Print access information
echo ""
echo "=========================================="
echo -e "${GREEN}✓ ReID Pipeline is running!${NC}"
echo "=========================================="
echo ""
echo "Access the services:"
echo "  • Web Interface: http://localhost:3000"
echo "  • API Documentation: http://localhost:8000/docs"
echo "  • API Health Check: http://localhost:8000/api/health"
echo ""
echo "Useful commands:"
echo "  • View logs:         docker compose logs -f"
echo "  • Stop services:     docker compose down"
echo "  • Restart services:  docker compose restart"
echo "  • View status:       docker compose ps"
echo ""
echo "For detailed documentation, see:"
echo "  • DEPLOYMENT.md      - Deployment guide"
echo "  • WEB_INTERFACE.md   - Web interface documentation"
echo "  • CLAUDE.md          - Pipeline internals"
echo ""

# Open browser (optional)
if command -v xdg-open &> /dev/null; then
    read -p "Open web interface in browser? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        xdg-open http://localhost:3000
    fi
elif command -v open &> /dev/null; then
    read -p "Open web interface in browser? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open http://localhost:3000
    fi
fi

echo ""
echo "Happy tracking! 🎥📹"
