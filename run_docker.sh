#!/bin/bash

# Object Detection API Docker Runner
# This script helps build and run the Docker container for the object detection service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="object-detection-api"
CONTAINER_NAME="object-detection-api-container"
PORT=${PORT:-8000}

print_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build          Build the Docker image"
    echo "  run            Run the container (will build if needed)"
    echo "  stop           Stop the running container"
    echo "  logs           Show container logs"
    echo "  shell          Open shell in running container"
    echo "  clean          Remove container and image"
    echo "  health         Check API health"
    echo ""
    echo "Options:"
    echo "  --port PORT    Set the port to expose (default: 8000)"
    echo "  --gpu ID       Specify GPU device ID (default: 0)"
    echo ""
    echo "Examples:"
    echo "  $0 build                    # Build the image"
    echo "  $0 run --port 8080         # Run on port 8080"
    echo "  $0 run --gpu 1             # Use GPU device 1"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi
}

check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:11.1-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}Warning: NVIDIA Docker runtime may not be available${NC}"
        echo "Make sure you have NVIDIA Container Runtime installed"
    fi
}

build_image() {
    echo -e "${GREEN}Building Docker image: $IMAGE_NAME${NC}"
    
    # Check if checkpoints exist
    if [ ! -d "checkpoints" ] || [ -z "$(ls -A checkpoints)" ]; then
        echo -e "${YELLOW}Checkpoints directory is empty. Running download script...${NC}"
        if [ -f "scripts/download_checkpoints.sh" ]; then
            bash scripts/download_checkpoints.sh
        else
            echo -e "${RED}Error: scripts/download_checkpoints.sh not found${NC}"
            echo "Please ensure the download script exists or manually download checkpoints"
            exit 1
        fi
    fi
    
    docker build -t $IMAGE_NAME .
    echo -e "${GREEN}Build completed successfully${NC}"
}

run_container() {
    local gpu_device=${GPU_DEVICE:-0}
    
    echo -e "${GREEN}Starting container: $CONTAINER_NAME${NC}"
    echo "Port: $PORT"
    echo "GPU Device: $gpu_device"
    
    # Stop existing container if running
    if docker ps -a --format 'table {{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        echo "Stopping existing container..."
        docker stop $CONTAINER_NAME &> /dev/null || true
        docker rm $CONTAINER_NAME &> /dev/null || true
    fi
    
    # Run new container
    docker run -d \
        --name $CONTAINER_NAME \
        --gpus "device=$gpu_device" \
        -p $PORT:8000 \
        -e CUDA_VISIBLE_DEVICES=$gpu_device \
        -v "$(pwd)/results:/app/results" \
        --restart unless-stopped \
        $IMAGE_NAME
    
    echo -e "${GREEN}Container started successfully${NC}"
    echo "API will be available at: http://localhost:$PORT"
    echo "API documentation: http://localhost:$PORT/docs"
    echo ""
    echo "Use '$0 logs' to view container logs"
    echo "Use '$0 health' to check API health"
}

stop_container() {
    echo -e "${YELLOW}Stopping container: $CONTAINER_NAME${NC}"
    docker stop $CONTAINER_NAME &> /dev/null || echo "Container not running"
    docker rm $CONTAINER_NAME &> /dev/null || echo "Container not found"
    echo -e "${GREEN}Container stopped${NC}"
}

show_logs() {
    echo -e "${GREEN}Container logs:${NC}"
    docker logs -f $CONTAINER_NAME
}

open_shell() {
    echo -e "${GREEN}Opening shell in container: $CONTAINER_NAME${NC}"
    docker exec -it $CONTAINER_NAME /bin/bash
}

clean_all() {
    echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
    docker stop $CONTAINER_NAME &> /dev/null || true
    docker rm $CONTAINER_NAME &> /dev/null || true
    docker rmi $IMAGE_NAME &> /dev/null || true
    echo -e "${GREEN}Cleanup completed${NC}"
}

check_health() {
    echo -e "${GREEN}Checking API health...${NC}"
    
    if ! docker ps --format 'table {{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        echo -e "${RED}Container is not running${NC}"
        exit 1
    fi
    
    # Wait a moment for the API to be ready
    sleep 2
    
    if curl -s -f "http://localhost:$PORT/health" > /dev/null; then
        echo -e "${GREEN}API is healthy and responding${NC}"
        curl -s "http://localhost:$PORT/health" | python3 -m json.tool
    else
        echo -e "${RED}API is not responding${NC}"
        echo "Check logs with: $0 logs"
        exit 1
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --gpu)
            GPU_DEVICE="$2"
            shift 2
            ;;
        build|run|stop|logs|shell|clean|health)
            COMMAND="$1"
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Main logic
check_docker

case ${COMMAND:-run} in
    build)
        build_image
        ;;
    run)
        check_nvidia_docker
        # Build if image doesn't exist
        if ! docker images --format 'table {{.Repository}}' | grep -q "^$IMAGE_NAME$"; then
            build_image
        fi
        run_container
        ;;
    stop)
        stop_container
        ;;
    logs)
        show_logs
        ;;
    shell)
        open_shell
        ;;
    clean)
        clean_all
        ;;
    health)
        check_health
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        print_usage
        exit 1
        ;;
esac
