#!/bin/bash
# This script is used to run the llmserver docker image

# Set default values
DEFAULT_MODEL="mistral-7b-instruct-v0.1.Q5_K_M.gguf"
DEFAULT_CONTEXT_SIZE=2048
DEFAULT_CHAT_FORMAT="chatml"
DEFAULT_GPU_LAYERS=99
DEFAULT_HOST="localhost"
DEFAULT_PORT=8000
DEFAULT_INT_REQ="false"

# Function to display help message
function display_help {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --model        Specify the model (default: $DEFAULT_MODEL)"
    echo "  --n_ctx        Specify the context size (default: $DEFAULT_CONTEXT_SIZE)"
    echo "  --chat_format  Specify the chat format (default: $DEFAULT_CHAT_FORMAT)"
    echo "  --gpu_layers   Specify the number of GPU layers (default: $DEFAULT_GPU_LAYERS)"
    echo "  --host         Specify the server host (default: $DEFAULT_HOST)"
    echo "  --port         Specify the server port (default: $DEFAULT_PORT)"
    echo "  --int_req      Allow interruptions to stop generation (default: $DEFAULT_INT_REQ)"
    echo "  -h             Display this help message"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift
            ;;
        --n_ctx)
            CONTEXT_SIZE="$2"
            shift
            ;;
        --chat_format)
            CHAT_FORMAT="$2"
            shift
            ;;
        --gpu_layers)
            GPU_LAYERS="$2"
            shift
            ;;
        --host)
            HOST="$2"
            shift
            ;;
        --port)
            PORT="$2"
            shift
            ;;
        --int_req)
            INT_REQ="$2"
            shift
            ;;
        -h|--help)
            display_help
            ;;
        *)
            echo "Unknown parameter passed: $1"
            display_help
            ;;
    esac
    shift
done

# Set default values if not specified
MODEL=${MODEL:-$DEFAULT_MODEL}
CONTEXT_SIZE=${CONTEXT_SIZE:-$DEFAULT_CONTEXT_SIZE}
CHAT_FORMAT=${CHAT_FORMAT:-$DEFAULT_CHAT_FORMAT}
GPU_LAYERS=${GPU_LAYERS:-$DEFAULT_GPU_LAYERS}
HOST=${HOST:-$DEFAULT_HOST}
PORT=${PORT:-$DEFAULT_PORT}
INT_REQ=${INT_REQ:-$DEFAULT_INT_REQ}

# Run the docker image
docker run -d \
    --runtime=nvidia --gpus all \
    -p $PORT:$PORT \
    -v $PWD:/app/models \
    -e MODEL=$MODEL \
    -e N_CTX=$CONTEXT_SIZE \
    -e CHAT_FORMAT=$CHAT_FORMAT \
    -e N_GPU_LAYERS=$GPU_LAYERS \
    -e HOST=$HOST \
    -e PORT=$PORT \
    -e INT_REQ=$INT_REQ \
    --name llmserver \
    --restart unless-stopped \
    llmserver

