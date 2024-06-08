#!/bin/bash
# Run vllm docker image
# 
# Usage: run.sh <model> <container_name>
# 
# Author: Jason A. Cox,
# Date: 27-Jan-2024
# https://github.com/jasonacox/TinyLLM

# Set Defaults
LLM_DEFAULT=mistralai/Mistral-7B-Instruct-v0.1
CONTAINER_DEFAULT=vllm-mistral-1x

# Check if user asked for help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 <model> <container_name>"
    echo "Example: $0 mistralai/Mistral-7B-Instruct-v0.1 vllm-mistral-1"
    exit 0
fi

# If user provided arguments, use them
if [[ ! -z "$1" && ! -z "$2" ]]; then
    LLM=$1
    CONTAINER=$2
fi

# Set variables to default if not set
if [[ -z "${LLM}" ]]; then
    LLM=$LLM_DEFAULT
    CONTAINER=$CONTAINER_DEFAULT
fi

echo "Stopping and removing any previous $CONTAINER instance..."
docker stop $CONTAINER
docker rm $CONTAINER

echo "Starting new $CONTAINER instance..."

docker run -d \
  -p 8000:8000 \
  --shm-size=10.24gb \
  --gpus all \
  -e MODEL=$LLM \
  -e PORT=8000 \
  -e GPU_MEMORY_UTILIZATION=0.95 \
  -e NUM_GPU=1 \
  -e SERVED_MODEL_NAME=tinyllm \
  -e HF_HOME=/app/models \
  -v $PWD/models:/app/models \
  --restart unless-stopped \
  --name $CONTAINER \
  vllm

# Additional options: -e EXTRA_ARGS="" -e MAX_MODEL_LEN=xxxxx -e QUANTIZATION=awq -e DTYPE=auto 

echo "Printing logs (^C to quit)..."

docker logs $CONTAINER -f
