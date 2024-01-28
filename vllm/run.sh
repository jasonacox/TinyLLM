#!/bin/bash
# Run vllm docker image
# 
# Author: Jason A. Cox,
# Date: 27-Jan-2024
# https://github.com/jasonacox/TinyLLM

LLM=mistralai/Mistral-7B-Instruct-v0.1
CONTAINER=vllm-mistral-1
#LLM=mistralai/Mistral-7B-Instruct-v0.2
#CONTAINER=vllm-mistral-2

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
  -v models:/app/models \
  --name $CONTAINER \
  vllm

# Additional options: -e EXTRA_ARGS="" -e MAX_MODEL_LEN=xxxxx -e QUANTIZATION=awq -e DTYPE=auto 

echo "Printing logs (^C to quit)..."

docker logs $CONTAINER -f
