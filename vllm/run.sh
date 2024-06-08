#!/bin/bash
# Run vllm docker image
# 
# Author: Jason A. Cox,
# Date: 27-Jan-2024
# https://github.com/jasonacox/TinyLLM

LLM=mistralai/Mistral-7B-Instruct-v0.1
CONTAINER=vllm

echo "Stopping and removing any previous $CONTAINER instance..."
docker stop $CONTAINER
docker rm $CONTAINER

echo "Starting new $CONTAINER instance..."

echo "Starting vLLM $CONTAINER..."
docker run -d --gpus all \
    -v $PWD/models:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN={Your_Hugingface_Token}" \
    --restart unless-stopped \
    --name $CONTAINER \
    vllm \
    --host 0.0.0.0 \
    --model=$MODEL \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --served-model-name $LLM \
    --dtype=float

    # Additional arguments to pass to the API server on startup:
    # --gpu-memory-utilization 
    # --max-model-len
    # --dtype auto|half
    # --quantization 
    # --disable-log-requests

echo "Printing logs (^C to quit)..."

docker logs $CONTAINER -f
