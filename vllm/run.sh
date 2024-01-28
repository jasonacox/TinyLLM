#!/bin/bash

# vLLM Docker Container Image

echo "Starting vLLM..."

nvidia-docker run -d -p 8001:8001 --gpus=all --shm-size=10.24gb \
  -e MODEL=mistralai/Mistral-7B-Instruct-v0.1 \
  -e PORT=8001 \
  -e HF_HOME=/app/models \
  -e NUM_GPU=4 \
  -e EXTRA_ARGS="--dtype float --max-model-len 20000" \
  -v /data/models:/app/models \
  --name vllm \
  vllm 
  
echo "Printing logs (^C to quite)..."

docker logs vllm -f