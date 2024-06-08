#!/bin/bash
# Run vllm docker image - Quantized with AWQ
# 
# Author: Jason A. Cox,
# Date: 27-Jan-2024
# https://github.com/jasonacox/TinyLLM

#LLM=TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ
#CONTAINER=vllm-mixtral
#QT=auto
LLM=TheBloke/Mistral-7B-Instruct-v0.1-AWQ
CONTAINER=vllm-mistral-awq
QT=auto

echo "Stopping and removing any previous $CONTAINER instance..."
docker stop $CONTAINER
docker rm $CONTAINER

echo "Starting new $CONTAINER instance..."

docker run -d --gpus all \
    -v $PWD/models:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN={Your_Hugingface_Token}" \
    --restart unless-stopped \
    --name $CONTAINER \
    vllm/vllm-openai:latest \
    --host 0.0.0.0 \
    --model=$MODEL \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --served-model-name $LLM \
    --disable-log-requests \
    --dtype=auto \
    --quantization awq

    # Additional arguments to pass to the API server on startup:
    # --gpu-memory-utilization 
    # --max-model-len
    # --dtype
    # --quantization
    # --enforce-eager
    # --disable-log-requests


-q ${QUANTIZATION} --dtype ${DTYPE}"

echo "Printing logs (^C to quit)..."

docker logs $CONTAINER -f
