#!/bin/bash
# Build vllm docker image
#
# Date: 27-Jan-2024
# https://github.com/jasonacox/TinyLLM

echo "Build vllm docker image..."

nvidia-docker build -t vllm .
