#!/bin/bash

echo "Build vllm docker image from source..."

nvidia-docker build -t vllm .
