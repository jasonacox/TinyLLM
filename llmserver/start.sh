#!/bin/bash

# Log a few things
echo "Starting llama_cpp.server for $MODEL --n_gpu_layers $N_GPU_LAYERS --host $HOST --port $PORT"

# Ensure llama-cpp-python installed
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install llama-cpp-python[server]

# Configure server
MODEL=llama-2-7b-chat.Q5_K_M.gguf  # Use the llama-2-7b-chat model
                                   # (mistral-7b-instruct-v0.1.Q5_K_M.gguf)
HOST=0.0.0.0                       # Listen on all interfaces
PORT=5000                          # Listen on port 5000
N_GPU_LAYERS=32                    # Use 32 GPU layers

# Start server
python3 -m llama_cpp.server \
    --model "$MODEL" \
    --host $HOST \
    --port $PORT \
    --interrupt_requests False \
    --n_gpu_layers $N_GPU_LAYERS

echo "Started... listening on http://$HOST:$PORT"
