#!/bin/bash

# Log a few things
echo "Starting llama_cpp.server for $MODEL --n_gpu_layers $N_GPU_LAYERS --host $HOST --port $PORT"

# Ensure llama-cpp-python installed
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install llama-cpp-python[server]

# Start server
python3 -m llama_cpp.server --model "$MODEL" --n_gpu_layers $N_GPU_LAYERS --port $PORT

echo "Stop"
