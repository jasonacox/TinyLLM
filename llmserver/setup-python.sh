#!/bin/bash

# Ensure llama-cpp-python installed
echo "Building llama-cpp-python..."
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install llama-cpp-python
echo "Building llama-cpp-python[server]..."
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install llama-cpp-python[server]

echo "Build complete. Run using 'run.sh'."
