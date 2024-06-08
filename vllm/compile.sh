#!/bin/bash
# Compile vLLM from source
# 
# Author: Jason A. Cox,
# Date: 27-Jan-2024
# https://github.com/jasonacox/TinyLLM

# Clone the vLLM Source
echo "Cloning vLLM source..."
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Copy helpful files
echo "Copying helpful files..."
cp ../run-pascal.sh run.sh
cp ../build.sh build.sh
cp ../vllm.patch vllm.patch

# Patch the source code
echo "Patching source code..."
patch < vllm.patch

# Build the docker image
echo "Building docker image..."
./build.sh

# Make models directory
echo "Creating models directory..."
mkdir -p models
echo "Models will be stored in ${PWD}/models."

# Done
echo "Build complete."
echo "To run vLLM, execute: ./run.sh"
