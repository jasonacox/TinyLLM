#!/bin/bash
# This script is used to build the llmserver docker image

# Install Python Libraries with Nvidia GPU support - Pin to v0.2.27 for now
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.27
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python[server]==0.2.27

# Optional - Build from Source - current
# git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
# cd llama-cpp-python
# make update.vendor
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install -e .
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install -e .[server]