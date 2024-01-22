#!/bin/bash
# This script is used to build the llmserver docker image

git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
make update.vendor
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install -e .
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install -e .[server]