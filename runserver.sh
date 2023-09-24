#!/bin/bash
#
# Llama_cpp.server Startup
#
# This starts the llama-cpp-python[server] OpenAI API Compatible Web Server.
#
# Requirements:
#   * pip install llama-cpp-python[server]
#
# Author: Jason A. Cox
# 23 Sept 2023
# https://github.com/jasonacox/TinyLLM
#

# location of models
cd /data/ai/llama.cpp/models

# Example run of 7B 5-bit Llama-2 Chat model
/usr/bin/python3 -m llama_cpp.server \
    --model llama-2-7b-chat.Q5_K_M.gguf \
    --host localhost \
    --n_gpu_layers 32
