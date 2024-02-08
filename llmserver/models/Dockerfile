# Dockerfile for llmserver - llama_cpp.server OpenAI API Server
#
#   Author: Jason A. Cox
#   21 Jan 2024
#   https://github.com/jasonacox/TinyLLM
#
#   Based on: https://github.com/abetlen/llama-cpp-python/blob/main/docker/cuda_simple/Dockerfile

# Use a base image - Be sure to match the CUDA version on your system
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
#FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Set environment variables
ENV MODEL=models/llama-2-7b-chat.Q5_K_M.gguf
ENV N_GPU_LAYERS=99
ENV HOST=0.0.0.0
ENV PORT=8000
ENV CHAT_FORMAT=llama-2
ENV INT_REQ=false
ENV N_CTX=2048

# Set the working directory
WORKDIR /app

# Install Python
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Copy local files into container
COPY build.sh .

# Setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install depencencies
RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

# Install latest llama-cpp-python (build with cuda)
RUN /app/build.sh

# Run the server
CMD python3 -m llama_cpp.server --model "$MODEL" --n_gpu_layers $N_GPU_LAYERS --port $PORT --chat_format $CHAT_FORMAT --interrupt_requests $INT_REQ --n_ctx $N_CTX

# Network
EXPOSE $PORT

# Build
#   docker build -t llmserver .
#
# Example docker run:
#    docker run -d \
#     --runtime=nvidia --gpus all \
#     -p $PORT:$PORT \
#     -v $PWD:/app/models \
#     -e MODEL=$MODEL \
#     -e N_CTX=$CONTEXT_SIZE \
#     -e CHAT_FORMAT=$CHAT_FORMAT \
#     -e N_GPU_LAYERS=$GPU_LAYERS \
#     -e HOST=$HOST \
#     -e PORT=$PORT \
#     -e INT_REQ=$INT_REQ \
#     --name llmserver \
#     --restart unless-stopped \
#     llmserver