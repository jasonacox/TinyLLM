# Use a base image
#FROM python:3.10
FROM ubuntu

# Set environment variables
ENV MODEL=models/llama-2-7b-chat.Q5_K_M.gguf
ENV N_GPU_LAYERS=32
ENV HOST=0.0.0.0
ENV PORT=8000

# Install CUDA
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
# RUN sudo dpkg -i cuda-keyring_1.1-1_all.deb
# RUN sudo apt-get update
# RUN sudo apt-get -y install cuda

# Install Python
RUN apt-get update && \
    apt-get install -y \
        python3-pip \
        python3-dev \
        libglib2.0-0

# Install the 'llama-cpp-python' package
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install llama-cpp-python
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install llama-cpp-python[server]

# Copy local files into container
# COPY . .

# Set the working directory
WORKDIR /app

# Run the llama_cpp.server command
CMD python3 -m llama_cpp.server --model "$MODEL" --n_gpu_layers $N_GPU_LAYERS --host "$HOST" --port $PORT

# Network
EXPOSE $PORT

# docker run \
#     -d \
#     -p 8000:8000 \
#     -v models:/app/models \
#     -e MODEL=models/llama-2-7b-chat.Q5_K_M.gguf \
#     -e N_GPU_LAYERS=32 \
#     -e HOST=localhost \
#     -e PORT=8000 \
#     --name llmserver \
#     --restart unless-stopped \
#     llmserver
