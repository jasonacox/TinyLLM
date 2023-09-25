# Use a base image
FROM python:3.10

# Set environment variables
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on" 
ENV FORCE_CMAKE=1
ENV MODEL=models/llama-2-7b-chat.Q5_K_M.gguf
ENV N_GPU_LAYERS=32
ENV HOST=localhost
ENV PORT=8000

# Install the 'llama-cpp-python' package
RUN pip3 install llama-cpp-python[server]

# Copy local files into container
# COPY . .

# Set the working directory
WORKDIR /app

# Run the llama_cpp.server command
CMD ["python3", "-m", "llama_cpp.server", "--model", "${MODEL}", "--n_gpu_layers", "${N_GPU_LAYERS}", "--host", "${HOST}", "--port", "${PORT}"]

# Network
EXPOSE ${PORT}

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
