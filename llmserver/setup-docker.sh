#!/bin/bash

# Install Nvidia support for docker
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test Container
# sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

# Ask user if they want to start the server
read -p "Do you want to start the server? (y/n): " start_server

if [[ $start_server == "y" ]]; then
  # Run Container
  docker run \
    --runtime=nvidia --gpus all \
    -d \
    -p 8000:8000 \
    -v $PWD/models:/app/models \
    -e MODEL=models/llama-2-7b-chat.Q5_K_M.gguf \
    -e N_GPU_LAYERS=32 \
    -e HOST=0.0.0.0 \
    -e PORT=8000 \
    --name llmserver \
    --restart unless-stopped \
    llmserver
else
  echo "Server not started."
fi
