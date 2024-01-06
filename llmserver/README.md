# LLM Server

This creates an instance of llama_cpp.server which serves up a LLM with OpenAI API interface.

## Setup

```bash
# Install Python Libraries with Nvidia GPU support - Pin to v0.2.7 for now
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.27
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python[server]==0.2.27

# Download Models from HuggingFace
cd models

# Mistral 7B GGUF Q-5bit model Q5_K_M
wget https://huggingface.co/TheBloke/Mistral-7B-Claude-Chat-GGUF/resolve/main/mistral-7b-claude-chat.Q5_K_M.gguf

# Meta LLaMA-2 7B GGUF Q-5bit model Q5_K_M
wget https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf

cd ..

# Run Test - API Server
python3 -m llama_cpp.server \
    --model ./models/llama-2-7b-chat.Q5_K_M.gguf \
    --host localhost \
    --n_gpu_layers 32 \
    -n_ctx 2048 \
    --chat_format llama-2

```

## Test

You can see the service running at http://localhost:8000/

To test the LLM, you will need to run the [chatbot](https://github.com/jasonacox/TinyLLM/tree/main/chatbot#web-based-chatbot) or the [chat.py](https://github.com/jasonacox/TinyLLM/blob/main/chat.py) command line utility:

```bash
# Test via CLI
python3 ../chat.py
```

```
ChatBot - Greetings! My name is Jarvis. Enter an empty line to quit chat.

> Hi 

Jarvis> Hello! How may I assist you today? Is there anything specific you would like to know or discuss?

> Pick a color.

Jarvis> Certainly! If I had to choose a color, I would select blue as it is a calming and versatile color that represents trust, loyalty, and wisdom. How about you? What is your favorite color?

> I love green.

Jarvis> Excellent choice! Green is a vibrant and natural color that symbolizes growth, harmony, and balance. It's also the color of many living things, including plants and trees. Is there anything else you would like to know or discuss?
```

### Option - Run as a Service

You can set up a Linux service using the `tinyllm.service` file.  Make sure to edit `tinyllm` and `tinyllm.service` 
to have the username and paths required for your system.

```bash
# Clone this project for helper files
git clone https://github.com/jasonacox/TinyLLM.git
cd TinyLLM

# Edit the tinyllm.service to match your environment (ExecStart, WorkingDirectory & User)
vim tinyllm.service

# Copy the service file into systemd
sudo cp tinyllm.service /etc/systemd/system/

# Copy the init.d file for tinyllm
sudo cp tinyllm /etc/init.d

# Start and activate the service
sudo /etc/init.d/tinyllm start
sudo /etc/init.d/tinyllm enable

# Check status and logs
sudo /etc/init.d/tinyllm status
sudo /etc/init.d/tinyllm logs
```

## Optional - Docker Setup

You can also run the `setup.sh` script.

```bash
# First - Install Nvidia Support for Docker
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Second - Build container
docker build -t llmserver .

# Third - Run container
docker run \
    --runtime=nvidia --gpus all \
    -d \
    -p 8000:8000 \
    -v ./models:/app/models \
    -e MODEL=models/llama-2-7b-chat.Q5_K_M.gguf \
    -e N_GPU_LAYERS=32 \
    -e HOST=0.0.0.0 \
    -e PORT=8000 \
    --name llmserver \
    --restart unless-stopped \
    llmserver
```

## Chat Formats

The llama.cpp server will support chat formats for various models. The format is specified with the `--chat_format chatml` parameter in the startup command. Here are the valid chat formats:

```
alpaca
baichuan
baichuan-2
chatglm3
chatml
intel
llama-2
mistrallite
oasst_llama
openbuddy
openchat
open-orca
phind
pygmalion
qwen
redpajama-incite
saiga
snoozy
vicuna
zephyr
```

```bash
# Run Test - API Server
python3 -m llama_cpp.server \
    --model ./models/llama-2-7b-chat.Q5_K_M.gguf \
    --host localhost \
    --n_gpu_layers 32 \
    -n_ctx 2048 \
    --chat_format llama-2
```