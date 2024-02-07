# LLM Server

LLMserver uses the llama-cpp-python library which has a built-in OpenAI API compatible server. This can be used to host your model locally and use OpenAI API tools against your self-hosted LLM.

## Setup

This creates an instance of llama_cpp.server which serves up a LLM with OpenAI API interface.

```bash
# Uninstall any old version of llama-cpp-python
pip3 uninstall llama-cpp-python -y

# Linux Target with Nvidia CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install llama-cpp-python==0.2.27 --no-cache-dir
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install llama-cpp-python[server]==0.2.27 --no-cache-dir

# MacOS Target with Apple Silicon M1/M2
CMAKE_ARGS="-DLLAMA_METAL=on" pip3 install -U llama-cpp-python --no-cache-dir
pip3 install 'llama-cpp-python[server]'

# Download Models from HuggingFace
cd llmserver/models

# Get the Mistral 7B GGUF Q-5bit model Q5_K_M and Meta LLaMA-2 7B GGUF Q-5bit model Q5_K_M
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf
wget https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf

```

## Run Server

```bash
# Run Test - API Server
python3 -m llama_cpp.server \
    --model ./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf \
    --host localhost \
    --n_gpu_layers 99 \
    --n_ctx 2048 \
    --chat_format llama-2

```

## Setup as Service - Optional

You can set up a Linux service using the `tinyllm.service` file.  Make sure to edit `tinyllm` and `tinyllm.service` 
to have the username and paths required for your system.

```bash
# Edit the tinyllm.service to match your environment:
cd ..
nano tinyllm.service

# Edit
#   ExecStart - make sure path to python3 and  --n_gpu_layers (e.g. 32 if GPU VRAM is 6GB)
#   WorkingDirectory - this is the absolute path to the gguf file downloaded above
#   User - this is your local username

# Set up the service in systemd
sudo cp tinyllm.service /etc/systemd/system/
sudo cp tinyllm /etc/init.d
sudo /etc/init.d/tinyllm start
sudo /etc/init.d/tinyllm enable

# Check status and logs to make sure service is running
sudo /etc/init.d/tinyllm status

# Test server with a Command Line chat
python3 ../chat.py
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

## Optional - Docker Setup

You can also run the `setup-docker.sh` script.

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
    -v $PWD/models:/app/models \
    -e MODEL=models/llama-2-7b-chat.Q5_K_M.gguf \
    -e N_GPU_LAYERS=32 \
    -e HOST=0.0.0.0 \
    -e PORT=8000 \
    --name llmserver \
    --restart unless-stopped \
    llmserver
```

## LLM Models

Here are some suggested models that work well with TinyLLM. You can test other models and different quantization, but in my experiments, the Q5_K_M models performed the best. Below are the download links from HuggingFace as well as the model card's suggested context length size and chat prompt mode.

| LLM | Quantized | Link to Download | Context Length | Chat Prompt Mode |
| --- | --- | --- | --- | --- |
|  |  | 7B Models |  |  |
| Mistral v0.1 7B | 5-bit | [mistral-7b-instruct-v0.1.Q5_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf) | 4096 | llama-2 |
| Llama-2 7B | 5-bit | [llama-2-7b-chat.Q5_K_M.gguf](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf) | 2048 | llama-2 |
| Mistrallite 32K 7B | 5-bit | [mistrallite.Q5_K_M.gguf](https://huggingface.co/TheBloke/MistralLite-7B-GGUF/resolve/main/mistrallite.Q5_K_M.gguf) | 16384 | mistrallite (can be glitchy) |
|  |  | 10B Models |  |  |
| Nous-Hermes-2-SOLAR 10.7B | 5-bit | [nous-hermes-2-solar-10.7b.Q5_K_M.gguf](https://huggingface.co/TheBloke/Nous-Hermes-2-SOLAR-10.7B-GGUF/resolve/main/nous-hermes-2-solar-10.7b.Q5_K_M.gguf) | 4096 | chatml |
|  |  | 13B Models |  |  |
| Claude2 trained Alpaca 13B | 5-bit | [claude2-alpaca-13b.Q5_K_M.gguf](https://huggingface.co/TheBloke/claude2-alpaca-13B-GGUF/resolve/main/claude2-alpaca-13b.Q5_K_M.gguf) | 2048 | chatml |
| Llama-2 13B | 5-bit | [llama-2-13b-chat.Q5_K_M.gguf](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf) | 2048 | llama-2 |
| Vicuna 13B v1.5| 5-bit | [vicuna-13b-v1.5.Q5_K_M.gguf](https://huggingface.co/TheBloke/vicuna-13B-v1.5-GGUF/resolve/main/vicuna-13b-v1.5.Q5_K_M.gguf) | 2048 | vicuna |
|  |  | Mixture-of-Experts (MoE) Models |  |  |
| Hai's Mixtral 11Bx2 MoE 19B | 5-bit | [mixtral_11bx2_moe_19b.Q5_K_M.gguf](https://huggingface.co/TheBloke/Mixtral_11Bx2_MoE_19B-GGUF/resolve/main/mixtral_11bx2_moe_19b.Q5_K_M.gguf) | 4096 | chatml |
| Mixtral-8x7B v0.1 | 3-bit | [Mixtral-8x7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf) | 4096 | llama-2 |
| Mixtral-8x7B v0.1 | 4-bit | [Mixtral-8x7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf) | 4096 | llama-2 |

## Chat Formats

The llama.cpp server will support chat formats for various models. It is important to select the right one as each model was trained to respond to a particular format. If the model doesn't seem to work or chat seems to be filled with random dialogue that you didn't insert, you probably have the wrong model. The chat format is specified with the `--chat_format chatml` parameter in the startup command. Here are the current valid chat formats for llama_cpp:

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

Example Run

```bash
# Run Test - API Server - see run.sh
python3 -m llama_cpp.server \
    --model ./models/llama-2-7b-chat.Q5_K_M.gguf \
    --host localhost \
    --n_gpu_layers 32 \
    --n_ctx 2048 \
    --chat_format llama-2
```