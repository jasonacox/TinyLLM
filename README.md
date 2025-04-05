# TinyLLM

TinyLLM? Yes, the name is a bit of a contradiction, but it means well. It's all about putting a large language model (LLM) on a tiny system that still delivers acceptable performance.

This project helps you build a small locally hosted LLM with a ChatGPT-like web interface using consumer grade hardware. To read more about my research with llama.cpp and LLMs, see [research.md](research.md).

## Table of Contents

- [Key Features](#key-features)
- [Hardware Requirements](#hardware-requirements)
- [Manual Setup](#manual-setup)
- [Run a Local LLM](#run-a-local-llm)
  - [Ollama Server (Option 1)](#ollama-server-option-1)
  - [vLLM Server (Option 2)](#vllm-server-option-2)
  - [Llama-cpp-python Server (Option 3)](#llama-cpp-python-server-option-3)
- [Run a Chatbot](#run-a-chatbot)
  - [Example Session](#example-session)
  - [Read URLs](#read-urls)
  - [Current News](#current-news)
  - [Manual Setup](#manual-setup-1)
- [LLM Models](#llm-models)
- [LLM Tools](#llm-tools)
- [References](#references)

## Key Features

* Supports multiple LLMs (see list below)
* Builds a local OpenAI API web service via [Ollama](https://ollama.com/), [llama.cpp](https://github.com/ggerganov/llama.cpp) or [vLLM](https://github.com/vllm-project/vllm). 
* Serves up a Chatbot web interface with customizable prompts, accessing external websites (URLs), vector databases and other sources (e.g. news, stocks, weather).

## Hardware Requirements

* CPU: Intel, AMD or Apple Silicon
* Memory: 8GB+ DDR4
* Disk: 128G+ SSD
* GPU: NVIDIA (e.g. GTX 1060 6GB, RTX 3090 24GB) or Apple M1/M2
* OS: Ubuntu Linux, MacOS
* Software: Python 3, CUDA Version: 12.2

## Quickstart

TODO - Quick start setup script.

## Manual Setup

```bash
# Clone the project
git clone https://github.com/jasonacox/TinyLLM.git
cd TinyLLM
```

## Run a Local LLM

To run a local LLM, you will need an inference server for the model. This project recommends these options: [vLLM](https://github.com/vllm-project/vllm), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), and [Ollama](https://ollama.com/). All of these provide a built-in OpenAI API compatible web server that will make it easier for you to integrate with other tools.  

### Ollama Server (Option 1)

The Ollama project has made it super easy to install and run LLMs on a variety of systems (MacOS, Linux, Windows) with limited hardware. It serves up an OpenAI compatible API as well. The underlying LLM engine is llama.cpp. Like llama.cpp, the downside with this server is that it can only handle one session/prompt at a time. To run the Ollama server container:

```bash
# Install and run Ollama server
docker run -d --gpus=all \
    -v $PWD/ollama:/root/.ollama \
    -p 11434:11434 \
    -p 8000:11434 \
    --restart unless-stopped \
    --name ollama \
    ollama/ollama

# Download and test run the llama3 model
docker exec -it ollama ollama run llama3

# Tell server to keep model loaded in GPU
curl http://localhost:11434/api/generate -d '{"model": "llama3", "keep_alive": -1}'
```
Ollama support several models (LLMs): https://ollama.com/library If you set up the docker container mentioned above, you can down and run them using:

```bash
# Download and run Phi-3 Mini, open model by Microsoft.
docker exec -it ollama ollama run phi3

# Download and run mistral 7B model, by Mistral AI
docker exec -it ollama ollama run mistral
```

If you use the TinyLLM Chatbot (see below) with Ollama, make sure you specify the model via: `LLM_MODEL="llama3"` This will cause Ollama to download and run this model. It may take a while to start on first run unless you run one of the `ollama run` or `curl` commands above.

### vLLM Server (Option 2)

vLLM offers a robust OpenAI API compatible web server that supports multiple simultaneous inference threads (sessions). It automatically downloads the models you specify from HuggingFace and runs extremely well in containers. vLLM requires GPUs with more VRAM since it uses non-quantized models. AWQ models are also available and more optimizations are underway in the project to reduce the memory footprint. Note, for GPUs with a compute capability of 6 or less, Pascal architecture (see [GPU table](https://github.com/jasonacox/TinyLLM/tree/main/vllm#nvidia-gpu-and-torch-architecture)), follow details [here](./vllm/) instead.

```bash
# Build Container
cd vllm
./build.sh 

# Make a Directory to store Models
mkdir models

# Edit run.sh or run-awq.sh to pull the model you want to use. Mistral is set by default.
# Run the Container - This will download the model on the first run
./run.sh  

# The trailing logs will be displayed so you can see the progress. Use ^C to exit without
# stopping the container. 
```

### Llama-cpp-python Server (Option 3)

The llama-cpp-python's OpenAI API compatible web server is easy to set up and use. It runs optimized GGUF models that work well on many consumer grade GPUs with small amounts of VRAM. As with Ollama, a downside with this server is that it can only handle one session/prompt at a time. The steps below outline how to setup and run the server via command line. Read the details in [llmserver](./llmserver/) to see how to set it up as a persistent service or docker container on your Linux host.

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

# Run Test - API Server
python3 -m llama_cpp.server \
    --model ./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf \
    --host localhost \
    --n_gpu_layers 99 \
    --n_ctx 2048 \
    --chat_format llama-2
```

## Run a Chatbot

The TinyLLM Chatbot is a simple web based python FastAPI app that allows you to chat with an LLM using the OpenAI API. It supports multiple sessions and remembers your conversational history. Some RAG (Retrieval Augmented Generation) features including:

* Summarizing external websites and PDFs (paste a URL in chat window)
* List top 10 headlines from current news (use `/news`)
* Display company stock symbol and current stock price (use `/stock <company>`)
* Provide current weather conditions (use `/weather <location>`)
* Use a vector databases for RAG queries - see [RAG](rag) page for details

```bash
# Move to chatbot folder
cd ../chatbot
touch prompts.json

# Pull and run latest container - see run.sh
docker run \
    -d \
    -p 5000:5000 \
    -e PORT=5000 \
    -e OPENAI_API_BASE="http://localhost:8000/v1" \
    -e LLM_MODEL="tinyllm" \
    -e USE_SYSTEM="false" \
    -e SENTENCE_TRANSFORMERS_HOME=/app/.tinyllm \
    -v $PWD/.tinyllm:/app/.tinyllm \
    --name chatbot \
    --restart unless-stopped \
    jasonacox/chatbot
```

### Example Session

Open http://localhost:5000 - Example session:

<img width="930" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/9eef2769-a352-4cc9-9698-ce15e41c2c45">

### Read URLs

If a URL is pasted in the text box, the chatbot will read and summarize it.

<img width="810" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/44d8a2f7-54c1-4b1c-8471-fdf13439be3b">

### Current News

The `/news` command will fetch the latest news and have the LLM summarize the top ten headlines. It will store the raw feed in the context prompt to allow follow-up questions.

<img width="930" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/2732fe07-99ee-4795-a8ac-42d9a9712f6b">

### Manual Setup

You can also test the chatbot server without docker using the following.

```bash
# Install required packages
pip3 install fastapi uvicorn python-socketio jinja2 openai bs4 pypdf requests lxml aiohttp

# Run the chatbot web server
python3 server.py
```

## LLM Models

Here are some suggested models that work well with llmserver (llama-cpp-python). You can test other models and different quantization, but in my experiments, the Q5_K_M models performed the best. Below are the download links from HuggingFace as well as the model card's suggested context length size and chat prompt mode.

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

Here are some suggested models that work well with vLLM.

| LLM | Quantized | Link to Download | Context Length | License |
| --- | --- | --- | --- | --- |
| Mistral v0.1 7B | None | [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) | 32k | Apache 2 |
| Mistral v0.2 7B | None | [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | 32k | Apache 2 |
| Mistral v0.1 7B AWQ | AWQ | [TheBloke/Mistral-7B-Instruct-v0.1-AWQ](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-AWQ) | 32k | Apache 2 |
| Mixtral-8x7B | None | [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | 32k | Apache 2 |
| Pixtral-12B-2409 12B Vision | None | [mistralai/Pixtral-12B-2409](https://huggingface.co/mistralai/Pixtral-12B-2409) | 128k | Apache 2 |
| Meta Llama-3 8B | None | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | 8k | Meta |
| Meta Llama-3.2 11B Vision | FP8 | [neuralmagic/Llama-3.2-11B-Vision-Instruct-FP8-dynamic](https://huggingface.co/neuralmagic/Llama-3.2-11B-Vision-Instruct-FP8-dynamic) | 128k | Meta |
| Qwen-2.5 7B | None | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 128k | Apache 2 |
| Yi-1.5 9B | None | [01-ai/Yi-1.5-9B-Chat-16K](https://huggingface.co/01-ai/Yi-1.5-9B-Chat-16K) | 16k | Apache 2 |
| Phi-3 Small 7B | None | [microsoft/Phi-3-small-8k-instruct](https://huggingface.co/microsoft/Phi-3-small-8k-instruct) | 16k | MIT |
| Phi-3 Medium 14B | None | [microsoft/Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) | 4k | MIT |
| Phi-3.5 Vision 4B | None | [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) | 128k | MIT |
| Phi-4 14B | None | [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) | 16k | MIT |

## LLM Tools

### LLM

A CLI utility (`llm`) and Python library for interacting with Large Language Models. To configure this tool to use your local LLM's OpenAI API:

```bash
# Install llm command line tool
pipx install llm

# Location to store configuration files:
dirname "$(llm logs path)"
```

You define the model in the `extra-openai-models.yaml` file. Create this file in the directory discovered above. Edit the model_name and api_base to match your LLM OpenAI API setup:

```yaml
- model_id: tinyllm
  model_name: meta-llama/Meta-Llama-3.1-8B-Instruct
  api_base: "http://localhost:8000/v1"
```

```bash
# Configure llm to use your local model
llm models default tinyllm

# Test
llm "What is love?"
```

## References

* LLaMa.cpp - https://github.com/ggerganov/llama.cpp
* LLaMa-cpp-python - https://github.com/abetlen/llama-cpp-python
* vLLM - https://github.com/vllm-project/vllm
