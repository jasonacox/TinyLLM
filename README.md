# TinyLLM

TinyLLM? Yes, the name is a bit of a contradiction, but it means well. It's all about putting a large language model (LLM) on a tiny system that still delivers acceptable performance.

This project helps you build a small locally hosted LLM with a ChatGPT-like web interface using consumer grade hardware. To read more about my research with llama.cpp and LLMs, see [research.md](research.md).

## Key Features

* Supports multiple LLMs (see list below)
* Builds a local OpenAI API web service via llama-cpp-python. 
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

## Run Local LLM

To run a local LLM, you will need a server to run inference on the model. This project recommends two options: [vLLM](https://github.com/vllm-project/vllm) and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python). Both provide a built-in OpenAI API compatible server.  

### vLLM Server (Option 1)

vLLM supports multiple simultaneous inference threads (session), automatically downloads the model and runs well in containers. Follow the details below to run vLLM. Note, for GPUs with a compute capability of 6 or less (e.g. Pascal, GTX 1060) follow details [here](./vllm/) instead. 

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

### Llama-cpp-python Server (Option 2)

The llama-cpp-python server is simple and runs optimized GGUF quantized models. However, it can only handle one session/prompt at a time. The steps below create an instance of llama_cpp.server which serves up an LLM with an OpenAI API compatible interface.

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
    -n_ctx 2048 \
    --chat_format llama-2
```

## Run Chatbot

Chatbot is a simple web based python flask app that allows you to chat with an LLM using the OpenAI API. It supports multiple sessions and remembers your conversational history. Some RAG (Retrieval Augmented Generation) features including:

* Summarizing external websites (just paste a URL in chat window)
* Fetch current news (use `/news`) - see [chatbot](chatbot) page for more details.
* Use a vector databases for RAG queries - see [RAG](rag) page for details

```bash
# Move to chatbot folder
cd ../chatbot

# Build Docker container
docker build -t chatbot .

# Run container as a service on port 5000
docker run \
    -d \
    -p 5000:5000 \
    -e PORT=5000 \
    -e OPENAI_API_BASE="http://localhost:8000/v1" \
    -e LLM_MODEL="tinyllm" \
    -e QDRANT_HOST="" \
    -e DEVICE="cuda" \
    -e RESULTS=1 \
    -v prompts.json:/app/prompts.json \
    --name chatbot \
    --restart unless-stopped \
    --net=host \
    chatbot
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
pip install openai flask flask-socketio bs4

# Run the chatbot web server
python3 server.py
```

## LLM Models

Here are some suggested models that work well with vLLM.

| LLM | Quantized | Link to Download | Context Length |
| --- | --- | --- | --- |
| Mistral v0.1 7B | None | [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) | 32k |
| Mistral v0.2 7B | None | [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | 32k |
| Mistral v0.1 7B AWQ | AWQ | [TheBloke/Mistral-7B-Instruct-v0.1-AWQ](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-AWQ) | 32k |
| Mixtral-8x7B | None | [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | 32k |

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

## References

* LLaMa.cpp - https://github.com/ggerganov/llama.cpp
* LLaMa-cpp-python - https://github.com/abetlen/llama-cpp-python
