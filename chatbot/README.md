# Web Based Chatbot

This is a web based python flask app that allows you to chat with a LLM using the OpenAI API. 

The intent of this project is to build and interact with a locally hosted LLM using consumer grade hardware. The examples below use a Llama 2 7B model served up with the OpenAI API compatible [llmserver](https://github.com/jasonacox/TinyLLM/tree/main/llmserver) on an Intel i5 systems with an Nvidia GeForce GTX 1060 GPU.

# Docker

```bash
# Build container
./build.sh

# Run container
./run.sh
```

# Manual

```bash
# Install required packages
pip install openai flask flask-socketio bs4

# Run the chatbot web server - change the base URL to be where you host your llmserver
OPENAI_API_BASE="http://localhost:8000/v1" python3 server.py
```

## Example Session

Open http://127.0.0.1:5000 - Example session:

<img width="946" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/08097e39-9c00-4f75-8c9a-d329c886b148">

## Read URL

If a URL is pasted in the text box, the chatbot will read and summarize it.

<img width="810" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/44d8a2f7-54c1-4b1c-8471-fdf13439be3b">

