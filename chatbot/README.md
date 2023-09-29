# Web Based Chatbot

This is a web based python flask app that allows you to chat with a LLM using the OpenAI API. To run the web server:

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
pip install openai flask flask-socketio

# Edit the server.py to adjust the global settings for your environment (e.g. openai.api_base)
# Run the chatbot web server
python3 server.py
```

## Example Session

Open http://127.0.0.1:5000 - Example session:

<img width="946" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/08097e39-9c00-4f75-8c9a-d329c886b148">

## Read URL

If a URL is pasted in the text box, the chatbot will read and summarize it.

<img width="810" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/44d8a2f7-54c1-4b1c-8471-fdf13439be3b">

