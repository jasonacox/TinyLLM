# Dockerfile for chatbot - OpenAI API ChatBot
# - Full Version: WITH Vector Search Support for RAG
#
#   Author: Jason A. Cox
#   23 Sept 2023
#   https://github.com/jasonacox/TinyLLM

# Use a base image
FROM python:3.10-slim

# Setting build related env vars
ENV PORT 5000
ENV OPENAI_API_KEY "DEFAULT_API_KEY"
ENV OPENAI_API_BASE "http://localhost:8000/v1"
ENV AGENT_NAME "Jarvis"
ENV MY_MODEL "models/7B/gguf-model.bin"
ENV DEBUG "false"
ENV DEVICE "cpu"
ENV STMODEL "all-MiniLM-L6-v2"
ENV WEAVIATE_HOST "localhsot"
ENV WEAVIATE_LIBRARY "tinyllm"
ENV RESULTS 1
ENV ONESHOT "true"
ENV USE_SYSTEM "false"

# Set the working directory
WORKDIR /app

# Install depencencies - Mini Install - Weaviate Vector Search
RUN pip install fastapi uvicorn python-socketio jinja2 openai bs4 pypdf requests lxml aiohttp
RUN pip install weaviate-client

# Copy local files into container
COPY server.py /app/server.py
COPY templates /app/templates

# Network
EXPOSE $PORT

# Run the server
CMD uvicorn server:app --host 0.0.0.0 --port $PORT
