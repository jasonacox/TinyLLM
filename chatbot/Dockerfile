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
ENV DEBUG "False"
ENV DEVICE "cpu"
ENV STMODEL "all-MiniLM-L6-v2"
ENV QDRANT_HOST ""
ENV RESULTS 1
ENV USE_SYSTEM "false"

# Set the working directory
WORKDIR /app

# Install depencencies - Mini Install - No Vector Search
RUN pip install fastapi uvicorn python-socketio jinja2 openai bs4 pypdf requests lxml aiohttp

# Add Suport for Vector Search - CPU Only - Comment out to disable vector search
# RUN pip install torch qdrant-client sentence-transformers --extra-index-url https://download.pytorch.org/whl/cpu

# Copy local files into container
COPY server.py /app/server.py
COPY templates /app/templates

# Network
EXPOSE $PORT

# Run the server
CMD uvicorn server:app --host 0.0.0.0 --port $PORT

# Example docker run command
# docker run \
#     -d \
#     -p 5000:5000 \
#     -e PORT=5000 \
#     -e OPENAI_API_BASE="http://localhost:8000/v1" \
#     -e LLM_MODEL="tinyllm" \
#     -e QDRANT_HOST="localhost" \
#     -e DEVICE="cpu" \
#     -e RESULTS=1 \
#     -e USE_SYSTEM="false" \
#     -e SENTENCE_TRANSFORMERS_HOME=/app/.tinyllm \
#     -v $PWD/.tinyllm:/app/.tinyllm \
#     --name chatbot \
#     --restart unless-stopped \
#     chatbot:0.12.0