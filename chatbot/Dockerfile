# Dockerfile for chatbot - OpenAI API ChatBot
# - Full Version: WITH Vector Search Support for RAG
#
#   Author: Jason A. Cox
#   23 Sept 2023
#   https://github.com/jasonacox/TinyLLM

# Use a base image
FROM python:3.10-slim

# Setting build related env vars
ENV PORT=5000
ENV OPENAI_API_KEY="DEFAULT_API_KEY"
ENV OPENAI_API_BASE="http://localhost:8000/v1"
ENV AGENT_NAME="Jarvis"
ENV MY_MODEL="models/7B/gguf-model.bin"
ENV DEBUG="false"
ENV WEAVIATE_HOST="localhsot"
ENV WEAVIATE_PORT="8080"
ENV WEAVIATE_GRPC_HOST="localhost"
ENV WEAVIATE_GRPC_PORT="50051"
ENV WEAVIATE_LIBRARY="tinyllm"
ENV RESULTS=1
ENV ONESHOT="false"
ENV RAG_ONLY="false"
ENV USE_SYSTEM="false"

# Set the working directory
WORKDIR /app

# Install depencencies - Weaviate Vector Search
RUN pip install fastapi uvicorn python-socketio jinja2 openai bs4 pypdf requests lxml aiohttp
RUN pip install weaviate-client pdfreader pypandoc
RUN pip install pandas openpyxl
RUN pip install python-multipart
RUN pip install pillow-heif

# Copy local files into container
COPY server.py /app/server.py
COPY templates /app/templates
COPY documents.py /app/documents.py
COPY version.py /app/version.py

# Network
EXPOSE $PORT

# Run the server
CMD uvicorn server:app --host 0.0.0.0 --port $PORT
