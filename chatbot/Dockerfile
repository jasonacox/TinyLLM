# Dockerfile for chatbot - OpenAI API ChatBot
# - Full Version: WITH Vector Search Support for RAG
#
#   Author: Jason A. Cox
#   23 Sept 2023
#   https://github.com/jasonacox/TinyLLM

# Use a base image
FROM python:3.12-slim

# Setting build related env vars
ENV PORT=5000
ENV OPENAI_API_BASE="http://localhost:8000/v1"
ENV AGENT_NAME="Jarvis"
ENV MY_MODEL="models/7B/gguf-model.bin"
ENV DEBUG="false"
ENV WEAVIATE_HOST="localhost"
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
RUN pip install slowapi

# Document generation dependencies
RUN pip install reportlab python-docx openpyxl python-pptx

# Copy local files into container
COPY app /app/app
COPY run.py /app/run.py

# Network
EXPOSE $PORT

# Run the server
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
