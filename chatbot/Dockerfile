# Dockerfile for chatbot - OpenAI API ChatBot
#
#   Author: Jason A. Cox
#   23 Sept 2023
#   https://github.com/jasonacox/TinyLLM

# Use a base image
FROM python:3.10

# Setting build related env vars
ENV PORT=5000
ENV OPENAI_API_KEY="DEFAULT_API_KEY"
ENV OPENAI_API_BASE="http://localhost:8000/v1"
ENV AGENT_NAME="Jarvis"
ENV MY_MODEL="models/7B/gguf-model.bin"
ENV DEBUG="False"
ENV DEVICE="cuda"
ENV STMODEL="all-MiniLM-L6-v2"
ENV QDRANT_HOST=""
ENV RESULTS=1

# Set the working directory
WORKDIR /app

# Copy local files into container
COPY . .

# Install depencencies
RUN pip install openai flask flask-socketio bs4
RUN pip install torch~=2.0.1 qdrant-client sentence-transformers~=2.2.2 pydantic~=2.4.2

# Run the server
CMD ["python3", "server.py"]

# Network
EXPOSE $PORT

# Example docker run:
# docker run \
#     -d \
#     -p 5000:5000 \
#     --name chatbot \
#     --restart unless-stopped \
#     chatbot