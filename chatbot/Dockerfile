# Dockerfile for chatbot - OpenAI API ChatBot
#
#   Author: Jason A. Cox
#   23 Sept 2023
#   https://github.com/jasonacox/TinyLLM

# Use a base image
FROM python:3.10-alpine

# Setting build related env vars
ENV PORT=5000
ENV OPENAI_API_KEY="DEFAULT_API_KEY"
ENV OPENAI_API_BASE="http://localhost:8000/v1"
ENV AGENT_NAME="Jarvis"
ENV MY_MODEL="models/7B/gguf-model.bin"
ENV DEBUG="False"

# Set the working directory
WORKDIR /app

# Copy local files into container
COPY server.py .

# Install depencencies
RUN pip install openai flask flask-socketio

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