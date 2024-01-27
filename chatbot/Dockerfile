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
ENV USE_SYSTEM="false"

# Set the working directory
WORKDIR /app

# Install depencencies
RUN pip install openai flask flask-socketio bs4 requests
RUN pip install torch qdrant-client sentence-transformers pydantic~=2.4.2 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install pypdf lxml

# Install depencencies - CPU only
# RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# RUN pip install openai flask flask-socketio bs4 qdrant-client requests
# RUN pip install transformers tqdm numpy scikit-learn scipy nltk sentencepiece
# RUN pip install --no-deps sentence-transformers
# RUN pip install pydantic~=2.4.2
# RUN pip install pypdf

# Copy local files into container
COPY . .

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