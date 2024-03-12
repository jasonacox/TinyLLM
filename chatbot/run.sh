#!/bin/bash
# 
# Run the chatbot container
#
# Author: Jason A. Cox
# 6 Feb 2024
# https://github.com/jasonacox/TinyLLM

echo "Run TinyLLM Chatbot - Local Model"
echo "---------------------------------"

# Stop and remove the chatbot container
echo "Removing old chatbot container..."
docker stop chatbot
docker rm chatbot

# Start the chatbot container
echo "Starting new chatbot container..."
if [ ! -d "./.tinyllm" ]; then
    echo "Creating .tinyllm directory..."
    mkdir .tinyllm
fi

docker run \
    -d \
    -p 5000:5000 \
    -e PORT=5000 \
    -e OPENAI_API_KEY="Asimov-3-Laws" \
    -e OPENAI_API_BASE="http://localhost:8000/v1" \
    -e LLM_MODEL="tinyllm" \
    -e USE_SYSTEM="false" \
    -e MAXCLIENTS=1000 \
    -e MAXTOKENS=8192 \
    -e TEMPERATURE=0.0 \
    -e ONESHOT="false" \
    -e RAG_ONLY="false" \
    -e WEAVIATE_HOST="" \
    -e WEAVIATE_LIBRARY="tinyllm" \
    -e TZ="America/Los_Angeles" \
    -v $PWD/.tinyllm:/app/.tinyllm \
    --name chatbot \
    --restart unless-stopped \
    jasonacox/chatbot
echo "Chatbot container started."
echo ""

# Show the logs
echo "Viewing chatbot container logs... ^C to exit"
echo ""
docker logs -f chatbot
