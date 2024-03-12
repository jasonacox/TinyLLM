#!/bin/bash
# 
# Run the TinyLLM Chatbot using OpenAI GPT-3.5
#
# Author: Jason A. Cox
# 11 Feb 2024
# https://github.com/jasonacox/TinyLLM

echo "Run TinyLLM Chatbot - OpenAI GPT-3.5"
echo "------------------------------------"

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
    -e OPENAI_API_KEY="YOUR_OPENAI_KEY" \
    -e OPENAI_API_BASE="https://api.openai.com/v1" \
    -e LLM_MODEL="gpt-3.5-turbo" \
    -e USE_SYSTEM="false" \
    -e MAXCLIENTS=1000 \
    -e MAXTOKENS=4000 \
    -e TEMPERATURE=0.0 \
    -e ONESHOT="false" \
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
