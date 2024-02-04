#!/bin/bash
# 
# Example of running the chatbot container

echo "Starting chatbot container..."

docker run \
    -d \
    -p 5000:5000 \
    -e PORT=5000 \
    -e OPENAI_API_BASE="http://localhost:8000/v1" \
    -e LLM_MODEL="tinyllm" \
    -e QDRANT_HOST="localhost" \
    -e DEVICE="cpu" \
    -e USE_SYSTEM="false" \
    -v prompts.json:/app/prompts.json \
    --name chatbot \
    --restart unless-stopped \
    --net=host \
    chatbot

echo "Chatbot container started."

# Logs
echo "Viewing chatbot container logs... ^C to exit"
echo ""
docker logs -f chatbot
```