#!/bin/bash

docker run \
    -d \
    -p 5000:5000 \
    -e PORT=5000 \
    -e OPENAI_API_BASE="http://localhost:8000/v1" \
    -e QDRANT_HOST="" \
    -e DEVICE="cuda" \
    -e RESULTS=1 \
    -v prompts.json:/app/prompts.json \
    --name chatbot \
    --restart unless-stopped \
    --net=host \
    chatbot