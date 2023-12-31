#!/bin/bash

docker run \
    -d \
    -p 5000:5000 \
    -e QDRANT_HOST="" \
    -e DEVICE="cuda" \
    -e RESULTS=1 \
    -v prompts.json:/app/prompts.json \
    --name chatbot \
    --restart unless-stopped \
    chatbot
