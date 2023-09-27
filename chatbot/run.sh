#!/bin/bash

docker run \
    -d \
    -p 5000:5000 \
    --name chatbot \
    --restart unless-stopped \
    chatbot
