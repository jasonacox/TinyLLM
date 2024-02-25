#!/bin/bash
echo "** TEST ** Build and Push jasonacox/chatbot to Docker Hub"
echo ""

# Get version of chatbot
string=`grep "VERSION =" server.py`
VER=$(echo $string | awk '{print $NF}' | sed 's/v//' | sed 's/"//g')
echo "Chatbot Version: $VER"
echo ""

# Build jasonacox/chatbot:x.y.z
echo "* BUILD jasonacox/chatbot:${VER}"
docker buildx build --no-cache --platform linux/amd64,linux/arm64 --push -t jasonacox/chatbot:${VER} .
echo ""

# Verify
echo "* VERIFY jasonacox/chatbot:${VER}"
docker buildx imagetools inspect jasonacox/chatbot:${VER} | grep Platform
echo ""
