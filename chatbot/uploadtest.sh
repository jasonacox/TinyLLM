#!/bin/bash
echo "** TEST ** Build and Push jasonacox/chatbot to Docker Hub"
echo ""

# Get version of chatbot
string=`grep "VERSION =" server.py`
VER=$(echo $string | awk '{print $NF}' | sed 's/v//' | sed 's/"//g')

# Build non-qdrant version
cp Dockerfile.basic Dockerfile
# Build jasonacox/chatbot:x.y.z
echo "* BUILD jasonacox/chatbot:${VER} - BASIC VERSION (no qdrant)"
docker buildx build --no-cache --platform linux/amd64,linux/arm64 --push -t jasonacox/chatbot:${VER} .
echo ""

# Build RAG version
cp Dockerfile.rag Dockerfile
# Build jasonacox/chatbot:x.y.z
echo "* BUILD jasonacox/chatbot:${VER}-rag - RAG VERSION (with qdrant)"
docker buildx build --no-cache --platform linux/amd64,linux/arm64 --push -t jasonacox/chatbot:${VER}-rag .
echo ""

# Verify
echo "* VERIFY jasonacox/chatbot:${VER}"
docker buildx imagetools inspect jasonacox/chatbot:${VER} | grep Platform
echo ""
echo "* VERIFY jasonacox/chatbot:${VER}-rag"
docker buildx imagetools inspect jasonacox/chatbot:${VER}-rag | grep Platform
echo ""

# Cleanup
cp Dockerfile.basic Dockerfile

