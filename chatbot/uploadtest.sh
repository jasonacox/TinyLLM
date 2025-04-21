#!/bin/bash
echo "** BETA Upload ** Build and Push TinyLLM chatbot and docman to Docker Hub"

BETA="beta${1}"

# Get version
string=`grep "VERSION =" app/core/config.py`
if [ -z "$string" ]; then
    echo "VERSION not found in app/core/config.py"
    exit 1
fi
VER=$(echo $string | awk '{print $NF}' | sed 's/v//' | sed 's/"//g')-$BETA
echo "** Version: $VER"
echo ""

# Ask for which container to build
echo "Which container do you want to build?"
echo "1. jasonacox/chatbot"
echo "2. jasonacox/docman"
echo "3. both"
echo ""
read -p "Enter selection: " container
echo ""

# If the user selects jasonacox/chatbot or 3
if [ $container -eq 1 ] || [ $container -eq 3 ]; then
    # Build jasonacox/chatbot:x.y.z
    echo "* BUILD jasonacox/chatbot:${VER}"
    docker buildx build --no-cache --platform linux/amd64,linux/arm64 --push -t jasonacox/chatbot:${VER} .
    echo ""

    # Verify
    echo "* VERIFY jasonacox/chatbot:${VER}"
    docker buildx imagetools inspect jasonacox/chatbot:${VER} | grep Platform
    echo ""
fi

if [ $container -eq 2 ] || [ $container -eq 3 ]; then
    # Build jasonacox/docman:x.y.z
    echo "* BUILD jasonacox/docman:${VER}"
    docker buildx build --no-cache --platform linux/amd64,linux/arm64 --push -t jasonacox/docman:${VER} -f Dockerfile-docman .
    echo ""

    # Verify
    echo "* VERIFY jasonacox/docman:${VER}"
    docker buildx imagetools inspect jasonacox/docman:${VER} | grep Platform
    echo ""
fi


