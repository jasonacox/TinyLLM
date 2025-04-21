#!/bin/bash
echo "** Build and Push TinyLLM chatbot and docman to Docker Hub"

# Get version
string=`grep "VERSION =" app/core/config.py`
if [ -z "$string" ]; then
    echo "VERSION not found in app/core/config.py"
    exit 1
fi
VER=$(echo $string | awk '{print $NF}' | sed 's/v//' | sed 's/"//g')
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

# If the user selects 1 jasonacox/chatbot or 3
if [ $container -eq 1 ] || [ $container -eq 3 ]; then
    echo "Build and Push jasonacox/chatbot to Docker Hub"
    echo ""

    # Build jasonacox/chatbot:x.y.z
    echo "* BUILD jasonacox/chatbot:${VER}"
    docker buildx build --no-cache --platform linux/amd64,linux/arm64 --push -t jasonacox/chatbot:${VER} .
    echo ""
    # Build jasonacox/chatbot:latest
    echo "* BUILD jasonacox/chatbot:latest"
    docker buildx build --platform linux/amd64,linux/arm64 --push -t jasonacox/chatbot:latest .
    echo ""

    # Verify
    echo "* VERIFY jasonacox/chatbot:${VER}"
    docker buildx imagetools inspect jasonacox/chatbot:${VER} | grep Platform
    echo ""
    echo "* VERIFY jasonacox/chatbot:latest"
    docker buildx imagetools inspect jasonacox/chatbot | grep Platform
    echo ""
fi

if [ $container -eq 2 ] || [ $container -eq 3 ]; then
    echo "Build and Push jasonacox/docman to Docker Hub"
    echo ""

    # Build jasonacox/docman:x.y.z
    echo "* BUILD jasonacox/docman:${VER}"
    docker buildx build --no-cache --platform linux/amd64,linux/arm64 --push -t jasonacox/docman:${VER} -f Dockerfile-docman .
    echo ""
    # Build jasonacox/docman:latest
    echo "* BUILD jasonacox/docman:latest"
    docker buildx build --platform linux/amd64,linux/arm64 --push -t jasonacox/docman:latest -f Dockerfile-docman .
    echo ""

    # Verify
    echo "* VERIFY jasonacox/docman:${VER}"
    docker buildx imagetools inspect jasonacox/docman:${VER} | grep Platform
    echo ""
    echo "* VERIFY jasonacox/docman:latest"
    docker buildx imagetools inspect jasonacox/docman | grep Platform
    echo ""
fi

# End
