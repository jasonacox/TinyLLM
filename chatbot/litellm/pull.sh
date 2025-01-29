#!/bin/bash
#
# Script to help pull a model for the OLLAMA server
# running in a Docker container.

echo "Pull the model for the OLLAMA server."
echo ""

# Ask user to input the model name
if [ -z "$1" ]; then
    echo "Model to pull: "
    read -r model
    MODEL="$model"
else 
    MODEL="$1"
fi

# Check if the model name is not empty
if [ -n "$MODEL" ]; then
    # Ask user if they want to continue
    read -p "Are you sure you want to pull the model: $MODEL? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
    echo ""
    echo "Pulling the model: $MODEL..."
else
    echo "No model name provided. Exiting..."
    exit 1
fi

# Check if the Docker container is running
if ! docker ps | grep -q ollama; then
    echo "Docker container 'ollama' is not running. Exiting..."
    exit 1
fi

docker exec -it ollama bash -c "ollama pull \"$MODEL\""
if [ $? -eq 0 ]; then
    echo "Done."
else
    echo "Failed to pull the model. Exiting..."
    exit 1
fi

echo ""
