#!/bin/bash
# 
# Run the chatbot container
#
# Author: Jason A. Cox
# 6 Feb 2024
# https://github.com/jasonacox/TinyLLM

# Check to see if container name exists already
EXISTING=$(docker ps -a --format '{{.Names}}' | grep chatbot)
if [ ! -z "$EXISTING" ]; then
    echo "Existing chatbot container found: $EXISTING"
    echo ""
    # Ask if we should stop and remove the existing container
    read -p "Would you like to stop and remove the existing container? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Check if container is in running state
        RUNNING=$(docker inspect --format="{{.State.Running}}" chatbot 2>/dev/null)
        if [ "$RUNNING" == "true" ]; then
            # Send restarting alert message to chatbot users
            curl -s -X POST -H "Content-Type: application/json" -d '{"token": "secret", "message": "Restarting"}' http://localhost:5000/alert > /dev/null 2>&1
            echo "Stopping chatbot container..."
            docker stop chatbot
        fi
        echo "Removing existing chatbot container..."
        docker rm chatbot
    else
        echo "Exiting..."
        exit 1
    fi
    echo ""
fi

# Start the chatbot container
echo "Starting chatbot container..."
if [ ! -d "./.tinyllm" ]; then
    echo "Creating .tinyllm directory..."
    mkdir .tinyllm
fi

docker run \
    -d \
    -p 5000:5000 \
    -e PORT=5000 \
    -e OPENAI_API_KEY="Asimov-3-Laws" \
    -e OPENAI_API_BASE="http://localhost:8000/v1" \
    -e LLM_MODEL="tinyllm" \
    -e USE_SYSTEM="false" \
    -e MAXCLIENTS=1000 \
    -e MAXTOKENS=16384 \
    -e TEMPERATURE=0.0 \
    -e QDRANT_HOST="localhost" \
    -v $PWD/.tinyllm:/app/.tinyllm \
    --network="host" \
    --name chatbot \
    --restart unless-stopped \
    jasonacox/chatbot
echo "Chatbot container started."
echo ""

# Show the logs
echo "Viewing chatbot container logs... ^C to exit"
echo ""
docker logs -f chatbot
