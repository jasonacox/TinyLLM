#!/bin/bash

# Set up RAG vector database - QDRANT
# Qdrant (read: quadrant) is a vector similarity search engine and vector database.

#docker run --name qdrant -p 6333:6333 qdrant/qdrant

echo "Setting up qdrant in $PWD..."
mkdir -p $PWD/storage
mkdir -p $PWD/config
if [ ! -f $PWD/config/config.yaml ]; then 
	curl -o $PWD/config/config.yaml https://raw.githubusercontent.com/qdrant/qdrant/master/config/config.yaml
fi

echo "Starting container qdrant..."
docker run -p 6333:6333 \
    -d \
    --name qdrant \
    -v $PWD/storage:/qdrant/storage \
    -v $PWD/config/config.yaml:/qdrant/config/production.yaml \
    qdrant/qdrant

echo "Dashboard available at http://localhost:6333/dashboard"