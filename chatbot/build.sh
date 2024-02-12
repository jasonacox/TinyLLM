#!/bin/bash
# Build chatbot container

# Get version of chatbot
string=`grep "VERSION =" server.py`
version=$(echo $string | awk '{print $NF}' | sed 's/v//' | sed 's/"//g')

# Create Container
docker build -t chatbot:$version .

# Done
echo "Built: chatbot:$version"
