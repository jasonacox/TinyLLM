#!/bin/bash
#
# Test TinyLLM Chatbot via Command Line
#
# Author: Jason A. Cox
# 11 Feb 2024
# https://github.com/jasonacox/TinyLLM

# Set the defaults
export PORT=5555
export DEVICE=cpu
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="Asimov-3-Laws"
export QDRANT_HOST="localhost"
export MAXCLIENTS=16384
export TEMPERATURE=0.0

# Uncomment and add your KEY to test against OpenAI - gpt-3.5-turbo
#export LLM_MODEL=gpt-3.5-turbo 
#export OPENAI_API_KEY=YOUR_OPENAI_KEY 
#export OPENAI_API_BASE=https://api.openai.com/v1 
#export MAXTOKENS=4000

# Start the server
echo "Starting TinyLLM Chatbot Server... ^C to exit"
uvicorn server:app --port $PORT
