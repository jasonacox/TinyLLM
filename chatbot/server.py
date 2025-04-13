#!/usr/bin/python3
"""
Web based ChatBot Example

Web chat client for OpenAI and the llama-cpp-python[server] OpenAI API Compatible 
Python FastAPI / socket.io based Web Server. Provides a simple web based chat session.

Features:
    * Uses OpenAI API to talk to LLM
    * Works with local hosted OpenAI API compatible LLMs, e.g. llama-cpp-python[server]
    * Retains conversational context for LLM
    * Uses response stream to render LLM chunks instead of waiting for full response
    * Supports multiple concurrent client sessions
    * Supports commands to reset context, get version, etc.
    * Uses FastAPI and Uvicorn ASGI high speed web server implementation
    * (Optional) Supports RAG prompts using Qdrant Vector Database

Requirements:
    * pip install fastapi uvicorn python-socketio jinja2 openai bs4 pypdf requests lxml aiohttp
    * pip install weaviate-client pdfreader pypandoc
    * pip install pandas openpyxl
    * pip install python-multipart
    * pip install pillow-heif

Environmental variables:
    * PORT - Port that Chatbot will listen on
    * PROMPT_FILE - File to store prompts
    * DEBUG - Set to True to enable debug mode
    * OPENAI_API_KEY - Required only for OpenAI
    * OPENAI_API_BASE - URL to OpenAI API Server or locally hosted version
    * LLM_MODEL - LLM Model to Use
    * USE_SYSTEM - Use system in chat prompt if True
    * MAXCLIENTS - Maximum number of clients to allow
    * MAXTOKENS - Maximum number of tokens to send to LLM
    * TEMPERATURE - LLM temperature
    * AGENT_NAME - Name for Bot
    * ALPHA_KEY - Alpha Vantage API Key for Stocks (Optional) - https://www.alphavantage.co/support/#api-key
    * WEAVIATE_HOST - Weaviate Host for RAG (Optional)
    * WEAVIATE_LIBRARY - Weaviate Library for RAG (Optional)
    * RESULTS - Number of results to return from RAG query
    * ONESHOT - Set to True to enable one-shot mode
    * RAG_ONLY - Set to True to enable RAG only mode
    * TOKEN - TinyLLM token for admin functions
    * PROMPT_FILE - File to store system prompts
    * PROMPT_RO - Set to True to enable read-only prompts
    * EXTRA_BODY - Extra body parameters for OpenAI API
    * TOXIC_THRESHOLD - Toxicity threshold for responses 0-1 or 99 disable
    * THINKING - Set to True to enable thinking mode by default
    * MAX_IMAGES - Maximum number of images the chatbot will keep in context (default 1)
    * SEARXNG - SearXNG URL for search engine (Optional)
    * INTENT_ROUTER - Set to True to enable intent detection an function calls.
    
Running a llama-cpp-python server:
    * CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
    * pip install llama-cpp-python[server]
    * python3 -m llama_cpp.server --model models/7B/ggml-model.bin

Web APIs:
    * GET / - Chatbot HTML main page
    * GET /upload - Chatbot Document Upload page
    * GET /version - Get version
    * POST /alert - Send alert to all clients

Author: Jason A. Cox
23 Sept 2023
https://github.com/jasonacox/TinyLLM

"""
# pylint: disable=invalid-name
# pylint: disable=global-statement
# pylint: disable=global-variable-not-assigned

# Import Libraries
import uvicorn
import sys

# TinyLLM Dependencies
from src.config import PORT, log
from src.routes import app

# Enable tracemalloc for memory usage
import tracemalloc
tracemalloc.start()

#
# Start dev server and listen for connections
#

if __name__ == '__main__':
    try:
        log(f"DEV MODE - Starting server on port {PORT}. Use uvicorn server:app for PROD mode.")
        config = uvicorn.Config(app, host="0.0.0.0", port=PORT)
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        log(f"Error starting server: {e}")
        sys.exit(1)
