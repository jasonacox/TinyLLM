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

# Standard library imports
import asyncio
import sys
import time
import traceback

# Third-party imports
import uvicorn

# Local imports
from app.api.routes import app  # The FastAPI app instance
from app.core.config import PORT, log, debug
from app.core.llm import init_llm

def start_server(max_retries: int = 3, retry_delay: int = 5):
    for attempt in range(max_retries):
        try:
            log(f"Starting server on port {PORT} (attempt {attempt + 1}/{max_retries})")
            config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="info")
            return uvicorn.Server(config)
        except Exception as e:
            log(f"Error starting server: {str(e)}")
            debug(f"Traceback:\n{traceback.format_exc()}")
            if attempt < max_retries - 1:
                log(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                log("Max retries reached. Server failed to start.")
                return None
    return None

if __name__ == '__main__':
    try:
        log(f"DEV MODE - Starting server on port {PORT}. Use uvicorn for PROD mode.")
        # Initialize LLM connection
        asyncio.run(init_llm())
        server = start_server()
        if server:
            server.run()
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        log("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        log(f"Fatal error: {e}")
        debug(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)
