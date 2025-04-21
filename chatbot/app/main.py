#!/usr/bin/python3
"""
Chatbot Server - Main entry point for the TinyLLM Chatbot.
It starts the FastAPI server and handles errors during startup.

Author: Jason A. Cox
23 Sept 2023
https://github.com/jasonacox/TinyLLM

"""
# pylint: disable=invalid-name
# pylint: disable=global-statement
# pylint: disable=global-variable-not-assigned
# pylint: disable=unused-import


# TinyLLM Dependencies
from app.core.config import PORT, log, debug
from app.api.routes import app
from app.core.llm import test_model

# Test LLM
async def main():
    await test_model()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
