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

# TinyLLM Dependencies
from app.core.config import PORT, log, debug
from app.api.routes import app 
