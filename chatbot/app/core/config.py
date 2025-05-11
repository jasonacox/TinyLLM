"""
ChatBot - Configuration Settings for the TinyLLM Chatbot.

It includes settings for the OpenAI API, LiteLLM Proxy, Weaviate, and other options.
It also includes settings for the chatbot's behavior, such as temperature, max tokens, and more.

Author: Jason A. Cox
20 Apr 2025
github.com/jasonacox/TinyLLM
"""

VERSION = "v0.16.1"  # Version of the TinyLLM (Major, Minor, Patch)

# Imports
import os
import json
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("TinyLLM %s" % VERSION)

def log(text):
    logger.info(text)

def debug(text):
    logger.debug(text)

# Configuration Settings

# Core Language Model Settings
API_KEY = os.environ.get("OPENAI_API_KEY", "Asimov-3-Laws")                 # Required, use bogus string for local LLMs
API_BASE = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")    # Required, use https://api.openai.com for OpenAI
MYMODEL = os.environ.get("LLM_MODEL", "models/7B/gguf-model.bin")           # Pick model to use e.g. gpt-3.5-turbo for OpenAI
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))                     # LLM temperature
USE_SYSTEM = os.environ.get("USE_SYSTEM", "false").lower() == "true"        # Use system in chat prompt if True
EXTRA_BODY = os.environ.get("EXTRA_BODY", None)                             # Extra body parameters for OpenAI API

# LiteLLM Proxy Settings
LITELLM_PROXY = os.environ.get("LITELLM_PROXY", None)                       # Optional - LITELLM Proxy URL
LITELLM_KEY = os.environ.get("LITELLM_KEY", "")                             # Optional - LITELLM Secret Key - Begins with sk-

# Chatbot Server Configuration
PORT = int(os.environ.get("PORT", 5000))                                    # Port to listen on
MAXCLIENTS = int(os.environ.get("MAXCLIENTS", 1000))                        # Maximum number of concurrent clients
TOKEN = os.environ.get("TOKEN", "secret")                                   # Secret TinyLLM token for admin functions
MAXTOKENS = int(os.environ.get("MAXTOKENS", 0))                             # Maximum number of tokens to send to LLM for RAG

# Chatbot Behavior Settings
AGENTNAME = os.environ.get("AGENT_NAME", "")                                # Set the name of your bot
ONESHOT = os.environ.get("ONESHOT", "false").lower() == "true"              # Set to True to enable one-shot mode
RAG_ONLY = os.environ.get("RAG_ONLY", "false").lower() == "true"            # Set to True to enable RAG only mode
THINKING = os.environ.get("THINKING", "false").lower() == "true"            # Set to True to enable thinking mode by default
THINK_FILTER = os.environ.get("THINK_FILTER", "false").lower() == "true"    # Set to True to enable thinking filter
TOXIC_THRESHOLD = float(os.environ.get("TOXIC_THRESHOLD", 99))              # Toxicity threshold for responses 0-1 or 99 disable
INTENT_ROUTER = os.environ.get("INTENT_ROUTER", "false").lower() == "true"  # Set to True to enable intent detection & routing
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", 1))                           # Maximum number of images to keep in context

# Prompt Management
PROMPT_FILE = os.environ.get("PROMPT_FILE", ".tinyllm/prompts.json")        # File to store system prompts
PROMPT_RO = os.environ.get("PROMPT_RO", "false").lower() == "true"          # Set to True to enable read-only prompts

# Web Search Integration
SEARXNG = os.environ.get("SEARXNG", "http://localhost:8080")                # SearXNG URL for internet search engine  (Optional)
WEB_SEARCH = os.environ.get("WEB_SEARCH", "false").lower() == "true"        # Set to True to enable web search for all queries

# Image Generation Settings
SWARMUI = os.environ.get("SWARMUI", "http://localhost:7801")                # Set to SwarmUI host URL, eg. http://localhost:7801
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", 
                           "OfficialStableDiffusion/sd_xl_base_1.0")        # Image model to use
IMAGE_CFGSCALE = float(os.environ.get("IMAGE_CFGSCALE", 7.5))               # CFG Scale for image generation (7 for SDXL, 1 for Flux)
IMAGE_STEPS = int(os.environ.get("IMAGE_STEPS", 20))                        # Steps for image generation (20 for SDXL, 4 for Flux)
IMAGE_SEED = int(os.environ.get("IMAGE_SEED", -1))                          # Seed for image generation
IMAGE_TIMEOUT = int(os.environ.get("IMAGE_TIMEOUT", 300))                   # Timeout for image generation (seconds)
IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH", 1024))                      # Width for image generation
IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT", 1024))                    # Height for image generation

# Debug Settings
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"                  # Set to True to enable debug mode

# Convert EXTRA_BODY to dictionary if it is proper JSON
if EXTRA_BODY:
    try:
        EXTRA_BODY = json.loads(EXTRA_BODY)
    except:
        log("EXTRA_BODY is not valid JSON")
        EXTRA_BODY = {}
else:
    if API_BASE.startswith("https://api.openai.com") or LITELLM_PROXY:
        EXTRA_BODY = {}
    else:
        # Extra stop tokens are needed for some non-OpenAI LLMs
        EXTRA_BODY = {"stop_token_ids":[128001, 128009]}

# LiteLLM Proxy
if LITELLM_PROXY:
    log(f"Using LiteLLM Proxy at {LITELLM_PROXY}")
    API_BASE = LITELLM_PROXY
    API_KEY = LITELLM_KEY

# RAG Configuration Settings
WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "")                         # Empty = no Weaviate support
WEAVIATE_GRPC_HOST = os.environ.get("WEAVIATE_GRPC_HOST", WEAVIATE_HOST)    # Empty = no Weaviate gRPC support
WEAVIATE_PORT = os.getenv('WEAVIATE_PORT', '8080')
WEAVIATE_GRPC_PORT = os.getenv('WEAVIATE_GRPC_PORT', '50051')
WEAVIATE_LIBRARY = os.environ.get("WEAVIATE_LIBRARY", "tinyllm")            # Weaviate library to use
WEAVIATE_AUTH_KEY = os.getenv('WEAVIATE_AUTH_KEY', None)                    # Weaviate Auth Key
RESULTS = int(os.environ.get("RESULTS", 1))                                 # Number of results to return from RAG query
ALPHA_KEY = os.environ.get("ALPHA_KEY", "alpha_key")                        # Optional - Alpha Vantage API Key
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/tmp")                     # Folder to store uploaded documents

# Debug Mode
if DEBUG:
    logger.setLevel(logging.DEBUG)
    log("Debug mode enabled.")
    # Display all default settings
    debug("Configuration Settings:")
    vars = globals()
    for n in list(vars):
        if n.isupper():
            if vars[n] and n in ["API_KEY", "TOKEN", "WEAVIATE_AUTH_KEY", "ALPHA_KEY"]:
                debug(f"   {n}: {'*' * len(vars[n])}")
            else:
                debug(f"   {n}: {vars[n]}")

# Log ONE_SHOT mode
if ONESHOT:
    log("ONESHOT mode enabled.")

# Globals
client = {}
prompts = {}
baseprompt = ""
stats = {
    "start_time": time.time(),
    "errors": 0,
    "ask": 0,
    "ask_llm": 0,
    "ask_context": 0,
}
llm_stream = None
