# TinyLLM Web Based Lab

This simple FastAPI app creates a UI to allow you to interact with an LLM and a Weaviate vector database.

## Command Line

```bash
# Install required packages
pip install fastapi uvicorn python-socketio jinja2 openai bs4 pypdf requests lxml aiohttp weaviate-client

# Run the chatbot web server - change the base URL to be where you host your llmserver
WEAVIATE_HOST="10.0.1.89" OPENAI_API_BASE="http://localhost:8000/v1" uvicorn server:app --port 8080
```
