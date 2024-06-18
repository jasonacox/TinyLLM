#!/usr/bin/python3
"""
Web based LLM Lab Bench

Web based LLM Lab Bench is a web application that allows you to interact with a Language
Model (LLM) using an OpenAI API. The interface provides tools to test the LLM with RAG
prompts using different backend vector databases. The application uses FastAPI and Uvicorn
ASGI high speed web server implementation.

Features:
    * Uses OpenAI API to talk to LLM
    * Works with local hosted OpenAI API compatible LLMs, e.g. llama-cpp-python[server]
    * Uses response stream to render LLM chunks instead of waiting for full response
    * Supports multiple concurrent client sessions
    * Supports commands to reset context, get version, etc.
    * Uses FastAPI and Uvicorn ASGI high speed web server implementation
    * (Optional) Supports RAG prompts using Qdrant Vector Database

Requirements:
    * pip install fastapi uvicorn python-socketio jinja2 openai bs4 pypdf requests lxml aiohttp
    * pip install weaviate-client

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
    * WEAVIATE_HOST - Weaviate Host for RAG (Optional)
    * WEAVIATE_LIBRARY - Weaviate Library for RAG (Optional)
    * RESULTS - Number of results to return from RAG query

Author: Jason A. Cox
17 Jun 2024
https://github.com/jasonacox/TinyLLM

"""
# Import Libraries
import asyncio
import datetime
import io
import json
import logging
import os
import time

import openai
import requests
import socketio
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from pypdf import PdfReader
import aiohttp

# TinyLLM Version
VERSION = "v0.14.7"

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("TinyLLM %s" % VERSION)

def log(text):
    logger.info(text)

# Configuration Settings
api_key = os.environ.get("OPENAI_API_KEY", "open_api_key")                  # Required, use bogus string for Llama.cpp
api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")    # Required, use https://api.openai.com for OpenAI
agentname = os.environ.get("AGENT_NAME", "")                                # Set the name of your bot
mymodel = os.environ.get("LLM_MODEL", "models/7B/gguf-model.bin")           # Pick model to use e.g. gpt-3.5-turbo for OpenAI
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"                  # Set to True to enable debug mode
MAXCLIENTS = int(os.environ.get("MAXCLIENTS", 1000))                        # Maximum number of concurrent clients
MAXTOKENS = int(os.environ.get("MAXTOKENS", 16*1024))                       # Maximum number of tokens to send to LLM
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))                     # LLM temperature
PORT = int(os.environ.get("PORT", 5000))                                    # Port to listen on
PROMPT_FILE = os.environ.get("PROMPT_FILE", f".tinyllm/prompts.json")       # File to store system prompts
PROMPT_RO = os.environ.get("PROMPT_RO", "false").lower() == "true"          # Set to True to enable read-only prompts
USE_SYSTEM = os.environ.get("USE_SYSTEM", "false").lower == "true"          # Use system in chat prompt if True
TOKEN = os.environ.get("TOKEN", "secret")                                   # Secret TinyLLM token for admin functions
ONESHOT = os.environ.get("ONESHOT", "false").lower() == "true"              # Set to True to enable one-shot mode
RAG_ONLY = os.environ.get("RAG_ONLY", "false").lower() == "true"            # Set to True to enable RAG only mode

# RAG Configuration Settings
WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "")                         # Empty = no Weaviate support
WEAVIATE_LIBRARY = os.environ.get("WEAVIATE_LIBRARY", "tinyllm")            # Weaviate library to use
RESULTS = int(os.environ.get("RESULTS", 1))                                 # Number of results to return from RAG query
ALPHA_KEY = os.environ.get("ALPHA_KEY", "alpha_key")                        # Optional - Alpha Vantage API Key

# Prompt Defaults
default_prompts = {}
default_prompts["greeting"] = "Hi"
default_prompts["agentname"] = "Jarvis"
default_prompts["baseprompt"] = "You are TinyLLM, a highly intelligent assistant. The current date is {date}.\n\nYou should give concise responses to very simple questions, but provide thorough responses to more complex and open-ended questions."
default_prompts["weather"] = "You are a weather forecaster. Keep your answers brief and accurate. Current date is {date} and weather conditions:\n[DATA]{context_str}[/DATA]\nProvide a weather update, current weather alerts, conditions, precipitation and forecast for {location} and answer this: {prompt}."
default_prompts["stock"] = "You are a stock analyst. Keep your answers brief and accurate. Current date is {date}."
default_prompts["news"] = "You are a newscaster who specializes in providing headline news. Use only the following context provided by Google News to summarize the top 10 headlines for today. Rank headlines by most important to least important. Always include the news organization and ID. Do not add any commentary.\nAlways use this format:\n#. [News Item] - [News Source] - LnkID:[ID]\nHere are some examples: \n1. The World is Round - Science - LnkID:91\n2. The Election is over and Children have won - US News - LnkID:22\n3. Storms Hit the Southern Coast - ABC - LnkID:55\nContext: {context_str}\nTop 10 Headlines with Source and LnkID:"
default_prompts["clarify"] = "You are a highly intelligent assistant. Keep your answers brief and accurate. {format}."
default_prompts["location"] = "What location is specified in this prompt, state None if there isn't one. Use a single word answer. [BEGIN] {prompt} [END]"
default_prompts["company"] = "What company is related to the stock price in this prompt? Please state none if there isn't one. Use a single word answer: [BEGIN] {prompt} [END]"
default_prompts["rag"] = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Back up your answer using facts from the following context.\\nContext: {context_str}\\nQuestion: {prompt}\\nAnswer:"
default_prompts["website"] = "Summarize the following text from URL {url}:\n[BEGIN] {website_text} [END]\nThe above article is about:"
default_prompts["LLM_temperature"] = TEMPERATURE
default_prompts["LLM_max_tokens"] = MAXTOKENS

# Log ONE_SHOT mode
if ONESHOT:
    log("ONESHOT mode enabled.")

# Test OpenAI API
def test_model():
    global api_key, api_base, mymodel, MAXTOKENS
    log("Testing OpenAI API...")
    try:
        log(f"Using openai library version {openai.__version__}")
        log(f"Connecting to OpenAI API at {api_base} using model {mymodel}")
        llm = openai.OpenAI(api_key=api_key, base_url=api_base)
        # Get models
        try:
            models = llm.models.list()
            if len(models.data) == 0:
                log("LLM: No models available - proceeding.")
        except Exception as e:
            log(f"LLM: Unable to get models, using default: {str(e)}")
            models = mymodel
        else:
            # build list of models
            model_list = [model.id for model in models.data]
            log(f"LLM: Models available: {model_list}")
            if not mymodel in model_list:
                log(f"LLM: Model {mymodel} not found in models list.")
                if len(model_list) == 1:
                    log("LLM: Switching to default model")
                    mymodel = model_list[0]
                else:
                    log(f"LLM: Unable to find requested model {mymodel} in models list.")
                    raise Exception(f"Model {mymodel} not found")
        # Test LLM
        log(f"LLM: Using and testing model {mymodel}")
        llm.chat.completions.create(
            model=mymodel,
            max_tokens=MAXTOKENS,
            stream=False,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": "Hello"}],
            extra_body={"stop_token_ids":[128001, 128009]},
        )
        return True
    except Exception as e:
        log("OpenAI API Error: %s" % e)
        log(f"Unable to connect to OpenAI API at {api_base} using model {mymodel}.")
        if "maximum context length" in str(e):
            if MAXTOKENS > 1024:
                MAXTOKENS = int(MAXTOKENS / 2)
                log(f"LLM: Maximum context length exceeded reducing MAXTOKENS to {MAXTOKENS}.")
        return False

while True:
    if test_model():
        break
    else:
        log("Sleeping 5 seconds...")
        time.sleep(5)

# Test Weaviate Connection
if WEAVIATE_HOST != "":
    import weaviate
    import weaviate.classes as wvc
    try:
        client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=8080,
            grpc_port=50051,
            additional_config=weaviate.config.AdditionalConfig(timeout=(15, 115))
        )
        log(f"RAG: Connected to Weaviate at {WEAVIATE_HOST}")
        client.close()
    except Exception as e:
        log(f"RAG: Unable to connect to Weaviate at {WEAVIATE_HOST}: {str(e)}")
        WEAVIATE_HOST = "" # Disable RAG support
        log("RAG: RAG support disabled.")

# Find document closely related to query
def query_index(query, library, num_results=RESULTS):
    references = "References:"
    content = ""
    try:
        client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=8080,
            grpc_port=50051,
            additional_config=weaviate.config.AdditionalConfig(timeout=(15, 115))
        )
        hr = client.collections.get(library)
        results = hr.query.near_text(
            query=query,
            limit=num_results
        )
        for ans in results.objects:
            references = references + f"\n - {ans.properties['title']} - {ans.properties['file']}"
            content = content + f"Document: {ans.properties['title']}\nDocument Source: {ans.properties['file']}\nContent: {ans.properties['content']}\n---\n"
            if (len(content)/4) > MAXTOKENS/2:
                break
        client.close()
        log(f"RAG: Retrieved: {references}")
        return content, references
    except Exception as er:
        log(f"Error querying Weaviate: {str(er)}")
        return None, None

# Globals
client = {}
prompts = {}
stats = {
    "start_time": time.time(),
    "errors": 0,
    "ask": 0,
    "ask_llm": 0,
}

#
# Configure FastAPI App and SocketIO
#

log("Starting server...")
sio = socketio.AsyncServer(async_mode="asgi")
socket_app = socketio.ASGIApp(sio)
app = FastAPI()

# Load system prompts from PROMPT_FILE
def load_prompts():
    global prompts
    try:
        with open(PROMPT_FILE, "r") as f:
            prompts = json.load(f)
        # Ensure prompts always include all keys from default_prompts
        for k in default_prompts:
            if k not in prompts:
                prompts[k] = default_prompts[k]
    except:
        log(f"Unable to load system prompts file {PROMPT_FILE}, creating with defaults.")
        reset_prompts()
        save_prompts()

# Save prompts to PROMPT_FILE
def save_prompts():
    global prompts
    try:
        os.makedirs(os.path.dirname(PROMPT_FILE), exist_ok=True)  # Create path if it doesn't exist
        with open(PROMPT_FILE, "w") as f:
            json.dump(prompts, f)
            log(f"Saved {len(prompts)} prompts.")
    except:
        log("Unable to save prompts.")

# Expand variables in prompt to values
def expand_prompt(prompt, values):
    # Always use current {date} and {time}
    current_date = datetime.datetime.now()
    values["date"] = current_date.strftime("%B %-d, %Y")
    values["time"] = current_date.strftime("%-I:%M %p")
    for k in values:
        prompt = prompt.replace(f"{{{k}}}", values[k])
    return prompt

# Reset prompts
def reset_prompts():
    global prompts
    prompts = {}
    for k in default_prompts:
        prompts[k] = default_prompts[k]

# Load prompts
load_prompts()
log(f"Loaded {len(prompts)} prompts.")

# Function to return base conversation prompt
def base_prompt(content=None, template=None):
    global baseprompt, agentname, USE_SYSTEM, prompts
    if agentname == "":
        agentname = prompts["agentname"]
    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime("%B %-d, %Y")
    values = {"agentname": agentname, "date": formatted_date}
    if template:
        baseprompt = expand_prompt(template, values)
    else:
        baseprompt = expand_prompt(prompts["baseprompt"], values)
    if not content:
        content = baseprompt
    if USE_SYSTEM:
        return [{"role": "system", "content": content}] 
    else:
        return [{"role": "user", "content": content}, {"role": "assistant", "content": "Okay, let's get started."}] 

# Function - Send user prompt to LLM for response
async def ask(prompt, sid=None, ctx="context"):
    global client, stats
    stats["ask"] += 1
    response = False
    while not response:
        try:
            # remember context
            if ONESHOT:
                client[sid][ctx] = base_prompt()
            client[sid][ctx].append({"role": "user", "content": prompt})
            log(f"messages = {client[sid]['context']} - model = {mymodel}")
            llm = openai.OpenAI(api_key=api_key, base_url=api_base)
            response = llm.chat.completions.create(
                model=mymodel,
                max_tokens=MAXTOKENS,
                stream=True, # Send response chunks as LLM computes next tokens
                temperature=TEMPERATURE,
                messages=client[sid][ctx],
                extra_body={"stop_token_ids":[128001, 128009]},
            )
        except openai.OpenAIError as er:
            # If we get an error, try to recover
            client[sid][ctx].pop()
            if "does not exist" in str(er):
                await sio.emit('update', {'update': '[Model Unavailable... Retrying]', 'voice': 'user'},room=sid)
                log(f"Model does not exist - retrying")
                test_model()
                await sio.emit('update', {'update': mymodel, 'voice': 'model'})
            elif "maximum context length" in str(er):
                if len(prompt) > 1000:
                    # assume we have very large prompt - cut out the middle
                    prompt = prompt[:len(prompt)//4] + " ... " + prompt[-len(prompt)//4:]
                    log(f"Reduce prompt size - now ~{len(prompt)/4}")
                elif len(client[sid][ctx]) > 4:
                    # our context has grown too large, truncate the top
                    client[sid][ctx] = client[sid][ctx][:1] + client[sid][ctx][3:]
                    log(f"Truncate context: ~{len(client[sid]['context'])/4}")
                else:
                    # our context has grown too large, reset
                    client[sid][ctx] = base_prompt()   
                    log(f"Reset context ~{len(client[sid]['context'])/4}")
            else:
                log(f"ERROR: {er}")
                stats["errors"] += 1
                await sio.emit('update', {'update': e, 'voice': 'user'},room=sid)

    if not client[sid]["remember"]:
        client[sid]["remember"] =True
        client[sid][ctx].pop()
        client[sid][ctx].append({"role": "user", "content": "Help me remember."})
    return response

async def ask_llm(query, format=""):
    # Ask LLM a question
    global stats
    stats["ask_llm"] += 1
    if format == "":
        format = f"Respond in {format}."
    content = base_prompt(expand_prompt(prompts["clarify"], {"format": format})) + [{"role": "user",
                "content": query}]
    log(f"ask_llm: {content}")
    llm = openai.AsyncOpenAI(api_key=api_key, base_url=api_base)
    response = await llm.chat.completions.create(
        model=mymodel,
        max_tokens=MAXTOKENS,
        stream=False,
        temperature=TEMPERATURE,
        messages=content,
        extra_body={"stop_token_ids":[128001, 128009]},
    )
    log(f"ask_llm -> {response.choices[0].message.content.strip()}")
    return response.choices[0].message.content.strip()

# Function - Get weather for location
async def get_weather(location):
    # Look up weather for location
    if location == "":
        location = "Los Angeles"
    location = location.replace(" ", "+")
    url = "https://wttr.in/%s?format=j2" % location
    log(f"Fetching weather for {location} from {url}")
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return "Unable to fetch weather for %s" % location
    
# Function - Get stock price for company
async def get_stock(company):
    if ALPHA_KEY == "alpha_key":
        return "Unable to fetch stock price for %s - No Alpha Vantage API Key" % company
    # First try to get the ticker symbol
    symbol = await ask_llm(f"What is the stock symbol for {company}? Respond with symbol.")
    if "none" in symbol.lower():
        return "Unable to fetch stock price for %s - No matching symbol" % company
    # Check to see if response has multiple lines and if so, pick the first one
    symbol = symbol.split("\n")[0].strip()
    # Check to see if there are multiple words and if so, pick the last one
    if len(symbol.split()) > 1:
        symbol = symbol.split()[-1]
    # Strip off any spaces or non-alpha characters
    symbol = ''.join(e for e in symbol if e.isalnum())
    # Now get the stock price
    url = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=%s&apikey=%s" % (symbol.upper(), ALPHA_KEY)
    log(f"Fetching stock price for {company} from {url}")
    response = requests.get(url)
    if response.status_code == 200:
        try:
            price = response.json()["Global Quote"]["05. price"]
            return f"The price of {company} (symbol {symbol}) is ${price}."
        except:
            return "Unable to fetch stock price for %s - No data available." % company

# Function - Get news for topic
async def get_top_articles(url, max=10):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            soup = BeautifulSoup(await response.text(), 'xml')
            items = soup.findAll('item')
            articles = ""
            links = {}
            count = 1
            for item in items:
                title = item.find('title').string.strip()
                #pubdate = item.find('pubDate').string.strip()
                #description = item.find('description').string.strip()
                link = item.find('link').string.strip()
                links[f"LnkID:{count+100}"] = link
                articles += f"Headline: {title} - LnkID:{count+100}\n"
                count += 1
                if count > max:
                    break
            return articles, links

# Function - Fetch news for topic
async def get_news(topic, max=10):
    if "none" in topic.lower() or "current" in topic.lower():
        url = "https://news.google.com/rss/"
    else:
        topic = topic.replace(" ", "+")
        url = "https://news.google.com/rss/search?q=%s" % topic
    log(f"Fetching news for {topic} from {url}")
    async with aiohttp.ClientSession() as session:
        response, links = await get_top_articles(url, max)
        return response, links

# Function - Extract text from URL
async def extract_text_from_url(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, allow_redirects=True) as response:
                if response.status == 200:
                    # Route extraction based on content type
                    if ";" in response.headers["Content-Type"]:
                        content_type = response.headers["Content-Type"].split(";")[0]
                    else:
                        content_type = response.headers["Content-Type"]
                    content_handlers = {
                        "application/pdf": extract_text_from_pdf,
                        "text/plain": extract_text_from_text,
                        "text/csv": extract_text_from_text,
                        "text/xml": extract_text_from_text,
                        "application/json": extract_text_from_text,
                        "text/html": extract_text_from_html,
                        "application/xml": extract_text_from_text,
                    }
                    if content_type in content_handlers:
                        return await content_handlers[content_type](response)
                    else:
                        return "Unsupported content type"
                else:
                    m = f"Failed to fetch the webpage. Status code: {response.status}"
                    log(m)
                    return m
    except Exception as er:
        log(f"An error occurred: {str(er)}")

# Function - Extract text from PDF
async def extract_text_from_pdf(response):
    # Convert PDF to text
    pdf_content = await response.read()
    pdf2text = ""
    f = io.BytesIO(pdf_content)
    reader = PdfReader(f)
    for page in reader.pages:
        pdf2text = pdf2text + page.extract_text() + "\n"
    return pdf2text

# Function - Extract text from text
async def extract_text_from_text(response):
    return await response.text()

# Function - Extract text from HTML
async def extract_text_from_html(response):
    html_content = await response.text()
    # get title of page from html
    source = "Document Source: " + str(response.url)
    soup = BeautifulSoup(html_content, 'html.parser')
    title = ("Document Title: " + soup.title.string + "\n") if soup.title else ""
    paragraphs = soup.find_all(['p', 'code', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre', 'ol'])
    website_text = f"{title}{source}\nDocument Content:\n" + '\n\n'.join([p.get_text() for p in paragraphs])
    return website_text

#
# FastAPI Routes
#

templates = Jinja2Templates(directory="templates")

# Display the main chatbot page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

# Serve static socket.io.js
@app.get("/socket.io.js")
def serve_socket_io_js():
    return FileResponse("templates/socket.io.js", media_type="application/javascript")

# Display settings and stats
@app.get("/stats")
async def home(format: str = None):
    global client, stats
    # Create a simple status page
    data = {
        "TinyLLM Chatbot Version": VERSION,
        "Start Time": datetime.datetime.fromtimestamp(stats["start_time"]).strftime("%Y-%m-%d %H:%M:%S"),
        "Uptime": str(datetime.timedelta(seconds=int(time.time() - stats["start_time"]))),
        "Errors": stats["errors"],
        "User Queries": stats["ask"],
        "LLM Queries": stats["ask_llm"],
        "OpenAI API Key (OPENAI_API_KEY)": "************" if api_key != "" else "Not Set",
        "OpenAI API URL (OPENAI_API_URL)": api_base,
        "Agent Name (AGENT_NAME)": agentname,
        "LLM Model (LLM_MODEL)": mymodel,
        "Debug Mode (DEBUG)": DEBUG,
        "Current Clients (MAXCLIENTS)": f"{len(client)} of {MAXCLIENTS}",
        "LLM Max tokens Limit (MAXTOKENS)": MAXTOKENS,
        "LLM Temperature (TEMPERATURE)": TEMPERATURE,
        "Server Port (PORT)": PORT,
        "Saved Prompts (PROMPT_FILE)": PROMPT_FILE,
        "Read-Only Prompts (PROMPT_RO)": PROMPT_RO,
        "LLM System Tags in Prompts (USE_SYSTEM)": USE_SYSTEM,
        "Run without conversation context (ONESHOT).": ONESHOT,
        "RAG: Run in RAG Only Mode (RAG_ONLY)": RAG_ONLY,
        "RAG: Weaviate (WEAVIATE_HOST)": WEAVIATE_HOST,
        "RAG: default Library (WEAVIATE_LIBRARY)": WEAVIATE_LIBRARY,
        "RAG: Default Results Retrieved (RESULTS)": RESULTS,
    }
    if format == "json":
        return data
    # Build a simple HTML page based on data facets
    html = "<html><head><title>TinyLLM Chatbot Status</title>"
    html += "<style>body { font-family: Helvetica, Arial, sans-serif; }</style>"
    html += "</head><body>"
    html += "<h1>TinyLLM Chatbot Status</h1>"
    # Provide link to project
    html += "<p>Settings and Current Status for <a href='https://github.com/jasonacox/TinyLLM/tree/main/chatbot'>TinyLLM Chatbot</a></p>"
    html += "<table>"
    for key in data:
        html += f"<tr><td>{key}</td><td>{data[key]}</td></tr>"
    html += "</table>"
    # Add JS to refresh page every 5 seconds
    html += "<script>setTimeout(function(){location.reload()},5000);</script>"
    html += "</body></html>"
    return HTMLResponse(content=html, status_code=200)

# Return the current prompts
@app.get('/prompts')
async def get_prompts():
    global prompts
    # Update TEMPERATURE and MAXTOKENS
    prompts["LLM_temperature"] = TEMPERATURE
    prompts["LLM_max_tokens"] = MAXTOKENS
    if PROMPT_RO:
        prompts["READONLY"] = True
    return prompts

# POST requests to update prompts
@app.post('/saveprompts')
async def update_prompts(data: dict):
    global prompts, baseprompt, sio, TEMPERATURE, MAXTOKENS, agentname
    if PROMPT_RO:
        return ({"Result": "Prompts are read-only"})
    oldbaseprompt = prompts["baseprompt"]
    oldagentname = prompts["agentname"]
    log(f"Received prompts: {data}")
    # Update prompts
    for key in data:
        prompts[key] = data[key]
    save_prompts()
    if oldbaseprompt != prompts["baseprompt"] or oldagentname != prompts["agentname"]:
        # Update baseprompt
        agentname = prompts["agentname"]
        current_date = datetime.datetime.now()
        formatted_date = current_date.strftime("%B %-d, %Y")
        values = {"agentname": agentname, "date": formatted_date}
        baseprompt = expand_prompt(prompts["baseprompt"], values)
    # Update TEMPERATURE and MAXTOKENS
    if "LLM_temperature" in data:
        TEMPERATURE = float(data["LLM_temperature"])
    if "LLM_max_tokens" in data:
        MAXTOKENS = int(data["LLM_max_tokens"])
    # Notify all clients of update
    log("Base prompt updated - notifying users")
    await sio.emit('update', {'update': '[Prompts Updated - Refresh to reload]', 'voice': 'user'})
    return ({"Result": "Prompts updated"})

# Reset prompts to default
@app.get('/resetprompts')
async def reset_prompts_route():
    # Send the user the default prompts
    global default_prompts
    return (default_prompts)

# Return the current version and LLM model
@app.get('/version')
async def show_version_web():
    global VERSION, DEBUG
    log(f"Version requested - DEBUG={DEBUG}")
    if DEBUG:
        return {'version': "%s DEBUG MODE" % VERSION}
    return {'version': VERSION, 'model': mymodel}

# Send an alert to all clients
@app.post('/alert')
async def alert(data: dict):
    # Send an alert to all clients
    # Make sure TOKEN is set and matches
    if "token" in data and "message" in data and data["token"] == TOKEN:
        log(f"Received alert: {data}")
        await sio.emit('update', {'update': data["message"], 'voice': 'user'})
        return ({'status': 'Alert sent'})
    else:
        log(f"Invalid token or missing message: {data}")
        return ({'status': 'Invalid Token or missing message'})

#
# SocketIO Events
#

app.mount("/", socket_app)  # Here we mount socket app to main fastapi app

# Client connected - start thread to send updates
@sio.on('connect')
async def handle_connect(session_id, env):
    log(f"Client connected: {session_id}")

    # Convert each character to its hex representation
    def string_to_hex(input_string):
        hex_values = [hex(ord(char)) for char in input_string]
        return hex_values

    # Continuous thread to send updates to connected clients
    async def send_update(session_id, watch="prompt"):
        global client
        if watch == "prompt":
            topic = "update"
            ctx = "context"
        else:
            topic = "rag"
            ctx = "ragcontext"
        log(f"Starting send_update thread for {session_id}")

        # Verify session is valid
        if session_id not in client:
            log(f"Invalid session {session_id}")
            return
        client[session_id][watch] = ""
        try:
            while not client[session_id]["stop_thread_flag"]:
                if client[session_id][watch] == "":
                    await sio.sleep(0.1)
                else:
                    try:
                        # Ask LLM for answers
                        response= await ask(client[session_id][watch],session_id,ctx)
                        completion_text = ''
                        tokens = 0
                        # iterate through the stream of events and print it
                        stime = time.time()
                        for event in response:
                            event_text = event.choices[0].delta.content
                            tokens += 1
                            if event_text:
                                chunk = event_text
                                completion_text += chunk
                                if DEBUG:
                                    print(string_to_hex(chunk), end="")
                                    print(f" = [{chunk}]")
                                await sio.emit(topic, {'update': chunk, 'voice': 'ai'},room=session_id)
                        # Update footer with stats
                        await sio.emit(topic, {'update': 
                                                  f"TinyLLM Chatbot {VERSION} - {mymodel} - Tokens: {tokens} - TPS: {tokens/(time.time()-stime):.1f}", 
                                                  'voice': 'footer'},room=session_id)
                        # Check for link injection
                        if client[session_id]["links"]:
                            await sio.emit(topic, {'update': json.dumps(client[session_id]["links"]), 'voice': 'links'},room=session_id)
                            client[session_id]["links"] = ""
                        # Check for references
                        if client[session_id]["references"]:
                            await sio.emit(topic, {'update': client[session_id]["references"], 'voice': 'ref'},room=session_id)
                            client[session_id]["references"] = ""
                        if not ONESHOT:
                            # remember context
                            client[session_id][ctx].append({"role": "assistant", "content" : completion_text})
                    except Exception as er:
                        # Unable to process prompt, give error
                        log(f"ERROR {er}")
                        await sio.emit(topic, {'update': 'An error occurred - unable to complete.', 'voice': 'ai'},room=session_id)
                        # Reset context
                        client[session_id][ctx] = base_prompt()
                    # Signal response is done
                    await sio.emit(topic, {'update': '', 'voice': 'done'},room=session_id)
                    client[session_id][watch] = ''
                    if DEBUG:
                        print(f"AI: {completion_text}")
        except KeyError:
            log(f"Thread ended: {session_id}")
        except Exception as er:
            log(f"Thread error: {er}")

    if session_id in client:
        # Client reconnected - restart thread
        #client[session_id]["thread"].join()
        log(f"Client reconnected: {session_id}")
        # Send defaults to client
        log(f"Sending defaults to {session_id}")
        data = {
            "baseprompt": client[session_id]["baseprompt"],
            "ragprompt": client[session_id]["ragprompt"],
            "vectorNum": client[session_id]["vectorNum"],
            "vectordb": client[session_id]["vectordb"],
        }
        await sio.emit('defaults', data, room=session_id)
    else:
        # New client connected
        log(f"New client connected: {session_id}")
        # Limit number of clients
        if len(client) > MAXCLIENTS:
            log(f"Too many clients connected: {len(client)}")
            await sio.emit('update', {'update': 'Too many clients connected. Try again later.', 'voice': 'user'},room=session_id)
            return
        # Create client session
        client[session_id]={}
        # Initialize context for this client
        client[session_id]["context"] = base_prompt()
        client[session_id]["ragcontext"] = base_prompt()
        client[session_id]["remember"] = True
        client[session_id]["visible"] = True
        client[session_id]["prompt"] = ""
        client[session_id]["stop_thread_flag"] = False
        client[session_id]["references"] = ""
        client[session_id]["links"] = {}
        # Set up defaults for Lab
        client[session_id]["baseprompt"] = prompts["baseprompt"]
        client[session_id]["ragprompt"] = prompts["rag"]
        client[session_id]["vectorNum"] = RESULTS
        client[session_id]["vectordb"] = WEAVIATE_LIBRARY
        # Send defaults to client
        log(f"Sending defaults to {session_id}")
        data = {
            "baseprompt": client[session_id]["baseprompt"],
            "ragprompt": client[session_id]["ragprompt"],
            "vectorNum": client[session_id]["vectorNum"],
            "vectordb": client[session_id]["vectordb"],
        }
        await sio.emit('defaults', data, room=session_id)
        # Start continuous tasks to send updates
        asyncio.create_task(send_update(session_id, "prompt"))
        asyncio.create_task(send_update(session_id, "rag"))

# Client disconnected
@sio.on('disconnect')
async def handle_disconnect(session_id):
    log(f"Client disconnected: {session_id}")
    # Remove client
    if session_id in client:
        # shutdown thread
        client[session_id]["stop_thread_flag"] = True
        client.pop(session_id)

# Client requests update baseprompt
@sio.on('baseprompt')
async def handle_baseprompt(session_id, data):
    global client
    if session_id not in client:
        log(f"Invalid session {session_id}")
        return
    client[session_id]["baseprompt"] = data["prompt"]
    client[session_id]["context"] = base_prompt(template=data["prompt"])
    log(f"Client requests baseprompt {session_id} - {data['prompt']}")
    await sio.emit('update', {'update': 'Base Prompt Updated', 'voice': 'user'}, room=session_id)

# Client requests defaults
@sio.on('defaults')
async def handle_defaults(session_id, data):
    global client
    log(f"Client requests defaults {session_id}")
    if session_id not in client:
        log(f"Invalid session {session_id}")
        return
    data = {
        "baseprompt": client[session_id]["baseprompt"],
        "ragprompt": client[session_id]["ragprompt"],
        "vectorNum": client[session_id]["vectorNum"],
        "vectordb": client[session_id]["vectordb"],
    }
    await sio.emit('defaults', data, room=session_id)
    await sio.emit('update', {'update': 'Data Refreshed', 'voice': 'user'}, room=session_id)

# Run User Prompt - Poll vector database for similar document
@sio.on('userprompt')
async def handle_userprompt(session_id, data):
    global client
    log(f"Client requests userprompt {session_id}")
    if session_id not in client:
        log(f"Invalid session {session_id}")
        return
    client[session_id]["userprompt"] = data["prompt"]
    client[session_id]["vectordb"] = data["vectordb"]
    client[session_id]["vectorNum"] = data["num"]
    _, references = query_index(data["prompt"], 
                        client[session_id]["vectordb"],
                        int(client[session_id]["vectorNum"]))
    await sio.emit('docs-found', {'update': references, 'voice': 'user'}, room=session_id)

# Run RAG Prompt - Import and summarize topic from library
@sio.on('rag')
async def handle_rag(session_id, data):
    global client
    log(f"Client requests rag {session_id}")
    if session_id not in client:
        log(f"Invalid session {session_id}")
        return
    # Grab data from client
    client[session_id]["baseprompt"] = data["baseprompt"]
    client[session_id]["vectordb"] = data["vectordb"]
    client[session_id]["vectorNum"] = data["num"]
    client[session_id]["userprompt"] = data["userprompt"]
    client[session_id]["ragtemplate"] = data["ragtemplate"]
    # Query Weaviate for documents
    log(f"VectorDB: {client[session_id]['vectordb']} - VectorNum: {client[session_id]['vectorNum']}")
    results, references = query_index(client[session_id]["userprompt"],
                        client[session_id]["vectordb"],
                        int(client[session_id]["vectorNum"]))
    await sio.emit('docs-found', {'update': references, 'voice': 'user'}, room=session_id)
    # Create RAG prompt from documents
    log(f"RAG: {results}")
    client[session_id]["rag"] = expand_prompt(client[session_id]["ragtemplate"],
                                                    {"context_str": results,
                                                     "prompt": client[session_id]["userprompt"]})
    await sio.emit('update', {'update': '[RAG: Running...]', 'voice': 'user'}, room=session_id)

# Client sent a message - handle it
@sio.on('message')
async def handle_message(session_id, data):
    global client
    # Handle incoming user prompts and store them
    log(f'Received message from {session_id}')
    log(f"Received Data: {data}")
    if session_id not in client:
        log(f"Invalid session {session_id}")
        await handle_invalid_session(session_id)
        return
    p = data["prompt"]
    client[session_id]["visible"] = data["show"]
    # Did we get a start command? Use greeting prompt.
    if p == "{start}":
        p = prompts["greeting"]
    # Did we get asked to fetch a URL?
    if p.startswith("http"):
        await handle_url_prompt(session_id, p)
    elif p.startswith("/"):
        await handle_command(session_id, p)
    else:
        await handle_normal_prompt(session_id, p)
    return {'status': 'Message received'}

async def handle_invalid_session(session_id):
    await sio.emit('update', {'update': '[Session Unrecognized - Try Refresh]', 'voice': 'user'}, room=session_id)

async def handle_url_prompt(session_id, p):
    url = p.strip()
    client[session_id]["visible"] = False
    client[session_id]["remember"] = True
    website_text = await extract_text_from_url(url)
    if website_text:
        log(f"* Reading {len(website_text)} bytes {url}")
        await sio.emit('update', {'update': '%s [Reading...]' % url, 'voice': 'user'}, room=session_id)
        url_encoded = requests.utils.quote(url)
        client[session_id]["prompt"] = expand_prompt(prompts["website"], {"url": url_encoded, "website_text": website_text})
    else:
        await sio.emit('update', {'update': '%s [ERROR: Unable to read URL]' % url, 'voice': 'user'}, room=session_id)
        client[session_id]["prompt"] = ''

async def handle_command(session_id, p):
    command = p[1:].split(" ")[0].lower()
    if command == "":
        await sio.emit('update', {'update': '[Commands: /reset /version /sessions /rag /news /weather /stock]', 'voice': 'user'}, room=session_id)
        client[session_id]["prompt"] = ''
    elif command == "reset":
        await reset_context(session_id)
    elif command == "version":
        await show_version(session_id)
    elif command == "sessions":
        await show_sessions(session_id)
    elif command == "news":
        await fetch_news(session_id, p)
    elif command == "rag":
        await handle_rag_command(session_id, p)
    elif command == "weather":
        await handle_weather_command(session_id, p)
    elif command == "stock":
        await handle_stock_command(session_id, p)
    else:
        await sio.emit('update', {'update': '[Invalid command]', 'voice': 'user'}, room=session_id)
        client[session_id]["prompt"] = ''

async def reset_context(session_id):
    client[session_id]["context"] = base_prompt()
    await sio.emit('update', {'update': '[Memory Reset]', 'voice': 'user'}, room=session_id)
    client[session_id]["prompt"] = prompts["greeting"]
    client[session_id]["visible"] = False

async def show_version(session_id):
    await sio.emit('update', {'update': '[TinyLLM Version: %s - Session: %s]' % (VERSION, session_id), 'voice': 'user'}, room=session_id)
    client[session_id]["prompt"] = ''

async def show_sessions(session_id):
    result = ""
    x = 1
    for s in client:
        result += f"* {x}: {s}\n"
        x += 1
    await sio.emit('update', {'update': '[Sessions: %s]\n%s' % (len(client), result), 'voice': 'user'}, room=session_id)
    client[session_id]["prompt"] = ''

async def fetch_news(session_id, p):
    log("News requested")
    topic = p[5:].strip() or "none"
    await sio.emit('update', {'update': '%s [Fetching News]' % p, 'voice': 'user'}, room=session_id)
    context_str, links = await get_news(topic, 25)
    log(f"News Raw Context = {context_str}")
    client[session_id]["visible"] = False
    client[session_id]["remember"] = True
    client[session_id]["links"] = links
    client[session_id]["prompt"] = expand_prompt(prompts["news"], {"context_str": context_str})

async def handle_rag_command(session_id, p):
    rag = p[4:].strip()
    parts = rag.split()
    library = ""
    if len(parts) >= 2:
        library = parts[0]
        if parts[1].isdigit():
            number = int(parts[1])
            prompt = ' '.join(parts[2:])
        else:
            number = RESULTS
            prompt = ' '.join(parts[1:])
    if not library or not prompt:
        await sio.emit('update', {'update': '[Usage: /rag {library} {opt:number} {prompt}] - Import and summarize topic from library.', 'voice': 'user'}, room=session_id)
    else:
        if WEAVIATE_HOST:
            log(f"Pulling {number} entries from {library} with prompt {prompt}")
            if not ONESHOT:
                await sio.emit('update', {'update': '%s [RAG Command Running...]' % p, 'voice': 'user'}, room=session_id)
            results, references = query_index(prompt, library, number)
            if results:
                context_str = ""
                client[session_id]["visible"] = False
                client[session_id]["remember"] = True
                context_str = results
                log(f" = {references}")
                client[session_id]["references"] = references
                client[session_id]["prompt"] = expand_prompt(prompts["rag"], {"context_str": context_str, "prompt": prompt})
            else:
                await sio.emit('update', {'update': '[Unable to access Vector Database for %s]' % library, 'voice': 'user'}, room=session_id)
        else:
            await sio.emit('update', {'update': '[RAG Support Disabled - Check Config]', 'voice': 'user'}, room=session_id)

async def handle_weather_command(session_id, p):
    log("Weather prompt")
    await sio.emit('update', {'update': '%s [Weather Command Running...]' % p, 'voice': 'user'}, room=session_id)
    location = await ask_llm(expand_prompt(prompts["location"], {"prompt": p}))
    location = ''.join(e for e in location if e.isalnum() or e.isspace())
    if "none" in location.lower():
        context_str = await get_weather("")
    else:
        context_str = await get_weather(location)
    client[session_id]["visible"] = False
    client[session_id]["remember"] = True
    client[session_id]["prompt"] = expand_prompt(prompts["weather"], {"prompt": p, "context_str": context_str, "location": location})

async def handle_stock_command(session_id, p):
    log("Stock prompt")
    prompt = p[6:].strip()
    if not prompt:
        await sio.emit('update', {'update': '[Usage: /stock {company}] - Fetch stock price for company.', 'voice': 'user'}, room=session_id)
        return
    await sio.emit('update', {'update': '%s [Fetching Stock Price...]' % p, 'voice': 'user'}, room=session_id)
    log(f"Stock Prompt: {prompt}")
    company = await ask_llm(expand_prompt(prompts["company"], {"prompt": prompt}))
    company = ''.join(e for e in company if e.isalnum() or e.isspace())
    if "none" in company.lower():
        context_str = "Unable to fetch stock price - Unknown company specified." 
    else:
        context_str = await get_stock(company)
    log(f"Company = {company} - Context = {context_str}")
    await sio.emit('update', {'update': context_str, 'voice': 'ai'}, room=session_id)
    client[session_id]["context"].append({"role": "user", "content" : "What is the stock price for %s?" % company})
    client[session_id]["context"].append({"role": "assistant", "content" : context_str})
    client[session_id]["prompt"] = ''

async def handle_normal_prompt(session_id, p):
    if client[session_id]["visible"]:
        await sio.emit('update', {'update': p, 'voice': 'user'}, room=session_id)
    if WEAVIATE_HOST and RAG_ONLY and p is not prompts["greeting"]:
        # Activate RAG every time
        await handle_rag_command(session_id, f"/rag {WEAVIATE_LIBRARY} {RESULTS} {p}")
    else:
        client[session_id]["prompt"] = p

#
# Start dev server and listen for connections
#

if __name__ == '__main__':
    log(f"DEV MODE - Starting server on port {PORT}. Use uvicorn server:app for PROD mode.")
    kwargs = {"host": "0.0.0.0", "port": PORT}
    uvicorn.run("server:app", **kwargs)
