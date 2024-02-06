#!/usr/bin/python3
"""
Web based ChatBot Example

Web chat client for OpenAI and the llama-cpp-python[server] OpenAI API Compatible 
Python Flask based Web Server. Provides a simple web based chat session.

Features:
    * Uses OpenAI API
    * Works with local hosted OpenAI compatible llama-cpp-python[server]
    * Retains conversational context for LLM
    * Uses response stream to render LLM chunks instead of waiting for full response
    * Multithreaded to support multiple clients
    * Supports commands to reset context, get version, etc.
    * Supports RAG prompts (BETA)

Requirements:
    * pip install openai flask flask-socketio bs4 pypdf
    * pip install qdrant-client sentence-transformers pydantic~=2.4.2

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
    * QDRANT_HOST - URL to Qdrant Vector Database (Optional) - https://qdrant.tech/
    * DEVICE - cuda or cpu - only used for Sentence Transformer
    * RESULTS - Number of results to return from RAG query
    * ST_MODEL - Sentence Transformer Model to use

Running a llama-cpp-python server:
    * CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
    * pip install llama-cpp-python[server]
    * python3 -m llama_cpp.server --model models/7B/ggml-model.bin

Author: Jason A. Cox
23 Sept 2023
https://github.com/jasonacox/TinyLLM

"""
# Import Libraries
import os
import io
import time
import datetime
import threading
import signal
import requests
import logging
import json
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import openai
from pypdf import PdfReader

# Constants
VERSION = "v0.11.2"

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
agentname = os.environ.get("AGENT_NAME", "Jarvis")                          # Set the name of your bot
mymodel = os.environ.get("LLM_MODEL", "models/7B/gguf-model.bin")           # Pick model to use e.g. gpt-3.5-turbo for OpenAI
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"                  # Set to True to enable debug mode
MAXCLIENTS = int(os.environ.get("MAXCLIENTS", 1000))                        # Maximum number of concurrent clients
MAXTOKENS = int(os.environ.get("MAXTOKENS", 16*1024))                       # Maximum number of tokens to send to LLM
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))                     # LLM temperature
PORT = int(os.environ.get("PORT", 5000))                                    # Port to listen on
PROMPT_FILE = os.environ.get("PROMPT_FILE", "prompts.json")                 # File to store prompts
USE_SYSTEM = os.environ.get("USE_SYSTEM", "false").lower == "true"          # Use system in chat prompt if True

# RAG Configuration Settings
STMODEL = os.environ.get("ST_MODEL", "all-MiniLM-L6-v2")                    # Sentence Transformer Model to use
QDRANT_HOST = os.environ.get("QDRANT_HOST", "")                             # Empty = disable RAG support
DEVICE = os.environ.get("DEVICE", "cpu")                   #  cuda or cpu   # Device to use for Sentence Transformer
RESULTS = os.environ.get("RESULTS", 1)                                      # Number of results to return from RAG query
ALPHA_KEY = os.environ.get("ALPHA_KEY", "alpha_key")                        # Optional - Alpha Vantage API Key

# Prompt Defaults
default_prompts = {}
default_prompts["greeting"] = "Hi"
default_prompts["agentname"] = "Jarvis"
default_prompts["baseprompt"] = "You are {agentname}, a highly intelligent assistant. Keep your answers brief and accurate."
default_prompts["weather"] = "You are a weather forecaster. Keep your answers brief and accurate. Current date is {date} and weather conditions:\n[DATA]{context_str}[/DATA]\nProvide a weather update, current weather alerts, conditions, precipitation and forecast for {location} and answer this: {prompt}."
default_prompts["stock"] = "You are a stock analyst. Keep your answers brief and accurate. Current date is {date}."
default_prompts["news"] = "You are a newscaster who specializes in providing headline news. Use only the following context provided by Google News to summarize the top 10 headlines for today. Do not display the pub date or timestamp. Rank headlines by most important to least important. Always include the news organization. Do not add any commentary.\nAlways use this format: \n1. [Headline Title] - [News Organization]\n2. [Headline Title] - [News Organization]\nContext: {context_str}\nAnswer:"
default_prompts["clarify"] = "You are a highly intelligent assistant. Keep your answers brief and accurate. {format}."
default_prompts["location"] = "What location is specified in this prompt, state None if there isn't one. Use a single word answer. [BEGIN] {prompt} [END]"
default_prompts["company"] = "What company is related to the stock price in this prompt? Please state none if there isn't one. Use a single word answer: [BEGIN] {prompt} [END]"
default_prompts["rag"] = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Back up your answer using bullet points and facts from the context.\nQuestion: {prompt}\nContext: {context_str}\nAnswer:"
default_prompts["website"] = "Summarize the following text from URL {url}:\n{website_text}"

# Import Qdrant and Sentence Transformer
try:
    from sentence_transformers import SentenceTransformer
    import qdrant_client as qc
    import qdrant_client.http.models as qmodels
except:
    logger.error("Unable to import sentence_transformers or qdrant_client - Disabling Qdrant vector DB support")
    QDRANT_HOST = ""

# Test OpenAI API
while True:
    log("Testing OpenAI API...")
    try:
        log(f"Using openai library version {openai.__version__}")
        log(f"Connecting to OpenAI API at {api_base} using model {mymodel}")
        llm = openai.OpenAI(api_key=api_key, base_url=api_base)
        llm.chat.completions.create(
            model=mymodel,
            max_tokens=MAXTOKENS,
            stream=False,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": "Hello"}],
        )
        break
    except Exception as e:
        log("OpenAI API Error: %s" % e)
        log(f"Unable to connect to OpenAI API at {api_base} using model {mymodel}.")
        log("Sleeping 10 seconds...")
        time.sleep(10)

# Sentence Transformer Setup
if QDRANT_HOST:
    log("Sentence Transformer starting...")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sent_model = SentenceTransformer(STMODEL, device=DEVICE) 

    # Qdrant Setup
    log("Connecting to Qdrant DB...")
    qdrant = qc.QdrantClient(url=QDRANT_HOST)

# Create embeddings for text
def embed_text(text):
    embeddings = sent_model.encode(text, convert_to_tensor=True)
    return embeddings

# Find document closely related to query
def query_index(query, library, top_k=5):
    vector = embed_text(query)
    try:
        results = qdrant.search(
            collection_name=library,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
        found=[]
        for res in results:
            found.append({"title": res.payload["title"],
                            "text": res.payload["text"],
                            "url": res.payload["url"],
                            "score": res.score})
        return found
    except Exception as e:
        log(f"Error querying Qdrant: {str(e)}")
        return None

# Configure Flask App and SocketIO
log("Starting server...")
app = Flask(__name__)
socketio = SocketIO(app)

# Globals
client = {}
prompts = {}

# Functions to mange prompts and PROMPT_FILE
# Load prompts from PROMPT_FILE
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
        log("Unable to load prompts, using defaults.")
        reset_prompts()

# Save prompts to PROMPT_FILE
def save_prompts():
    global prompts
    try:
        with open(PROMPT_FILE, "w") as f:
            json.dump(prompts, f)
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
if len(prompts) == 0:
    # Add default prompts
    reset_prompts()
    save_prompts()
    log(f"Setting defaults for {len(prompts)} prompts.")
else:
    log(f"Loaded {len(prompts)} prompts.")

# Set base prompt and initialize the context array for conversation dialogue
current_date = datetime.datetime.now()
formatted_date = current_date.strftime("%B %-d, %Y")
values = {"agentname": agentname, "date": formatted_date}
baseprompt = expand_prompt(prompts["baseprompt"], values)

# Function to return base conversation prompt
def base_prompt(content=baseprompt):
    if USE_SYSTEM:
        return [{"role": "system", "content": content}] 
    else:
        return [{"role": "user", "content": content}, {"role": "assistant", "content": "Okay, let's get started."}] 

# Initialize context 
context = base_prompt()

# Function - Send user prompt to LLM for response
def ask(prompt, sid=None):
    global client

    response = False
    log(f"Context size = {len(context)}")
    while not response:
        try:
            # remember context
            client[sid]["context"].append({"role": "user", "content": prompt})
            log(f"messages = {client[sid]['context']} - model = {mymodel}")
            llm = openai.OpenAI(api_key=api_key, base_url=api_base)
            response = llm.chat.completions.create(
                model=mymodel,
                max_tokens=MAXTOKENS,
                stream=True, # Send response chunks as LLM computes next tokens
                temperature=TEMPERATURE,
                messages=client[sid]["context"],
            )
        except openai.OpenAIError as e:
            log(f"ERROR {e}")
            client[sid]["context"].pop()
            if "maximum context length" in str(e):
                if len(prompt) > 1000:
                    # assume we have very large prompt - cut out the middle
                    prompt = prompt[:len(prompt)//4] + " ... " + prompt[-len(prompt)//4:]
                    log(f"Reduce prompt size - now {len(prompt)}")
                elif len(client[sid]["context"]) > 4:
                    # our context has grown too large, truncate the top
                    client[sid]["context"] = client[sid]["context"][:1] + client[sid]["context"][3:]
                    log(f"Truncate context: {len(client[sid]['context'])}")
                else:
                    # our context has grown too large, reset
                    client[sid]["context"] = base_prompt()   
                    log(f"Reset context {len(client[sid]['context'])}")
                    socketio.emit('update', {'update': '[Memory Reset]', 'voice': 'user'},room=sid)

    if not client[sid]["remember"]:
        client[sid]["remember"] =True
        client[sid]["context"].pop()
        client[sid]["context"].append({"role": "user", "content": "Help me remember."})
    return response

def ask_llm(query, format=""):
    # Ask LLM a question
    if format == "":
        format = f"Respond in {format}."
    content = base_prompt(expand_prompt(prompts["clarify"], {"format": format})) + [{"role": "user",
                "content": query}]
    log(f"ask_llm: {content}")
    llm = openai.OpenAI(api_key=api_key, base_url=api_base)
    response = llm.chat.completions.create(
        model=mymodel,
        max_tokens=MAXTOKENS,
        stream=False,
        temperature=TEMPERATURE,
        messages=content,
    )
    log(f"ask_llm -> {response.choices[0].message.content.strip()}")
    return response.choices[0].message.content.strip()

# Function - Get weather for location
def get_weather(location):
    # Look up weather for location
    if location == "":
        location = "New York"
    location = location.replace(" ", "+")
    url = "https://wttr.in/%s?format=j2" % location
    log(f"Fetching weather for {location} from {url}")
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return "Unable to fetch weather for %s" % location
    
# Function - Get stock price for company
def get_stock(company):
    if ALPHA_KEY == "alpha_key":
        return "Unable to fetch stock price for %s - No Alpha Vantage API Key" % company
    # First try to get the ticker symbol
    symbol = ask_llm(f"What is the stock symbol for {company}? Respond with symbol.")
    if "none" in symbol.lower():
        return "Unable to fetch stock price for %s - No matching symbol" % company
    # Check to see if response has multiple words and if so, pick the last one
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
def get_top_articles(url, max=10):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'xml')
    items = soup.findAll('item')
    articles = ""
    count = 0
    for item in items:
        title = item.find('title').string.strip()
        pubdate = item.find('pubDate').string.strip()
        articles += f"Headline: {title} - Pub Date: {pubdate}\n"
        count += 1
        if count >= max:
            break
    return articles

def get_news(topic, max=10):
    # Look up news for topic
    if "none" in topic.lower() or "current" in topic.lower():
        url = "https://news.google.com/rss/"
    else:
        topic = topic.replace(" ", "+")
        url = "https://news.google.com/rss/search?q=%s" % topic
    log(f"Fetching news for {topic} from {url}")
    response = get_top_articles(url, max)
    return response
    
def extract_text_from_url(url):
    try:
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
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
                "text/html": extract_text_from_html
            }
            if content_type in content_handlers:
                return content_handlers[content_type](response)
            else:
                return "Unsupported content type"
        else:
            m = f"Failed to fetch the webpage. Status code: {response.status_code}"
            log(m)
            return m
    except Exception as e:
        log(f"An error occurred: {str(e)}")

def extract_text_from_pdf(response):
    # Convert PDF to text
    pdf_content = response.content
    pdf2text = ""
    f = io.BytesIO(pdf_content)
    reader = PdfReader(f)
    for page in reader.pages:
        pdf2text = pdf2text + page.extract_text() + "\n"
    return pdf2text

def extract_text_from_text(response):
    return response.text

def extract_text_from_html(response):
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all(['p', 'code', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre', 'ol'])
    website_text = '\n'.join([p.get_text() for p in paragraphs])
    return website_text

@app.route('/')
def index():
    # Send the user the chatbot HTML page
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    if session_id in client:
        # Client reconnected - restart thread
        client[session_id]["thread"].join()
        log(f"Client reconnected: {session_id}")
    else:
        # New client connected
        log(f"Client connected: {session_id}")
        # Limit number of clients
        if len(client) > MAXCLIENTS:
            log(f"Too many clients connected: {len(client)}")
            socketio.emit('update', {'update': 'Too many clients connected. Try again later.', 'voice': 'user'},room=session_id)
            return
        # Create client session
        client[session_id]={}
        # Initialize context for this client
        client[session_id]["context"] = base_prompt()
        client[session_id]["remember"] = True
        client[session_id]["visible"] = True
        client[session_id]["prompt"] = ""
        client[session_id]["stop_thread_flag"] = False
        # Create a background thread to send updates
        update_thread = threading.Thread(target=send_update, args=(session_id,))
        update_thread.daemon = True  # Thread will terminate when the main program exits
        update_thread.start()
        client[session_id]["thread"] = update_thread

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    log(f"Client disconnected: {session_id}")
    # Remove client
    if session_id in client:
        # shutdown thread
        client[session_id]["stop_thread_flag"] = True
        client[session_id]["thread"].join()
        client.pop(session_id)

@socketio.on('message')
def handle_message(data):
    global client
    session_id = request.sid
    # Handle incoming user prompts and store them
    log(f'Received message from {session_id}')
    log(f"Received Data: {data}")
    if session_id not in client:
        log(f"Invalid session {session_id}")
        socketio.emit('update', {'update': '[Session Unrecognized - Try Refresh]', 'voice': 'user'},room=session_id)
        return
    p = data["prompt"]
    client[session_id]["visible"] = data["show"]
    # Did we get a start command? Use greeting prompt.
    if p == "{start}":
        p = prompts["greeting"]
    # Did we get asked to fetch a URL?
    if p.startswith("http"):
        # Summarize text at URL
        url = p
        client[session_id]["visible"] = False # Don't display full document but...
        client[session_id]["remember"] = True # Remember full content to answer questions
        website_text = extract_text_from_url(p.strip())
        if website_text:
            log(f"* Reading {len(website_text)} bytes {url}")
            socketio.emit('update', {'update': '%s [Reading...]' % url, 'voice': 'user'},room=session_id)
            url_encoded = requests.utils.quote(url)
            client[session_id]["prompt"] = expand_prompt(prompts["website"], {"url": url_encoded, "website_text": website_text})
        else:
            socketio.emit('update', {'update': '[Unable to read URL]', 'voice': 'user'},room=session_id)
            client[session_id]["prompt"] = ''
    elif p.startswith("/"):
        # Handle commands
        command = p[1:].split(" ")[0]
        if command == "":
            # Display help
            socketio.emit('update', {'update': '[Commands: /reset /version /sessions /rag /news /weather /stock]', 'voice': 'user'},room=session_id)
            client[session_id]["prompt"] = ''
        elif command == "reset":
            # Reset context
            client[session_id]["context"] = base_prompt()
            socketio.emit('update', {'update': '[Memory Reset]', 'voice': 'user'},room=session_id)
            client[session_id]["prompt"] = prompts["greeting"]
            client[session_id]["visible"] = False
        elif command == "version":
            # Display version
            socketio.emit('update', {'update': '[TinyLLM Version: %s - Session: %s]' % ( VERSION, session_id ), 'voice': 'user'},room=session_id)
            client[session_id]["prompt"] = ''
        elif command == "sessions":
            # Display sessions
            result = ""
            x = 1
            for s in client:
                result += f"<br> * {x}: {s}\n"
                x += 1
            socketio.emit('update', {'update': '[Sessions: %s]\n%s' % (len(client), result), 'voice': 'user'},room=session_id)
            client[session_id]["prompt"] = ''
        elif command == "news":
            log("News requested")
            socketio.emit('update', {'update': '/news [Fetching News]', 'voice': 'user'},room=session_id)
            context_str = get_news("none", 25)
            log(f"News Raw Context = {context_str}")
            client[session_id]["visible"] = False
            client[session_id]["remember"] = True
            client[session_id]["prompt"] = (
                expand_prompt(prompts["news"], {"context_str": context_str})
            )
        elif command == "rag":
             # RAG Command - IMPORT from library - Format: /rag library [opt:number=1] [prompt]
            log("RAG requested")
            rag = p[4:].strip()
            parts = rag.split()
            library = ""
            # Check to see if second element is a number
            if len(parts) >= 2:
                library = parts[0]
                if parts[1].isdigit():
                    number = int(parts[1])
                    prompt = ' '.join(parts[2:])
                else:
                    number = 1
                    prompt = ' '.join(parts[1:])
            if not library or not prompt:
                socketio.emit('update', {'update': '[Usage: /rag {library} {opt:number} {prompt}] - Import and summarize topic from library.', 'voice': 'user'},room=session_id)
            else:
                if QDRANT_HOST:
                    log(f"Pulling {number} entries from {library} with prompt {prompt}")
                    socketio.emit('update', {'update': '%s [RAG Command Running...]' % p, 'voice': 'user'},room=session_id)
                    # Query Vector Database for library
                    results = query_index(prompt, library, top_k=number)
                    if results:
                        context_str = ""
                        client[session_id]["visible"] = False
                        client[session_id]["remember"] = True
                        for result in results:
                            context_str += f" <li> {result['title']}: {result['text']}\n"
                            log(" * " + result['title'])
                        log(f" = {context_str}")
                        client[session_id]["prompt"] = (
                            expand_prompt(prompts["rag"], {"context_str": context_str})
                        )
                    else:
                        socketio.emit('update', {'update': '[Unable to access Vector Database for %s]' % library, 'voice': 'user'},room=session_id)
                else:
                    socketio.emit('update', {'update': '[RAG Support Disabled - Check Config]', 'voice': 'user'},room=session_id)
        elif command == "weather":
            # Weather prompt
            log("Weather prompt")
            socketio.emit('update', {'update': '%s [Weather Command Running...]' % p, 'voice': 'user'},room=session_id)
            # "What location is specified in this prompt, state None if there isn't one. Use a single word answer. [BEGIN] {prompt} [END]"
            location = ask_llm(expand_prompt(prompts["location"], {"prompt": p}))
            # Remove any non-alpha characters
            location = ''.join(e for e in location if e.isalnum() or e.isspace())
            if "none" in location.lower():
                context_str = get_weather("")
            else:
                context_str = get_weather(location)
            client[session_id]["visible"] = False
            client[session_id]["remember"] = True
            client[session_id]["prompt"] = (
                expand_prompt(prompts["weather"], {"prompt": p, "context_str": context_str, "location": location})
            )
        elif command == "stock":
            # Stock prompt
            log("Stock prompt")
            socketio.emit('update', {'update': '%s [Fetching Stock Price...]' % p, 'voice': 'user'},room=session_id)
            prompt = p[6:].strip()
            log(f"Stock Prompt: {prompt}")
            company = ask_llm(expand_prompt(prompts["company"], {"prompt": prompt}))
            # Remove any non-alpha characters
            company = ''.join(e for e in company if e.isalnum() or e.isspace())
            if "none" in company.lower():
                context_str = "Unable to fetch stock price - Unknown company specified." 
            else:
                context_str = get_stock(company)
            log(f"Company = {company} - Context = {context_str}")
            socketio.emit('update', {'update': context_str, 'voice': 'ai'},room=session_id)
            # remember context
            client[session_id]["context"].append({"role": "user", "content" : "What is the stock price for %s?" % company})
            client[session_id]["context"].append({"role": "assistant", "content" : context_str})
            client[session_id]["prompt"] = ''
    else:
        # Normal prompt
        client[session_id]["prompt"] = p
    return jsonify({'status': 'Message received'})

@app.route('/prompts')
def get_prompts():
    global prompts
    return jsonify(prompts)

# Add a route for POST requests to update prompts
@app.route('/saveprompts', methods=['POST'])
def update_prompts():
    global prompts, baseprompt, socketio
    oldbaseprompt = prompts["baseprompt"]
    oldagentname = prompts["agentname"]
    # Get the JSON data sent from the client
    data = request.get_json(force=True)
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
    # Notify all clients of update
    log("Base prompt updated - notifying users")
    socketio.emit('update', {'update': '[Prompts Updated]', 'voice': 'user'})
    return jsonify({"Result": "Prompts updated"})

@app.route('/resetprompts')
def reset_prompts_route():
    # Send the user the default prompts
    global default_prompts
    return jsonify(default_prompts)

@app.route('/version')
def version():
    global VERSION, DEBUG
    if DEBUG:
        return jsonify({'version': "%s DEBUG MODE" % VERSION})
    return jsonify({'version': VERSION})

# On app shutdown - close all sessions
def remove_sessions(exception):
    global client
    log("Shutting down sessions...")
    for session_id in client:
        log(f"Shutting down session {session_id}")
        socketio.emit('update', {'update': '[Shutting Down]', 'voice': 'user'},room=session_id)
        client[session_id]["stop_thread_flag"] = True
        client[session_id]["thread"].join()
    log("Shutdown complete.")

# Convert each character to its hex representation
def string_to_hex(input_string):
    hex_values = [hex(ord(char)) for char in input_string]
    return hex_values

# Continuous thread to send updates to connected clients
def send_update(session_id): 
    global client
    log(f"Starting send_update thread for {session_id}")

    # Verify session is valid
    if session_id not in client:
        log(f"Invalid session {session_id}")
        return
    while not client[session_id]["stop_thread_flag"]:
        if client[session_id]["prompt"] == "":
            time.sleep(.5)
        else:
            update_text = client[session_id]["prompt"] 
            if client[session_id]["visible"] :
                socketio.emit('update', {'update': update_text, 'voice': 'user'},room=session_id)
            try:
                # Ask LLM for answers
                response=ask(client[session_id]["prompt"],session_id)
                completion_text = ''
                # iterate through the stream of events and print it
                for event in response:
                    event_text = event.choices[0].delta.content
                    if event_text:
                        chunk = event_text
                        completion_text += chunk
                        if DEBUG:
                            print(string_to_hex(chunk), end="")
                            print(f" = [{chunk}]")
                        socketio.emit('update', {'update': chunk, 'voice': 'ai'},room=session_id)
                # remember context
                client[session_id]["context"].append({"role": "assistant", "content" : completion_text})
            except Exception as e:
                # Unable to process prompt, give error
                log(f"ERROR {e}")
                socketio.emit('update', {'update': 'An error occurred - unable to complete.', 'voice': 'ai'},room=session_id)
                # Reset context
                client[session_id]["context"] = base_prompt()
            # Signal response is done
            socketio.emit('update', {'update': '', 'voice': 'done'},room=session_id)
            client[session_id]["prompt"] = ''
            if DEBUG:
                print(f"AI: {completion_text}")

def sigTermHandler(signum, frame):
    # Shutdown all threads
    remove_sessions(None)
    raise SystemExit(f"Signal {signum} received - Shutting down.")

# Start server - Dev Mode
if __name__ == '__main__':
    signal.signal(signal.SIGTERM, sigTermHandler)
    signal.signal(signal.SIGINT, sigTermHandler) 
    socketio.run(app, host='0.0.0.0', port=PORT, debug=DEBUG, allow_unsafe_werkzeug=True)

