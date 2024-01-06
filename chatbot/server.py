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
    * OPENAI_API_KEY - Required only for OpenAI
    * OPENAI_API_BASE - URL to OpenAI API Server or locally hosted version
    * AGENT_NAME - Name for Bot
    * AGENT_NAME - LLM Model to Use
    * ALPHA_KEY - Alpha Vantage API Key for Stocks (Optional) - https://www.alphavantage.co/support/#api-key
    * QDRANT_HOST - URL to Qdrant Vector Database (Optional) - https://qdrant.tech/
    * DEVICE - cuda or cpu
    * RESULTS - Number of results to return from RAG query
    * MAXCLIENTS - Maximum number of clients to allow
    * MAXTOKENS - Maximum number of tokens to send to LLM
    * TEMPERATURE - LLM temperature
    * ST_MODEL - Sentence Transformer Model to use
    * PORT - Port to listen on

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
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import openai
import qdrant_client as qc
import qdrant_client.http.models as qmodels
from pypdf import PdfReader

# Constants
VERSION = "v0.9.2"

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("TinyLLM %s" % VERSION)
# Sentence Transformer
from sentence_transformers import SentenceTransformer

def log(text):
    logger.info(text)

# Configuration Settings - Showing local LLM
api_key = os.environ.get("OPENAI_API_KEY", "open_api_key")    # Required, use bogus string for Llama.cpp
api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com") 
agentname = os.environ.get("AGENT_NAME", "Jarvis")                              # Set the name of your bot
mymodel = os.environ.get("LLM_MODEL", "models/7B/gguf-model.bin")                # Pick model to use e.g. gpt-3.5-turbo for OpenAI
DEBUG = os.environ.get("DEBUG", "False") == "True"
STMODEL = os.environ.get("ST_MODEL", "all-MiniLM-L6-v2")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "") # Empty = disable RAG support
DEVICE = os.environ.get("DEVICE", "cuda")
RESULTS = os.environ.get("RESULTS", 1)
MAXCLIENTS = int(os.environ.get("MAXCLIENTS", 10))
MAXTOKENS = int(os.environ.get("MAXTOKENS", 2048))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.7))
ALPHA_KEY = os.environ.get("ALPHA_KEY", "alpha_key") # Optional - Alpha Vantage API Key
PORT = int(os.environ.get("PORT", 5000))

# Test OpenAI API
while True:
    log("Testing OpenAI API...")
    try:
        log(f"Using openai library version {openai.__version__}")
        llm = openai.OpenAI(api_key=api_key, base_url=api_base)
        llm.chat.completions.create(
            model=mymodel,
            max_tokens=MAXTOKENS,
            stream=False,
            temperature=TEMPERATURE,
            messages=[{"role": "system", "content": "Hello"}],
        )
        break
    except Exception as e:
        log("OpenAI API Error: %s" % e)
        log(f"Unable to connect to OpenAI API at {openai.api_base} using model {mymodel}.")
        log("Sleeping 10 seconds...")
        time.sleep(10)

# Sentence Transformer Setup
if QDRANT_HOST:
    log("Sentence Transformer starting...")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model = SentenceTransformer(STMODEL, device=DEVICE) 

    # Qdrant Setup
    log("Connecting to Qdrant DB...")
    qdrant = qc.QdrantClient(url=QDRANT_HOST)

# Create embeddings for text
def embed_text(text):
    embeddings = model.encode(text, convert_to_tensor=True)
    return embeddings

# Find document closely related to query
def query_index(query, library, top_k=5):
    vector = embed_text(query)
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

# Configure Flask App and SocketIO
log("Starting server...")
app = Flask(__name__)
socketio = SocketIO(app)

# Globals
client = {}

# Set base prompt and initialize the context array for conversation dialogue
current_date = datetime.datetime.now()
formatted_date = current_date.strftime("%B %-d, %Y")
baseprompt = "You are %s, a highly intelligent assistant. Keep your answers brief and accurate. Current date is %s." % (agentname, formatted_date)
context = [{"role": "system", "content": baseprompt}]

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
                    client[sid]["context"] = [{"role": "system", "content": baseprompt}]   
                    log(f"Reset context {len(client[sid]['context'])}")
                    socketio.emit('update', {'update': '[Memory Reset]', 'voice': 'user'},room=sid)

    if not client[sid]["remember"]:
        client[sid]["remember"] =True
        client[sid]["context"].pop()
    return response

def classify(prompt):
    # Ask LLM to classify the prompt
    #return "something else"
    content = [{"role": "system", 
                "content": "You are a highly intelligent assistant. Keep your answers brief and accurate."},
                {"role": "user",
                "content": f"Examine the following statement and identify, with a single word answer if it is about a greeting, weather, stock price, news, or something else. [BEGIN] {prompt} [END]"}]
    log(f"content: {content}")
    llm = openai.OpenAI(api_key=api_key, base_url=api_base)
    response = llm.chat.completions.create(
        model=mymodel,
        max_tokens=MAXTOKENS,
        stream=False,
        temperature=TEMPERATURE,
        messages=content,
    )
    log(f"classify = {response.choices[0].message.content.strip()}")
    return response.choices[0].message.content.strip()
    
def clarify(prompt, format="text"):
    # Ask LLM to clarify the prompt
    content = [{"role": "system", 
                "content": f"You are a highly intelligent assistant. Keep your answers brief and accurate. Respond in {format}."},
                {"role": "user",
                "content": f"{prompt}"}]
    log(f"content: {content}")
    llm = openai.OpenAI(api_key=api_key, base_url=api_base)
    response = llm.chat.completions.create(
        model=mymodel,
        max_tokens=MAXTOKENS,
        stream=False,
        temperature=TEMPERATURE,
        messages=content,
    )
    log(f"clarify = {response.choices[0].message.content.strip()}")
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
    symbol = clarify(f"What is the stock symbol for {company}? Respond with symbol.")
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
    
def extract_text_from_blog(url):
    try:
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            # Check to see if response is a PDF
            if response.headers["Content-Type"] == "application/pdf":
                # Convert PDF to text
                pdf2text = ""
                f = io.BytesIO(response.content)
                reader = PdfReader(f)
                for page in reader.pages:
                    pdf2text = pdf2text + page.extract_text() + "\n"
                return pdf2text
            # Check to see if response is HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find and extract all text within paragraph (p) tags
            paragraphs = soup.find_all('p')

            # Concatenate the text from all paragraphs
            blog_text = '\n'.join([p.get_text() for p in paragraphs])

            return blog_text
        else:
            m = f"Failed to fetch the webpage. Status code: {response.status_code}"
            log(m)
            return m
    except Exception as e:
        log(f"An error occurred: {str(e)}")

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
        client[session_id]["context"] = [{"role": "system", "content": baseprompt}]
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
    # Did we get asked to fetch a URL?
    if p.startswith("http"):
        # Summarize text at URL
        url = p
        client[session_id]["visible"] = False
        client[session_id]["remember"] = False # Don't add blog to context window, just summary
        blogtext = extract_text_from_blog(p.strip())
        if blogtext:
            log(f"* Reading {len(blogtext)} bytes {url}")
            socketio.emit('update', {'update': '[Reading: %s]' % url, 'voice': 'user'},room=session_id)
            client[session_id]["prompt"] = "Summarize the following text:\n" + blogtext
        else:
            socketio.emit('update', {'update': '[Unable to read URL]', 'voice': 'user'},room=session_id)
            client[session_id]["prompt"] = ''
    elif p.startswith("/"):
        # Handle commands
        command = p[1:].split(" ")[0]
        if command == "reset":
            # Reset context
            client[session_id]["context"] = [{"role": "system", "content": baseprompt}]
            socketio.emit('update', {'update': '[Memory Reset]', 'voice': 'user'},room=session_id)
            client[session_id]["prompt"] = 'Hi'
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
            socketio.emit('update', {'update': '/news\n[Fetching News]', 'voice': 'user'},room=session_id)
            context_str = get_news("none", 25)
            log(f"News Raw Context = {context_str}")
            client[session_id]["visible"] = False
            client[session_id]["remember"] = True
            client[session_id]["prompt"] = (
                "You are a newscaster who specializes in providing headline news. Use the following context provided by Google News to summarize the top 10 headlines for today. Include the source and rank them by most important to least important."
                f"\nContext: {context_str}"
                "\nAnswer:"
            )
        else:
            # Display help
            socketio.emit('update', {'update': '[Commands: /reset /version /sessions /news]', 'voice': 'user'},room=session_id)
            client[session_id]["prompt"] = ''
    
    # RAG Command - IMPORT from library - Format: #library [opt:number=1] [prompt]
    elif p.startswith("#"):
        parts = p[1:].split()
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
            socketio.emit('update', {'update': '[Usage: #{library} {opt:number} {prompt}] - Import and summarize topic from library.', 'voice': 'user'},room=session_id)
        else:
            if QDRANT_HOST:
                log(f"Pulling {number} entries from {library} with prompt {prompt}")
                socketio.emit('update', {'update': '%s [RAG Command Running...]' % p, 'voice': 'user'},room=session_id)
                # Query Vector Database for library
                results = query_index(prompt, library, top_k=number)
                context_str = ""
                client[session_id]["visible"] = False
                client[session_id]["remember"] = True
                for result in results:
                    context_str += f" <li> {result['title']}: {result['text']}\n"
                    log(" * " + result['title'])
                log(f" = {context_str}")
                client[session_id]["prompt"] = (
                    "Consider and summarize this information:\n"
                    f"{context_str}"
                    "\n"
                )
            else:
                socketio.emit('update', {'update': '[RAG Support Disabled - Check Config]', 'voice': 'user'},room=session_id)

    # RAG Commands - ANSWER from Library - Format: @library [opt:number=1] [prompt]
    elif p.startswith("@"):
        parts = p[1:].split()
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
            socketio.emit('update', {'update': '[Usage: @{library} {opt:number} {prompt} - Answer prompt based on library]', 'voice': 'user'},room=session_id)
        else:
            if QDRANT_HOST:
                log(f"Using library {library} with prompt {prompt}")
                socketio.emit('update', {'update': '%s [RAG Command Running...]' % p, 'voice': 'user'},room=session_id)
                # Query Vector Database for library
                results = query_index(prompt, library, top_k=number)
                context_str = ""
                client[session_id]["visible"] = False # Don't show prompt
                client[session_id]["remember"] = False # Don't add blog to context window, just summary
                for result in results:
                    context_str += f"{result['title']}: {result['text']}\n"
                    log(" * " + result['title'])
                client[session_id]["prompt"] = (
                    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
                    f"Question: {prompt}"
                    f"Context: {context_str}"
                    "Answer:"
                )
                # Alternative
                #     "[BEGIN]\n"
                #     f"{context_str}"
                #     "\n[END]\n"
                #     f"Using the text between [BEGIN] and [END] answer the question: {prompt}\n"
                #     "Answer:"
            else:
                socketio.emit('update', {'update': '[RAG Support Disabled - Check Config]', 'voice': 'user'},room=session_id)
    elif p.startswith(":"):
        # Use LLM to classify prompt and take action
        p = p[1:] # remove :
        context_str = ""
        prompttype = classify(p)
        log(f"Prompt type = {prompttype}")
        socketio.emit('update', {'update': '%s\n[Topic: %s]' % (p,prompttype), 'voice': 'user'},room=session_id)
        # check if prompttype string contains weather, stock, or news regardless of case
        if "weather" in prompttype.lower():
            # Weather prompt
            log("Weather prompt")
            location = clarify(f"What location is specified in this prompt, state None if there isn't one. Use a single word answer. [BEGIN] {p} [END]")
            if "none" in location.lower():
                context_str = get_weather("")
            else:
                context_str = get_weather(location)
        elif "stock" in prompttype.lower():
            # Stock prompt
            log("Stock prompt")
            company = clarify(f"What company is related to the stock price in this prompt? Please state none if there isn't one. Use a single word answer: [BEGIN] {p} [END]")
            context_str = get_stock(company)
            log(f"Company = {company} - Context = {context_str}")
            socketio.emit('update', {'update': context_str, 'voice': 'ai'},room=session_id)
            # remember context
            client[session_id]["context"].append({"role": "assistant", "content" : context_str})
            return jsonify({'status': 'Message received'})
        elif "news" in prompttype.lower():
            # News prompt
            log("News prompt")
            subject = clarify(f"What subject is specified in this prompt, state none if there isn't one. Use a single word answer. [BEGIN] {p} [END]")
            context_str = get_news(subject)
            log(f"Subject = {subject} - Context = {context_str}")
        else:
            # Normal prompt
            log("Normal prompt")
            context_str = ""
        if context_str == "":
            client[session_id]["prompt"] = p
        else:
            client[session_id]["visible"] = False
            client[session_id]["remember"] = True
            client[session_id]["prompt"] = (
                "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know."
                f"\nQuestion: {p}"
                f"\nContext: {context_str}"
                "\nAnswer:"
            )
    else:
        # Normal prompt
        client[session_id]["prompt"] = p
    return jsonify({'status': 'Message received'})

@app.route('/version')
def version():
    global VERSION, DEBUG
    if DEBUG:
        return jsonify({'version': "%s DEBUG MODE" % VERSION})
    return jsonify({'version': VERSION})

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
                client[session_id]["context"] = [{"role": "system", "content": baseprompt}]
            # Signal response is done
            socketio.emit('update', {'update': '', 'voice': 'done'},room=session_id)
            client[session_id]["prompt"] = ''
            if DEBUG:
                print(f"AI: {completion_text}")

def sigTermHandler(signum, frame):
    raise SystemExit

# Start server
if __name__ == '__main__':
    signal.signal(signal.SIGTERM, sigTermHandler);
    socketio.run(app, host='0.0.0.0', port=PORT, debug=DEBUG, allow_unsafe_werkzeug=True)

