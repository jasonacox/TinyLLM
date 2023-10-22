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
  * pip install openai flask flask-socketio bs4 
  * pip install qdrant-client sentence-transformers pydantic~=2.4.2

Environmental variables:
  * OPENAI_API_KEY - Required only for OpenAI
  * OPENAI_API_BASE - URL to OpenAI API Server or locally hosted version
  * AGENT_NAME - Name for Bot
  * AGENT_NAME - LLM Model to Use

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
import time
import datetime
import threading
import signal
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import openai
import qdrant_client as qc
import qdrant_client.http.models as qmodels
from sentence_transformers import SentenceTransformer

VERSION = "v0.5"

MAXTOKENS = 2048
TEMPERATURE = 0.7
MAXCLIENTS = 10

# Configuration Settings - Showing local LLM
openai.api_key = os.environ.get("OPENAI_API_KEY", "DEFAULT_API_KEY")            # Required, use bogus string for Llama.cpp
openai.api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1") # Use API endpoint or comment out for OpenAI
agentname = os.environ.get("AGENT_NAME", "Jarvis")                              # Set the name of your bot
mymodel = os.environ.get("LLM_MODEL", "models/7B/gguf-model.bin")                # Pick model to use e.g. gpt-3.5-turbo for OpenAI
DEBUG = os.environ.get("DEBUG", "False") == "True"
STMODEL = os.environ.get("ST_MODEL", "all-MiniLM-L6-v2")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "") # Empty = disable RAG support
DEVICE = os.environ.get("DEVICE", "cuda")
RESULTS = os.environ.get("RESULTS", 1)

# Sentence Transformer Setup
if QDRANT_HOST:
    print("Sentence Transformer starting...")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model = SentenceTransformer(STMODEL, device=DEVICE) 

    # Qdrant Setup
    print("Connecting to Qdrant DB...")
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
print("Starting server...")
app = Flask(__name__)
socketio = SocketIO(app)

# Globals
client = {}

# Set base prompt and initialize the context array for conversation dialogue
current_date = datetime.datetime.now()
formatted_date = current_date.strftime("%m/%d/%Y")
baseprompt = "You are %s, a highly intelligent assistant. Keep your answers brief and accurate. Current date is %s." % (agentname, formatted_date)
context = [{"role": "system", "content": baseprompt}]

# Function - Send user prompt to LLM for response
def ask(prompt, sid=None):
    global client

    response = False
    print(f"Context size = {len(context)}")
    while not response:
        try:
            # remember context
            client[sid]["context"].append({"role": "user", "content": prompt})
            response = openai.ChatCompletion.create(
                model=mymodel,
                max_tokens=MAXTOKENS,
                stream=True, # Send response chunks as LLM computes next tokens
                temperature=TEMPERATURE,
                messages=client[sid]["context"],
            )
        except openai.error.OpenAIError as e:
            print(f"ERROR {e}")
            client[sid]["context"].pop()
            if "maximum context length" in str(e):
                if len(prompt) > 1000:
                    # assume we have very large prompt - cut out the middle
                    prompt = prompt[:len(prompt)//4] + " ... " + prompt[-len(prompt)//4:]
                    print(f"Reduce prompt size - now {len(prompt)}")
                elif len(client[sid]["context"]) > 4:
                    # our context has grown too large, truncate the top
                    client[sid]["context"] = client[sid]["context"][:1] + client[sid]["context"][3:]
                    print(f"Truncate context: {len(client[sid]['context'])}")
                else:
                    # our context has grown too large, reset
                    client[sid]["context"] = [{"role": "system", "content": baseprompt}]   
                    print(f"Reset context {len(client[sid]['context'])}")
                    socketio.emit('update', {'update': '[Memory Reset]', 'voice': 'user'},room=sid)

    if not client[sid]["remember"]:
        client[sid]["remember"] =True
        client[sid]["context"].pop()
    return response

def extract_text_from_blog(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find and extract all text within paragraph (p) tags
            paragraphs = soup.find_all('p')

            # Concatenate the text from all paragraphs
            blog_text = '\n'.join([p.get_text() for p in paragraphs])

            return blog_text
        else:
            print(f"Failed to fetch the webpage. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

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
        print(f"Client reconnected: {session_id}")
    else:
        # New client connected
        print(f"Client connected: {session_id}")
        # Limit number of clients
        if len(client) > MAXCLIENTS:
            print(f"Too many clients connected: {len(client)}")
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
    print(f"Client disconnected: {session_id}")
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
    print(f'Received message from {session_id}', data)
    print("Received Data:", data)
    if session_id not in client:
        print(f"Invalid session {session_id}")
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
        print(f"* Reading {len(blogtext)} bytes {url}")
        socketio.emit('update', {'update': '[Reading: %s]' % url, 'voice': 'user'},room=session_id)
        client[session_id]["prompt"] = "Summarize the following text:\n" + blogtext
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
        else:
            # Display help
            socketio.emit('update', {'update': '[Commands: /reset /version /help]', 'voice': 'user'},room=session_id)
            client[session_id]["prompt"] = ''
    elif p.startswith("@"):
        # Lookup and pull in context from DB - Format: @library [number] [prompt]
        parts = p[1:].split()
        if len(parts) >= 3:
            library = parts[0]
            number = int(parts[1])
            prompt = ' '.join(parts[2:])
            
            if QDRANT_HOST:
                print(f"Pulling {number} entries from {library} with prompt {prompt}")
                socketio.emit('update', {'update': '[%s...]' % p, 'voice': 'user'},room=session_id)
                # Query Vector Database for library
                results = query_index(prompt, library, top_k=number)
                context_str = ""
                client[session_id]["visible"] = False
                client[session_id]["remember"] = True
                for result in results:
                    context_str += f" <li> {result['title']}: {result['text']}\n"
                    print(" * " + result['title'])
                print(f" = {context_str}")
                client[session_id]["prompt"] = (
                    f"Consider and summarize this information found on {prompt}:\n"
                    f"{context_str}"
                    "\n"
                )
        else:
            socketio.emit('update', {'update': '[Usage: @{library} {number} {prompt}]', 'voice': 'user'},room=session_id)
    elif p.startswith("!"):
        # RAG Commands - Format: !library [prompt]
        library = p[1:].split(" ")[0]
        if library and len(p[1:].split(" ", 1)) > 1:
            prompt = p[1:].split(" ", 1)[1]
        if not library or not prompt:
            socketio.emit('update', {'update': '[Usage: !{library} {prompt}]', 'voice': 'user'},room=session_id)
        else:
            if QDRANT_HOST:
                print(f"Using library {library} with prompt {prompt}")
                socketio.emit('update', {'update': '[RAG Prompt: Reading %s...]' % library, 'voice': 'user'},room=session_id)
                # Query Vector Database for library
                results = query_index(prompt, library, top_k=RESULTS)
                context_str = ""
                client[session_id]["visible"] = False # Don't show prompt
                client[session_id]["remember"] = False # Don't add blog to context window, just summary
                for result in results:
                    context_str += f"{result['title']}: {result['text']}\n"
                    print(" * " + result['title'])
                client[session_id]["prompt"] = (
                    "[BEGIN]\n"
                    f"{context_str}"
                    "\n[END]\n"
                    f"Using the text between [BEGIN] and [END] answer the question: {prompt}\n"
                )
            else:
                socketio.emit('update', {'update': '[RAG Support Disabled - Check Config]', 'voice': 'user'},room=session_id)
    else:
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
    print(f"Starting send_update thread for {session_id}")

    # Verify session is valid
    if session_id not in client:
        print(f"Invalid session {session_id}")
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
                    event_text = event['choices'][0]['delta']
                    if 'content' in event_text:
                        chunk = event_text.content
                        completion_text += chunk
                        if DEBUG:
                            print(string_to_hex(chunk), end="")
                            print(f" = [{chunk}]")
                        socketio.emit('update', {'update': chunk, 'voice': 'ai'},room=session_id)
                # remember context
                client[session_id]["context"].append({"role": "assistant", "content" : completion_text})
            except:
                # Unable to process prompt, give error
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
    socketio.run(app, host='0.0.0.0', port=5000, debug=DEBUG, allow_unsafe_werkzeug=True)

