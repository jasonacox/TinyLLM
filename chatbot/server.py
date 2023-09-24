#!/usr/bin/python3
"""
Web based ChatBot Example

Web chat client for OpenAI and the llama-cpp-python[server] OpenAI API Compatible 
Web Server. Provides a simple web based chat session.

Features:
  * Uses OpenAI API
  * Works with local hosted OpenAI compatible llama-cpp-python[server]
  * Retains conversational context for LLM
  * Uses response stream to render LLM chunks instead of waiting for full response

Requirements:
  * pip install openai flask flask-socketio

Running a llama-cpp-python server:
  * CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
  * pip install llama-cpp-python[server]
  * python3 -m llama_cpp.server --model models/7B/ggml-model.bin

Author: Jason A. Cox
23 Sept 2023
https://github.com/jasonacox/TinyLLM

"""
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import threading
import time
import openai
import datetime

# Configuration Settings - Showing local LLM
openai.api_key = "OPENAI_API_KEY"                # Required, use bogus string for Llama.cpp
openai.api_base = "http://10.0.1.89:8000/v1"     # Use API endpoint or comment out for OpenAI
agentname = "Jarvis"                             # Set the name of your bot
mymodel  ="models/7B/gguf-model.bin"             # Pick model to use e.g. gpt-3.5-turbo for OpenAI

# Configure Flask App and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Globals
prompt = ""

# Set base prompt and initialize the context array for conversation dialogue
current_date = datetime.datetime.now()
formatted_date = current_date.strftime("%m/%d/%Y")
baseprompt = "You are %s, a highly intelligent assistant. Keep your answers brief and accurate. Current date is %s." % (agentname, formatted_date)
context = [{"role": "system", "content": baseprompt}]

# Function - Send user prompt to LLM for response
def ask(prompt):
    global context
    # remember context
    context.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
        model=mymodel,
        max_tokens=1024,
        stream=True, # Send response chunks as LLM computes next tokens
        temperature=0.7,
        messages=context,
    )
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    global prompt
    # Handle incoming user prompts and store them
    data = request.json
    print("Received Data:", data)
    prompt = data["prompt"]
    return jsonify({'status': 'Message received'})

# Function to send updates to all connected clients
def send_update():
    global x, prompt

    while True:
        if prompt == "":
            time.sleep(.5)
        else:
            update_text = prompt 
            socketio.emit('update', {'update': update_text, 'voice': 'user'})
            # Ask LLM for answers
            response=ask(prompt)
            completion_text = ''
            # iterate through the stream of events and print it
            for event in response:
                event_text = event['choices'][0]['delta']
                if 'content' in event_text:
                    chunk = event_text.content
                    completion_text += chunk
                    socketio.emit('update', {'update': chunk, 'voice': 'ai'})
            socketio.emit('update', {'update': '', 'voice': 'done'})
            prompt = ''
            # remember context
            context.append({"role": "assistant", "content" : completion_text})

# Create a background thread to send updates
update_thread = threading.Thread(target=send_update)
update_thread.daemon = True  # Thread will terminate when the main program exits
update_thread.start()

if __name__ == '__main__':
    socketio.run(app, debug=True)
