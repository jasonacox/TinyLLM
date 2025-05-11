"""
ChatBot - FastAPI / SocketIO Routes Definition

This module contains the FastAPI and SocketIO routes for the TinyLLM Chatbot.
It handles user interactions, including sending and receiving messages, uploading files,
and managing the chatbot's state.

Author: Jason A. Cox
20 Apr 2025
github.com/jasonacox/TinyLLM
"""
# pylint: disable=invalid-name
# pylint: disable=global-statement
# pylint: disable=global-variable-not-assigned

# Standard library imports
import asyncio
import base64
import datetime
import io
import json
import re
import time
from contextlib import asynccontextmanager
import requests

# Third-party imports
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import pillow_heif
import socketio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

# Local imports
# pylint: disable=unused-import
from app.core.config import (log, debug, stats, VERSION, API_KEY, API_BASE, LITELLM_PROXY,
                        LITELLM_KEY, AGENTNAME, MYMODEL, DEBUG, MAXCLIENTS,
                        MAXTOKENS, TEMPERATURE, PORT, PROMPT_FILE, PROMPT_RO,
                        USE_SYSTEM, TOKEN, ONESHOT, RAG_ONLY, EXTRA_BODY,
                        TOXIC_THRESHOLD, THINKING, THINK_FILTER, SEARXNG,
                        INTENT_ROUTER, WEB_SEARCH, WEAVIATE_LIBRARY, RESULTS,
                        WEAVIATE_HOST, WEAVIATE_GRPC_HOST, WEAVIATE_PORT,
                        WEAVIATE_GRPC_PORT, ALPHA_KEY, SWARMUI, IMAGE_MODEL,
                        IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CFGSCALE, IMAGE_STEPS,
                        IMAGE_SEED, IMAGE_TIMEOUT, baseprompt)
from app.rag.rag import (rag_documents, get_weather, get_stock, get_news,
                        extract_text_from_url, query_index)
from app.core.llm import (client, get_models, ask, ask_llm, ask_context)
from app.core.prompts import (prompts, base_prompt, default_prompts, save_prompts, expand_prompt)
from app.image import ImageGenerator

# Ensure pillow_heif is properly registered with PIL
pillow_heif.register_heif_opener()

# Configure FastAPI App and SocketIO
#
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, socketio_path="socket.io")
app = FastAPI()

# Rate limiter configuration
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Image generator instance
image_generator = ImageGenerator(host=SWARMUI, model=IMAGE_MODEL, width=IMAGE_WIDTH,
                                    height=IMAGE_HEIGHT, cfgscale=IMAGE_CFGSCALE,
                                    steps=IMAGE_STEPS, seed=IMAGE_SEED,
                                    gen_timeout=IMAGE_TIMEOUT)
if not image_generator.test_connection():
    log("Image generator not available - set SwarmUI server using SWARMUI environment variable")
    image_generator = None
else:
    log(f"Image generator activated - {SWARMUI}")

# Session management
SESSION_TIMEOUT = 3600  # 1 hour
last_activity = {}

def cleanup_old_sessions():
    """Clean up sessions that have been inactive for too long"""
    current_time = time.time()
    sessions_to_remove = []
    for session_id, last_time in last_activity.items():
        if current_time - last_time > SESSION_TIMEOUT:
            sessions_to_remove.append(session_id)
    for session_id in sessions_to_remove:
        if session_id in client:
            del client[session_id]
        del last_activity[session_id]

@asynccontextmanager
async def lifespan(app: FastAPI): # pylint: disable=unused-argument,redefined-outer-name
    log("Starting chatbot...")
    yield
    log("Shutting down chatbot...")
    rag_documents.close()

app.router.lifespan_context = lifespan

#
# FastAPI Routes
#

templates = Jinja2Templates(directory="app/templates")

# Display the main chatbot page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

# Serve static socket.io.js
@app.get("/socket.io.js")
def serve_socket_io_js():
    return FileResponse("app/templates/socket.io.js", media_type="application/javascript")

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
        "LLM Main User Queries": stats["ask"],
        "LLM Helper Queries": stats["ask_llm"],
        "LLM CoT Context Queries": stats["ask_context"],
        "OpenAI API URL (OPENAI_API_URL)": API_BASE if not LITELLM_PROXY else "Disabled",
        "OpenAI API Key (OPENAI_API_KEY)": "************" if API_KEY != "" else "Not Set",
        "LiteLLM Proxy (LITELLM_PROXY)": LITELLM_PROXY or "Disabled",
        "LiteLLM Secret Key (LITELLM_KEY)": "************" if LITELLM_KEY != "" else "Not Set",
        "Agent Name (AGENT_NAME)": AGENTNAME,
        "LLM Model (LLM_MODEL)": MYMODEL,
        "Debug Mode (DEBUG)": DEBUG,
        "Current Clients (MAXCLIENTS)": f"{len(client)} of {MAXCLIENTS}",
        "LLM Max Tokens to Send (MAXTOKENS)": MAXTOKENS,
        "LLM Temperature (TEMPERATURE)": TEMPERATURE,
        "Server Port (PORT)": PORT,
        "Saved Prompts (PROMPT_FILE)": PROMPT_FILE,
        "Read-Only Prompts (PROMPT_RO)": PROMPT_RO,
        "LLM System Tags in Prompts (USE_SYSTEM)": USE_SYSTEM,
        "Run without conversation context (ONESHOT).": ONESHOT,
        "RAG: Run in RAG Only Mode (RAG_ONLY)": RAG_ONLY,
        "RAG: Weaviate (WEAVIATE_HOST)": WEAVIATE_HOST,
        "RAG: Weaviate gRPC (WEAVIATE_GRPC_HOST)": WEAVIATE_GRPC_HOST,
        "RAG: Weaviate Port (WEAVIATE_PORT)": WEAVIATE_PORT,
        "RAG: Weaviate gRPC Port (WEAVIATE_GRPC_PORT)": WEAVIATE_GRPC_PORT,
        "RAG: default Library (WEAVIATE_LIBRARY)": WEAVIATE_LIBRARY,
        "RAG: Default Results Retrieved (RESULTS)": RESULTS,
        "Alpha Vantage API Key (ALPHA_KEY)": "************" if ALPHA_KEY != "" else "Not Set",
        "Toxicity Threshold (TOXIC_THRESHOLD)": TOXIC_THRESHOLD,
        "Extra Body Parameters (EXTRA_BODY)": EXTRA_BODY,
        "Thinking Mode (THINKING)": THINKING,
        "Think Tag Filter (THINK_FILTER)": THINK_FILTER,
        "SearXNG Search Engine (SEARXNG)": SEARXNG,
        "Intent Router (INTENT_ROUTER)": INTENT_ROUTER,
        "Web Search (WEB_SEARCH)": WEB_SEARCH,
        "SwarmUI Host (SWARMUI)": SWARMUI,
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
    global prompts, baseprompt, sio, TEMPERATURE, MAXTOKENS, AGENTNAME
    if PROMPT_RO:
        return ({"Result": "Prompts are read-only"})
    oldbaseprompt = prompts["baseprompt"]
    oldagentname = prompts["agentname"]
    debug(f"Received prompts: {data}")
    # Update prompts
    for key in data:
        prompts[key] = data[key]
    save_prompts()
    if oldbaseprompt != prompts["baseprompt"] or oldagentname != prompts["agentname"]:
        # Update baseprompt
        AGENTNAME = prompts["agentname"]
        current_date = datetime.datetime.now()
        formatted_date = current_date.strftime("%B %-d, %Y")
        values = {"agentname": AGENTNAME, "date": formatted_date}
        baseprompt = expand_prompt(prompts["baseprompt"], values)
    # Update TEMPERATURE and MAXTOKENS
    if "LLM_temperature" in data:
        TEMPERATURE = float(data["LLM_temperature"])
    if "LLM_max_tokens" in data:
        MAXTOKENS = int(data["LLM_max_tokens"])
    # Notify all clients of update
    debug("Base prompt updated - notifying users")
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
async def show_version_api():
    global VERSION, DEBUG
    debug(f"Version requested - DEBUG={DEBUG}")
    if DEBUG:
        return {'version': "%s DEBUG MODE" % VERSION}
    return {'version': VERSION, 'model': MYMODEL}

# Send an alert to all clients
@app.post('/alert')
async def alert(data: dict):
    # Send an alert to all clients
    # Make sure TOKEN is set and matches
    if "token" in data and "message" in data and data["token"] == TOKEN:
        debug(f"Received alert: {data}")
        await sio.emit('update', {'update': data["message"], 'voice': 'user'})
        return ({'status': 'Alert sent'})
    else:
        log(f"Alert: Invalid token or missing message: {data}")
        return ({'status': 'Invalid Token or missing message'})

# Return list of available models
@app.get('/models')
async def list_models():
    return get_models()

# Upload a file
@app.post('/upload')
@limiter.limit("15/minute")
async def upload_file(request: Request, file: UploadFile = File(...), session_id: str = Form(...)): # pylint: disable=unused-argument
    global client
    file_name = file.filename
    session_id = session_id.strip()

    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")

    # Update session activity
    last_activity[session_id] = time.time()
    cleanup_old_sessions()

    # Open the image, checking for HEIC format
    try:
        image = Image.open(io.BytesIO(content))
    except Exception as e:
        await sio.emit('update', {'update': f"Image error: {str(e)}", 'voice': 'user'}, room=session_id)
        return {"error": f"Unable to open image: {str(e)}"}
    # Resize image if height or width is greater than 1024
    if image.height > 1024 or image.width > 1024:
        image.thumbnail((1024, 1024))
    # Convert image to RGB if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # Save image to memory as JPEG
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    content = img_byte_arr.getvalue()
    # Convert image to base64
    image_data = base64.b64encode(content).decode('utf-8')
    # Validate session
    if session_id not in client:
        log(f"Invalid session {session_id}")
        return {"result": "Bad Session ID", "filename": file.filename, "size": len(content)}
    debug(f"Received image upload from {session_id} - {file_name} [{len(image_data)} bytes]")
    # Add to client session
    client[session_id]["image_data"] = image_data
    # Determine file size in a human-readable format
    file_size = len(content)
    if file_size < 1024:
        file_size = f"{file_size} bytes"
    elif file_size < 1024 * 1024:
        file_size = f"{file_size / 1024:.1f} KB"
    else:
        file_size = f"{file_size / 1024 / 1024:.1f} MB"
    update = f"Uploaded image: {file_name} [{file_size}]"
    await sio.emit('update', {'update': update, 'voice': 'user'}, room=session_id)
    return {"result": "Success", "filename": file.filename, "size": len(content), "image_data": image_data}

#
# SocketIO Events
#

app.mount("/", socket_app)  # Here we mount socket app to main fastapi app

# Helper function to check for repeating text from LLM
async def is_repeating_exact(text: str, window: int = 100, repeats: int = 5) -> bool:
    if len(text) < window * (repeats + 1):
        return False
    tail = text[-window:]
    return text[:-window].count(tail) >= repeats

# Client connected - start thread to send updates
@sio.on('connect')
async def handle_connect(session_id, env): # pylint: disable=unused-argument
    debug(f"Client connected: {session_id}")

    # Convert each character to its hex representation
    def string_to_hex(input_string):
        hex_values = [hex(ord(char)) for char in input_string]
        return hex_values

    # Continuous thread to send updates to connected clients
    async def send_update(session_id):
        global client
        debug(f"Starting send_update thread for {session_id}")

        # Verify session is valid
        if session_id not in client:
            debug(f"Invalid session {session_id}")
            return
        try:
            while not client[session_id]["stop_thread_flag"]:
                if client[session_id]["prompt"] == "":
                    await sio.sleep(0.1)
                else:
                    # Check to see of CoT is enabled but not while processing a file/image
                    client_cot = client[session_id]["cot"]
                    client_image_data = client[session_id]["image_data"]
                    client_visible = client[session_id]["visible"]
                    if client_cot and not client_image_data and client_visible:
                        try:
                            # Remember original prompt
                            client[session_id]["cot_prompt"] = client[session_id]["prompt"]
                            # Check to see if the prompt needs COT processing
                            cot_check = expand_prompt(prompts["chain_of_thought_check"], {"prompt": client[session_id]["prompt"]})
                            debug("Running CoT check")
                            # Ask LLM for answers
                            response = await ask_llm(cot_check, model=client[session_id]["model"])
                            if "a" in response.lower() or "d" in response.lower() or client[session_id]["cot_always"]:
                                debug("Running deep thinking CoT to answer")
                                # Build prompt for Chain of Thought and create copy of context
                                cot_prompt = expand_prompt(prompts["chain_of_thought"], {"prompt": client[session_id]["prompt"]})
                                temp_context = client[session_id]["context"].copy()
                                temp_context.append({"role": "user", "content": cot_prompt})
                                # Send thinking status to client and ask LLM for answer
                                await sio.emit('update', {'update': 'Thinking... ', 'voice': 'ai'},room=session_id)
                                answer = await ask_context(temp_context, model=client[session_id]["model"])
                                await sio.emit('update', {'update': '\n\n', 'voice': 'ai'},room=session_id)
                                # Load request for CoT conclusion into conversational thread
                                cot_prompt = expand_prompt(prompts["chain_of_thought_summary"], {"context_str": answer,
                                                                                                "prompt": client[session_id]["cot_prompt"]})
                                client[session_id]["prompt"] = cot_prompt
                        except Exception as erro:
                            log(f"CoT error - continuing with original prompt: {erro}")
                            await sio.emit('update', {'update': '\n\n', 'voice': 'ai'},room=session_id)
                    else:
                        client_cot = False
                    try:
                        # Ask LLM for answers
                        response= await ask(client[session_id]["prompt"],session_id, sio)
                        completion_text = ''
                        tokens = 0
                        in_thinking = client[session_id]["think"]
                        # Iterate through the stream of tokens and send to client
                        stime = time.time()
                        for event in response:
                            event_text = event.choices[0].delta.content
                            # check for no tokens or a string just full of nany number of newlines only
                            if tokens == 0 and event_text and event_text.strip() == "":
                                continue
                            if event_text:
                                chunk = event_text
                                completion_text += chunk
                                tokens += 1
                                # Check for repeating text
                                if await is_repeating_exact(completion_text):
                                    debug(f"Repeating text detected - ending - {completion_text}")
                                    await sio.emit('update', {'update': '[Repeating Text Detected - Ending]', 'voice': 'user'},room=session_id)
                                    break
                                # Check for thinking tags
                                if client[session_id]["think"]:
                                    if "<think>" in event_text:
                                        in_thinking = True
                                        await sio.emit('update', {'update': '', 'voice': 'ai'},room=session_id)
                                        continue
                                    elif "</think>" in event_text:
                                        in_thinking = False
                                        token = 0
                                        continue
                                    if in_thinking:
                                        continue
                                await sio.emit('update', {'update': chunk, 'voice': 'ai'}, room=session_id)
                        # Check to see if thinking filter blocked all tokens
                        if in_thinking:
                            # Disable thinking filter and reprocess
                            client[session_id]["think"] = False
                            await sio.emit('update', {'update': '', 'voice': 'ai'},room=session_id)
                            await sio.emit('update', {'update': '[No thinking tags found - disabling filter]', 'voice': 'user'},room=session_id)
                            # Send entire completion_text to client
                            await sio.emit('update', {'update': completion_text, 'voice': 'ai'},room=session_id)
                            tokens = len(completion_text)/4
                        # Update footer with stats
                        await sio.emit('update', {'update':
                                                  f"TinyLLM Chatbot {VERSION} - {client[session_id]['model']} - Tokens: {tokens} - TPS: {tokens/(time.time()-stime):.1f}",
                                                  'voice': 'footer', 'model': client[session_id]['model']},room=session_id)
                        # Check for link injection
                        if client[session_id]["links"]:
                            await sio.emit('update', {'update': json.dumps(client[session_id]["links"]), 'voice': 'links'},room=session_id)
                            client[session_id]["links"] = ""
                        # Check for references
                        if client[session_id]["references"]:
                            # remove any leading or trailing newlines
                            client[session_id]["references"] = client[session_id]["references"].strip()
                            await sio.emit('update', {'update': client[session_id]["references"], 'voice': 'ref'},room=session_id)
                            client[session_id]["references"] = ""
                        if not ONESHOT:
                            # If COT mode replace CoT context in conversation thread with user prompt
                            if client_cot:
                                client[session_id]["context"].pop()
                                client[session_id]["context"].append({"role": "user", "content": client[session_id]["cot_prompt"]} )
                            # Remember answer
                            client[session_id]["context"].append({"role": "assistant", "content" : completion_text})
                    except Exception as erro:
                        # Unable to process prompt, give error
                        log(f"ERROR {erro}")
                        await sio.emit('update', {'update': 'An error occurred - unable to complete.', 'voice': 'ai'},room=session_id)
                        # Reset context
                        client[session_id]["context"] = base_prompt()
                    # Signal response is done
                    await sio.emit('update', {'update': '', 'voice': 'done'},room=session_id)
                    client[session_id]["prompt"] = ''
                    debug(f"LLM -> client [{session_id}]: {completion_text}")
        except KeyError:
            debug(f"Thread ended: {session_id}")
        except Exception as erro:
            log(f"Thread error: {erro}")

    if session_id in client:
        # Client reconnected - restart thread
        #client[session_id]["thread"].join()
        debug(f"Client reconnected: {session_id}")
    else:
        # New client connected
        debug(f"New client connected: {session_id}")
        # Limit number of clients
        if len(client) > MAXCLIENTS:
            log(f"Too many clients connected: {len(client)}")
            await sio.emit('update', {'update': 'Too many clients connected. Try again later.', 'voice': 'user'},room=session_id)
            return
        # Create client session
        client[session_id]={}
        # Initialize context for this client
        client[session_id]["context"] = base_prompt()
        client[session_id]["remember"] = True
        client[session_id]["visible"] = True
        client[session_id]["prompt"] = ""
        client[session_id]["stop_thread_flag"] = False
        client[session_id]["references"] = ""
        client[session_id]["links"] = {}
        client[session_id]["toxicity"] = 0.0
        client[session_id]["rag_only"] = False
        client[session_id]["cot"] = THINKING
        client[session_id]["cot_always"] = False
        client[session_id]["library"] = WEAVIATE_LIBRARY
        client[session_id]["results"] = RESULTS
        client[session_id]["image_data"] = ""
        client[session_id]["model"] = MYMODEL
        client[session_id]["think"] = THINK_FILTER
        client[session_id]["internet"] = WEB_SEARCH
        client[session_id]["intent"] = INTENT_ROUTER
        # Start continuous task to send updates
        asyncio.create_task(send_update(session_id))

# Client disconnected
@sio.on('disconnect')
async def handle_disconnect(session_id):
    debug(f"Client disconnected: {session_id}")
    # Remove client
    if session_id in client:
        # shutdown thread
        client[session_id]["stop_thread_flag"] = True
        client.pop(session_id)

# Change the LLM model
@sio.on('model')
async def change_model(session_id, model):
    global client
    # Remember the current model
    current_model = client[session_id]["model"]
    if session_id in client:
        if model != current_model:
            # If empty model selected, use default
            if model == "":
                model = MYMODEL
            # Verify model is valid
            list_of_models = get_models()
            if model not in list_of_models:
                log(f"Requested invalid model {model}")
                if len(client[session_id]["context"]) > 2:
                    await sio.emit('update', {'update': f"Model not found: {model}", 'voice': 'user'}, room=session_id)
                return
            debug(f"Changing model for {session_id} to {model}")
            client[session_id]["model"] = model
        # Update footer
        await sio.emit('update', {'update': f"TinyLLM Chatbot {VERSION} - {model}", 'voice': 'footer',
                                  'model': model}, room=session_id)
        # Check to see if this is a new session
        log(f"context length: {len(client[session_id]['context'])}")
        if len(client[session_id]["context"]) > 2 and current_model != model:
            await sio.emit('update', {'update': f'[Model changed to {model}]', 'voice': 'user'}, room=session_id)
    else:
        log(f"Invalid session {session_id}")
        await handle_invalid_session(session_id)
    return {'status': 'Model changed'}

# Client sent a message - handle it
@sio.on('message')
async def handle_message(session_id, data):
    global client
    # Handle incoming user prompts and store them
    debug(f'Received message from {session_id}: {data}')
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

# Upload an image to context via socket - NOT USED
@sio.on('image_upload')
async def handle_image_upload(session_id, data):
    global client
    await sio.emit('update', {'update': 'Image uploaded', 'voice': 'user'}, room=session_id)
    file_name = data['fileName']
    image_data = data['data'].split(",")[1]  # Extract base64 part of the image
    debug(f"Received image upload from {session_id} - {file_name} [{len(image_data)} bytes]")
    # Verify that this is a valid image
    if not image_data.startswith("iVBORw0KGgoAAAANSUhE"):
        log(f"Invalid image data: {image_data[:20]}")
        await sio.emit('update', {'update': 'Invalid image data', 'voice': 'user'}, room=session_id)
        return
    # Add to client session
    client[session_id]["image_data"] = image_data
    # Send image back to client to display
    #await sio.emit('update', {'filename': file_name,
    #                            'image_data': image_data,
    #                            'voice': 'image'}, room=session_id)

# Client sent a request for conversation thread
@sio.on('request_conversation')
async def handle_request_conversation(session_id):
    global client
    # Send conversation context to client
    if session_id in client:
        debug(f"Sending full conversation context to {session_id}")
        await sio.emit('update', {'update': client[session_id]["context"], 'voice': 'conversation'},room=session_id)
    else:
        log(f"Invalid session {session_id}")
        await handle_invalid_session(session_id)


async def handle_invalid_session(session_id):
    await sio.emit('update', {'update': '[Session Unrecognized - Try Refresh]', 'voice': 'user'}, room=session_id)

async def handle_url_prompt(session_id, p):
    url = p.strip()
    client[session_id]["visible"] = False
    client[session_id]["remember"] = True
    await sio.emit('update', {'update': '%s [Reading...]' % url, 'voice': 'user'}, room=session_id)
    website_text = await extract_text_from_url(url)
    if website_text:
        debug(f"* Reading {len(website_text)} bytes {url}")
        url_encoded = requests.utils.quote(url)
        client[session_id]["prompt"] = expand_prompt(prompts["website"], {"url": url_encoded, "website_text": website_text})
    else:
        await sio.emit('update', {'update': '[ERROR: Unable to read URL]', 'voice': 'user'}, room=session_id)
        client[session_id]["prompt"] = ''

async def handle_command(session_id, p):
    command = p[1:].split(" ")[0].lower()
    if command == "":
        await sio.emit('update', {'update': '[Commands: /image /intent /model /news /rag /reset /search /sessions /stock /think /version /weather]', 'voice': 'user'}, room=session_id)
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
    elif command == "think":
        await handle_think_command(session_id, p)
    elif command == "model" or command == "models":
        await handle_model_command(session_id, p)
    elif command == "search":
        await handle_search_command(session_id, p)
    elif command == "intent":
        await handle_intent_command(session_id, p)
    elif command == "image":
        await handle_image_command(session_id, p)
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
    debug("News requested")
    topic = p[5:].strip() or "none"
    await sio.emit('update', {'update': '%s [Fetching News]' % p, 'voice': 'user'}, room=session_id)
    context_str, links = await get_news(topic, 25)
    debug(f"News Raw Context = {context_str}")
    client[session_id]["visible"] = False
    client[session_id]["remember"] = True
    client[session_id]["links"] = links
    client[session_id]["prompt"] = expand_prompt(prompts["news"], {"context_str": context_str})

async def handle_rag_command(session_id, p):
    """
    Options:
    /rag {library} {opt:number} {prompt}
    /rag on {library} {opt:number}
    /rag off
    /rag list
    """
    # If WEAVIATE_HOST is not set, return
    if not WEAVIATE_HOST:
        await sio.emit('update', {'update': '[RAG Support Disabled - Check Config]', 'voice': 'user'}, room=session_id)
        return
    prompt = ""
    rag = p[4:].strip()
    parts = rag.split()
    library = ""
    # Do we have /rag on? - Get library and number - client session only not global
    if parts and parts[0] == "on":
        library = WEAVIATE_LIBRARY
        number = RESULTS
        if len(parts) >= 2:
            library = parts[1]
        if len(parts) >= 3 and parts[2].isdigit():
            number = int(parts[2])
        # Set mode in client session
        client[session_id]["rag_only"] = True
        client[session_id]["library"] = library
        client[session_id]["results"] = number
        await sio.emit('update', {'update': '[Auto-RAG On]', 'voice': 'user'}, room=session_id)
        return
    elif parts and parts[0] == "off":
        # Turn off RAG mode
        client[session_id]["rag_only"] = False
        await sio.emit('update', {'update': '[Auto-RAG Off]', 'voice': 'user'}, room=session_id)
        return
    elif parts and parts[0] == "list":
        # List available libraries
        array_of_libraries = rag_documents.all_collections()
        # convert array into string
        mes = f'[Available Libraries: {", ".join(array_of_libraries)}]'
        await sio.emit('update', {'update': mes, 'voice': 'user'}, room=session_id)
        return
    if len(parts) >= 2:
        library = parts[0]
        if parts[1].isdigit():
            number = int(parts[1])
            prompt = ' '.join(parts[2:])
        else:
            number = RESULTS
            prompt = ' '.join(parts[1:])
    if not library or not prompt:
        if RAG_ONLY:
            mes = f'Auto-RAG On: All prompts are processed by RAG using library {WEAVIATE_LIBRARY}\n.'
        else:
            library = WEAVIATE_LIBRARY if RAG_ONLY else client[session_id]["library"]
            number = RESULTS if RAG_ONLY else client[session_id]["results"]
            rag_state = f"[Auto-RAG is {'On' if client[session_id]['rag_only'] or RAG_ONLY else 'Off'} - Library: {library} - Results: {number}]\n\n"
            mes = rag_state + 'RAG Commands:\n * /rag {library} {opt:number} {prompt}\n * /rag on {library} {opt:number}\n * /rag off\n * /rag list\n'
        await sio.emit('update', {'update': mes, 'voice': 'user'}, room=session_id)
    else:
        if WEAVIATE_HOST:
            debug(f"Pulling {number} entries from {library} with prompt {prompt}")
            if not ONESHOT:
                await sio.emit('update', {'update': '%s [RAG Command Running...]' % p, 'voice': 'user'}, room=session_id)
            results, references = query_index(prompt, library, number)
            if results:
                context_str = ""
                client[session_id]["visible"] = False
                client[session_id]["remember"] = True
                context_str = results
                debug(f" = {references}")
                client[session_id]["references"] = references
                client[session_id]["prompt"] = expand_prompt(prompts["rag"], {"context_str": context_str, "prompt": prompt})
            else:
                await sio.emit('update', {'update': '[Unable to access Vector Database for %s]' % library, 'voice': 'user'}, room=session_id)
        else:
            await sio.emit('update', {'update': '[RAG Support Disabled - Check Config]', 'voice': 'user'}, room=session_id)

async def handle_think_command(session_id, p):
    """
    Options:
    /think on
    /think off
    /think always
    /think filter
    """
    think = p[6:].strip()
    parts = think.split()
    if parts and parts[0] == "on":
        client[session_id]["cot"] = True
        client[session_id]["cot_always"] = False
        await sio.emit('update', {'update': '[Chain of Thought Mode On]', 'voice': 'user'}, room=session_id)
        return
    elif parts and parts[0] == "off":
        client[session_id]["cot"] = False
        await sio.emit('update', {'update': '[Chain of Thought Mode Off]', 'voice': 'user'}, room=session_id)
        return
    elif parts and parts[0] == "always":
        client[session_id]["cot"] = True
        client[session_id]["cot_always"] = True
        await sio.emit('update', {'update': '[Chain of Thought Mode Always On]', 'voice': 'user'}, room=session_id)
        return
    elif parts and parts[0] == "filter":
        state = ""
        if len(parts) >= 2:
            if parts[1] == "on":
                client[session_id]["think"] = True
                state = "ON"
            else:
                client[session_id]["think"] = False
                state = "OFF"
        else:
            client[session_id]["think"] = not client[session_id]["think"]
            state = "ON" if client[session_id]["think"] else "OFF"
        await sio.emit('update', {'update': f'[Think Filter is {state}]', 'voice': 'user'}, room=session_id)
        return
    else:
        state = "ON" if client[session_id]["cot"] else "OFF"
        state = "ALWAYS" if client[session_id]["cot_always"] else state
        await sio.emit('update', {'update': f'Chain of Thought is {state}\n - Chain of Thought: /think {{on|off|always}}\n - Filter Think Tags: /think filter {{on|off}}', 'voice': 'user'}, room=session_id)

async def handle_model_command(session_id, p):
    # Remember current model
    current_model = client[session_id]["model"]
    # List or set LLM Models
    words = p.split()
    args = words[1] if len(words) > 1 else ""
    if not args:
         # Open Model Dialog
        await sio.emit('model_dialog', {}, room=session_id)
        return
    model_list = get_models()
    if not args in model_list:
        args = args.lower()
    if args in model_list and args != current_model:
        debug(f"Changing model to {args}")
        client[session_id]["model"] = args
        await sio.emit('update', {'update': f'[Model changed to {args}]', 'voice': 'user'}, room=session_id)
        await sio.emit('update', {'update': f"TinyLLM Chatbot {VERSION} - {args}", 'voice': 'footer',
                                  'model': args}, room=session_id)
    elif args in ["list", "help"]:
        msg = f'Current LLM Model: {client[session_id]["model"]}\n'
        msg += f'- Available Models: {", ".join(model_list)}\n'
        msg += '- Usage: /model {model_name}'
        await sio.emit('update', {'update': msg, 'voice': 'user'}, room=session_id)
    else:
        await sio.emit('update', {'update': f'[Model {args} not found]', 'voice': 'user'}, room=session_id)

async def handle_weather_command(session_id, p):
    debug("Weather prompt")
    await sio.emit('update', {'update': '%s [Weather Command Running...]' % p, 'voice': 'user'}, room=session_id)
    location = await ask_llm(expand_prompt(prompts["location"], {"prompt": p}), model=client[session_id]["model"])
    location = ''.join(e for e in location if e.isalnum() or e.isspace())
    if "none" in location.lower():
        context_str = await get_weather("")
    else:
        context_str = await get_weather(location)
    client[session_id]["visible"] = False
    client[session_id]["remember"] = True
    client[session_id]["prompt"] = expand_prompt(prompts["weather"], {"prompt": p, "context_str": context_str, "location": location})

async def handle_stock_command(session_id, p):
    debug("Stock prompt")
    prompt = p[6:].strip()
    if not prompt:
        await sio.emit('update', {'update': '[Usage: /stock {company}] - Fetch stock price for company.', 'voice': 'user'}, room=session_id)
        return
    await sio.emit('update', {'update': '%s [Fetching Stock Price...]' % p, 'voice': 'user'}, room=session_id)
    debug(f"Stock Prompt: {prompt}")
    company = await ask_llm(expand_prompt(prompts["company"], {"prompt": prompt}), model=client[session_id]["model"])
    company = ''.join(e for e in company if e.isalnum() or e.isspace())
    if "none" in company.lower():
        context_str = "Unable to fetch stock price - Unknown company specified."
    else:
        context_str = await get_stock(company, model=client[session_id]["model"])
    debug(f"Company = {company} - Context = {context_str}")
    await sio.emit('update', {'update': context_str, 'voice': 'ai'}, room=session_id)
    client[session_id]["context"].append({"role": "user", "content" : "What is the stock price for %s?" % company})
    client[session_id]["context"].append({"role": "assistant", "content" : context_str})
    client[session_id]["prompt"] = ''

async def handle_image_command(session_id, p):
    if not image_generator:
        await sio.emit('update', {'update': '[Image Generation is not enabled]', 'voice': 'user'}, room=session_id)
        return
    prompt = p[6:].strip()
    if not prompt:
        await sio.emit('update', {'update': '[Usage: /image {prompt}] - Generate image for prompt.', 'voice': 'user'}, room=session_id)
        return
    await sio.emit('update', {'update': '%s [Generating Image...]' % p, 'voice': 'user'}, room=session_id)
    debug(f"Image Prompt: {prompt}")
    client[session_id]["visible"] = False
    client[session_id]["remember"] = True
    # Generate image from image_generator

    image_encoded = await image_generator.generate(prompt)
    if image_encoded:
        image = Image.open(io.BytesIO(base64.b64decode(image_encoded.split(",")[1])))
        #image.show()
    else:
        await sio.emit('update', {'update': '[Unable to generate image]', 'voice': 'user'}, room=session_id)
        return
    # Resize image if height or width is greater than 1024
    if image.height > 1024 or image.width > 1024:
        image.thumbnail((1024, 1024))
    # Convert image to RGB if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # Save image to memory as JPEG
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    content = img_byte_arr.getvalue()
    # Convert image to base64
    image_data = base64.b64encode(content).decode('utf-8')
    # Add to client session
    client[session_id]["image_data"] = image_data
    # Send image back to client to display
    await sio.emit('update', {'filename': "generated.png",
                            'generated': True,
                            'image_data': image_data,
                            'voice': 'image'}, room=session_id)
    await sio.emit('update', {'update': '[Image Generated]', 'voice': 'user'}, room=session_id)
    # Create a prompt to ask LLM to describe image
    client[session_id]["prompt"] = "Describe the image in detail. The original prompt was: %s\n" % prompt

# Intent Engine Helpers

async def prompt_similarity(session_id, prompt1, prompt2):
    # Check for similarity between two prompts
    query = f"""You help compare two sentences to see if they are semantically similar. Please rate the similarity between 0 and 10 with 0 being not similar at all, 10 being equivalent:

    Sentence 1: {prompt1}

    Sentence 2: {prompt2}
    
    Provide the answer in JSON format with keys: score and justification."""
    answer = await ask_llm(query, model=client[session_id]["model"])
    # replace any markdown formatting for json code
    answer = answer.replace('```json', '')
    answer = answer.replace('```', '')
    # process JSON response looking for score and justification
    try:
        answer = json.loads(answer)
        score = int(answer.get("score", 0))
        justification = answer.get("justification", "None")
        debug(f"Similarity Score: {score} - Justification: {justification}")
        return score, justification
    except:
        debug(f"Unable to process response: {answer}")
        return 10, "Unable to process response"

intent_questions = {
    "weather": "Is this request asking about current weather conditions or forecast?",
    "news": "s this request asking about news?",
    "search": "Is this request asking to search the internet or web?",
    "stock": "Does this prompt contain a company name?",
    "dynamic": "Is is possible that the answer to this question changed over last 4 years, dependent on current events or news, or subject to change due to external factors?",
    "followup": "Does this seems like a follow-up question or statement?"
}

async def handle_normal_prompt(session_id, p):
    if client[session_id]["visible"]:
        await sio.emit('update', {'update': p, 'voice': 'user'}, room=session_id)
    if WEAVIATE_HOST and RAG_ONLY and p is not prompts["greeting"]:
        # Activate RAG every time
        await handle_rag_command(session_id, f"/rag {WEAVIATE_LIBRARY} {RESULTS} {p}")
        return
    if client[session_id]["rag_only"]:
        # Use RAG
        await handle_rag_command(session_id, f"/rag {client[session_id]['library']} {client[session_id]['results']} {p}")
        return
    if TOXIC_THRESHOLD != 99:
        # Test toxicity of prompt
        toxicity_check = await ask_llm(expand_prompt(prompts["toxic_filter"], {"prompt": p}), model=client[session_id]["model"])
        # extract floating point number from response
        toxicity = re.findall(r"[-+]?\d*\.\d+|\d+", toxicity_check)
        # convert float string to float
        try:
            toxicity = float(toxicity[0])
        except:
            toxicity = 0
        client[session_id]["toxicity"] = toxicity
        if toxicity > TOXIC_THRESHOLD:
            # Prompt is toxic - remove newlines from toxicity_check
            reason = toxicity_check.replace('\n', ' ')
            await sio.emit('update', {'update': f'[Filter Activated ({toxicity}) {reason}]\nPlease try a different topic.', 'voice': 'ai'}, room=session_id)
            client[session_id]["prompt"] = ""
            client[session_id]["toxicity"] = 0.0
            log(f"Toxic Prompt Detected [{toxicity}] - {p}")
            return
    if client[session_id]["intent"] and len(p) < 200 and not client[session_id]["image_data"]:
        # Intent Engine is enabled -  Give LLM context on previous prompt
        prompt_context = client[session_id]["context"][-2]["content"] if isinstance(client[session_id]["context"][-2]["content"], str) else ""
        prompt_context += client[session_id]["context"][-1]["content"] if isinstance(client[session_id]["context"][-1]["content"], str) else ""
        # remove "stock" and "news" from context
        prompt_context = prompt_context.replace("stock", "")
        prompt_context = prompt_context.replace("news", "")
        prompt_context += "\nNew Request: " + p
        # If it is just a greeting or pleasantry, skip intent detection
        answer = await ask_llm(f"PROMPT: {p}\nIs the user greeting or thanking us? (yes or no):", model=client[session_id]["model"])
        if "yes" in answer.lower():
            client[session_id]["prompt"] = p
            return
        intent = await ask_llm(expand_prompt(prompts["intent"], {"prompt": prompt_context}), model=client[session_id]["model"])
        if intent:
            intent = intent.lower().split()[-1]
            debug(f"Intent detected: {intent}")
            if "weather" in intent:
                # Get the weather
                if await double_check(session_id, p, intent_questions["weather"]):
                    await handle_weather_command(session_id, p)
                    return
            if "news" in intent:
                # Get news
                type_of_news = await ask_llm(f"What subject, company, person, place or type of news are they looking for in this request: <REQUEST>\n{prompt_context}\n</REQUEST>\nList a single word or state 'general news' if general news is requested. Otherwise list the company, placer, person or subject if given:", model=client[session_id]["model"])
                log(f"Type of news: {type_of_news}")
                score, _ = await prompt_similarity(session_id, p, "What is the current news?")
                if score > 1:
                    if "general news" in type_of_news.lower():
                        type_of_news = ""
                    await fetch_news(session_id, f"/news {type_of_news}")
                    return
            if "code" in intent:
                # Code requested
                client[session_id]["prompt"] = p
                return
            if "image" in intent:
                # Image requested
                image_prompt = await ask_llm(f"Create a single image generation prompt from the following request: <REQUEST>\n{prompt_context}\n</REQUEST>\n", model=client[session_id]["model"])
                if image_prompt:
                    # Add instructions to also list company stock ticker
                    p = f"/image {image_prompt}"
                    await handle_image_command(session_id, p)
                    return
            if "retry" in intent and SEARXNG:
                # Assume last answer was bad, find the previous prompt and ask again
                if len(client[session_id]["context"]) > 2:
                    # Grab prompt from context
                    last_prompt = client[session_id]["context"][-2]["content"]
                    if len(last_prompt) < 200:
                        await sio.emit('update', {'update': 'Let me search the internet...', 'voice': 'ai'}, room=session_id)
                        debug(f"Complaint detected - retrying: {last_prompt}")
                        if await process_search(session_id, last_prompt, prompt_context):
                            return
                    else:
                        client[session_id]["prompt"] = p
                return
            if await double_check(session_id, p, intent_questions["followup"]):
                # Follow-up question
                client[session_id]["prompt"] = p
                return
            if await double_check(session_id, p, intent_questions["dynamic"]):
                intent = "search"
            if SEARXNG and ("search" in intent or "stock" in intent):
                debug(f"Search the web: {p}")
                if "stock" in intent:
                    # Add instructions to also list company stock ticker
                    p = f"{p} - Please also list the stock ticker symbol if relevant."
                await process_search(session_id, p, prompt_context)
                return
    # Default: Process the prompt
    client[session_id]["prompt"] = p


async def double_check(session_id, prompt, question):
    # Ask LLM to double check the prompt
    check = await ask_llm(f"<PROMPT>\n{prompt}\n</PROMPT>\n{question} - Answer yes or no:", model=client[session_id]["model"])
    if check:
        check = check.lower()
        if "yes" in check:
            return True
    return False


async def process_search(session_id, prompt, context):
    # Ask LLM to determin topic to search
    search_topic = await ask_llm(f"Context: {context}\n\nCreate a single internet search phrase to best answer the fllowing prompt. Don't add commentary.\nPROMPT: {prompt}", model=client[session_id]["model"])
    if search_topic:
        search_topic = search_topic.strip()
        search_topic = search_topic.replace('"', '')
        # limit to first line only if newlines found
        if "\n" in search_topic:
            search_topic = search_topic.split("\n")[0]
        if search_topic:
            await handle_search_command(session_id, f"/search {search_topic}", original_prompt=prompt)
            return True
        else:
            return False


async def handle_intent_command(session_id, p):
    # Allow user to toggle intent engine
    if p == "/intent on":
        client[session_id]["intent"] = True
        await sio.emit('update', {'update': '[Intent Engine Enabled]', 'voice': 'user'}, room=session_id)
    elif p == "/intent off":
        client[session_id]["intent"] = False
        await sio.emit('update', {'update': '[Intent Engine Disabled]', 'voice': 'user'}, room=session_id)
    else:
        current_state = "ON" if client[session_id]["intent"] else "OFF"
        await sio.emit('update', {'update': '[Usage: /intent on|off] - Enable or disable the intent engine. Currently: ' + current_state, 'voice': 'user'}, room=session_id)

async def handle_search_command(session_id, p, original_prompt=""):
    # Check to see if SEARXNG is enabled
    if not SEARXNG:
        await sio.emit('update', {'update': '[Search Engine Disabled - Check Config]', 'voice': 'user'}, room=session_id)
        return
    # format /search {opt:max} {prompt} or /search on|off
    parts = p.split()
    max_results = 5
    prompt = ""
    # check if turn on/off
    if len(parts) == 2 and parts[0] == "/search" and parts[1] in ["on", "off"]:
        client[session_id]["internet"] = parts[1].lower() == "on"
        await sio.emit('update', {'update': f'[Auto Web Search is {"On" if client[session_id]["internet"] else "Off"}]', 'voice': 'user'}, room=session_id)
        return
    # check to see if optional max results is provided
    if len(parts) >= 2 and parts[1].isdigit():
        max_results = int(parts[1])
        prompt = ' '.join(parts[2:])
    else:
        prompt = ' '.join(parts[1:])
    if not prompt:
        current_state = "ON" if client[session_id]["internet"] else "OFF"
        await sio.emit('update', {'update': '[Usage: /search {opt:number} {query} or /search {on|off}] - Auto Web Search: ' + current_state, 'voice': 'user'}, room=session_id)
        return
    await sio.emit('update', {'update': '%s [Searching...]' % p, 'voice': 'user'}, room=session_id)
    if original_prompt:
        prompt = original_prompt
    context_str = await search_web(session_id, prompt, max_results) or "[Error searching the web]"
    client[session_id]["visible"] = False
    client[session_id]["remember"] = True
    client[session_id]["prompt"] = expand_prompt(prompts["internet_rag"], {"context_str": context_str, "prompt": prompt})

async def search_web(session_id, prompt, results=5):
    # Search the web using SEARXNG
    context_str = ""
    references = ""
    try:
        # Search the SEARXNG service with URI: /search?q={prompt}&format=json
        # use GET request to get the search results (max results)
        search_results = requests.get(f"{SEARXNG}/search?q={prompt}&format=json", timeout=10).json()
        count = 0
        for result in search_results.get('results', []):
            if count >= results:
                break
            count += 1
            context_str += f"* {result['title']}\n{result['url']}\n{result['content']}\n\n"
            # Build references
            if "url" in result:
                references = references + f"\n - {result['title']} - <a href='{result['url']}' target='_blank'>[Link]</a>"
        if not context_str:
            context_str = "[No search results found]"
        if references:
            client[session_id]["references"] = references
    except Exception as e:
        log(f"Error searching web: {e}")
        context_str = "[Error searching the web]"
    return context_str
