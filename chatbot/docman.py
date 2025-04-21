#!/usr/bin/python3
"""
TinyLLM Document Manager

The document manager allows you to manage the collections and documents in the 
Weaviate vector database. It provides an easy way for you to upload and ingest 
the content from files or URL. It performs simple chunking (if requested). The 
simple UI let's you navigate through the collections and documents.

Environment Variables:
- MAX_CHUNK_SIZE: Maximum size of a chunk in bytes (default 1024)
- UPLOAD_FOLDER: Folder where uploaded files are stored (default uploads)
- HOST: Weaviate host (default localhost)
- COLLECTIONS: Comma separated list of collections allowed (default all)
- PORT: Port for the web server (default 8000)
- COLLECTIONS_ADMIN: Allow users to create and delete collections (default True)

Setup:
- pip install fastapi uvicorn jinja2 bs4 pypdf requests lxml aiohttp
- pip install weaviate-client pdfreader pypandoc
- pip install python-multipart openpyxl
- pip install passlib bcrypt

Run:
- uvicorn docman:app --reload 

Author: Jason Cox
Date: 2024-09-21
TinyLLM: github.com/jasonacox/TinyLLM
"""

# Import Libraries
import json
import os
import time
import sys
import uuid
import datetime
from io import BytesIO
import logging

from chatbot.app.rag.documents import Documents

import requests
import socketio
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from passlib.context import CryptContext
import secrets
from contextlib import asynccontextmanager
from pypdf import PdfReader

from app.rag.documents import Documents

# TinyLLM Doc Manager Version
from app.core.config import VERSION

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("TinyLLM Document Manager %s" % VERSION)

def log(text):
    logger.info(text)

# Globals
stats = {"start_time": time.time()}
client = {}

# Environment variables
MAX_CHUNK_SIZE = int(os.getenv('MAX_CHUNK_SIZE', "1024"))
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
WEAVIATE_HOST = os.getenv('WEAVIATE_HOST', 'localhost')
WEAVIATE_GRPC_HOST = os.getenv('WEAVIATE_GRPC_HOST', WEAVIATE_HOST)
WEAVIATE_PORT = os.getenv('WEAVIATE_PORT', '8080')
WEAVIATE_GRPC_PORT = os.getenv('WEAVIATE_GRPC_PORT', '50051')
WEAVIATE_AUTH_KEY = os.getenv('WEAVIATE_AUTH_KEY', None)
WEAVIATE_TLS_ENABLE = os.getenv('WEAVIATE_TLS_ENABLE', "false").lower() == "true"
COLLECTIONS = os.getenv('COLLECTIONS', None)
PORT = int(os.getenv('PORT', "5001"))
COLLECTIONS_ADMIN = os.environ.get("COLLECTIONS_ADMIN", "true").lower() == "true"
MAXCLIENTS = int(os.environ.get("MAXCLIENTS", 1000))
USERNAME = os.environ.get("USERNAME", None) # Set to enable HTTP Basic Auth
PASSWORD = os.environ.get("PASSWORD", "nJjdh83")

# Create a new instance of the Documents class to manage the database
documents = Documents(host=WEAVIATE_HOST, grpc_host=WEAVIATE_GRPC_HOST, port=WEAVIATE_PORT, 
                      grpc_port=WEAVIATE_GRPC_PORT, auth_key=WEAVIATE_AUTH_KEY, secure=WEAVIATE_TLS_ENABLE)
try:
    if not documents.connect():
        print("Failed to connect to the vector database")
        sys.exit(1)
except Exception as e:
    print(f"Failed to connect to the vector database: {e}")
    sys.exit(1)

#
# Configure FastAPI App and SocketIO
#

log("Starting server...")
sio = socketio.AsyncServer(async_mode="asgi")
socket_app = socketio.ASGIApp(sio)
app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    log("Starting up...")
    yield
    log("Shutting down...")
    documents.close()

app.router.lifespan_context = lifespan

# Password hashing setup using Passlib
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Basic authentication scheme
security = HTTPBasic()

# Helper functions to verify passwords and find users
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_user(username: str):
    if username == USERNAME:
        return {"username": USERNAME, 
                "hashed_password": pwd_context.hash(PASSWORD),}
    return None

# Authenticate a user using HTTP Basic credentials
def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

# Dependency to check credentials
def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = False
    correct_password = False
    user = authenticate_user(credentials.username, credentials.password)
    if user:
        correct_username = secrets.compare_digest(credentials.username, user["username"])
        correct_password = verify_password(credentials.password, user["hashed_password"])
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create list of documents in the collection
def collection_files(collection=None):
    if not collection:
        return []
    d = documents.list_documents(collection)
    file_entries = []
    for doc in d:
        filename = doc
        count = len(d[doc])
        file_entries.append(({"filename": filename, "count": count}))
    return file_entries

# Create a Title from Filename
def create_title(filename):
    words = filename.replace("_"," ").replace("-"," ").split('.')[:-1]
    title =  ' '.join(words).title()
    # Check for double spaces and remove them
    while "  " in title:
        title = title.replace("  ", " ")
    return title

# Validate access to collection or set to default
def validate_collection(collection):
    if COLLECTIONS:
        if collection not in COLLECTIONS.split(","):
            # Select the first collection in the list
            collection = COLLECTIONS.split(",")[0]
    all_collections = documents.all_collections()
    if collection not in all_collections:
        # Select the first collection in the list
        collection = all_collections[0] if all_collections else None
    return collection

# Get a title from a URL
def get_title_from_url(url):
    response = requests.get(url, verify=False)
    if response.status_code != 200:
        return None
    content_type = response.headers.get('content-type', '')
    if 'pdf' in content_type:
        pdf_content = BytesIO(response.content)
        pdf = PdfReader(pdf_content)
        title = pdf.metadata.get('/Title') or pdf.metadata.get('/Subject') or pdf.metadata.get('/Author') or f'PDF Document {url}'
        return title
    elif 'html' in content_type:
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.title.string if soup.title else 'No title found'
    return None

#
# FastAPI Routes
#

templates = Jinja2Templates(directory="docman")

# Display the main chatbot page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user: dict = Depends(get_current_user if USERNAME else lambda: {"username": "nobody"})):
    # Check to see if the collection cookie is set
    if user:
        log(f"User {user['username']} logged in")
    collection = request.cookies.get('collection')
    collection = validate_collection(collection)
    return templates.TemplateResponse(request, "index.html",
                                      {"request": request,
                                       "collection": collection,
                                       "version": VERSION})

@app.post("/upload")
async def upload_file(request: Request, user: dict = Depends(get_current_user if USERNAME else lambda: {"username": "nobody"})):
    # Set defaults
    filename = ""
    tmp_filename = ""
    title = ""
    # Get the form data
    form = await request.form()
    collection = form['collection']
    url = form['url']
    if 'file' in form and not url:
        file = form['file']
        filename = f"{file.filename}"
        title = create_title(filename)
        # Create a temporary filename and save the file
        file_uuid = str(uuid.uuid4())
        if '.' in filename:
            file_extension = filename.split('.')[-1].lower()
        else:
            file_extension = "txt"
        tmp_filename = os.path.join(UPLOAD_FOLDER, f"{file_uuid}.{file_extension}")
        # Save the file
        with open(tmp_filename, "wb") as buffer:
            buffer.write(await file.read())
    elif url:
        filename = url
        title = url
        # Get title by pulling URL
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            title = get_title_from_url(url)
    return templates.TemplateResponse(request, "embed.html", {"request": request,
                                                     "filename": filename,
                                                     "tmp_filename": tmp_filename,
                                                     "title": title,
                                                     "chunk_size": MAX_CHUNK_SIZE,
                                                     "collection": collection,
                                                     "version": VERSION})

@app.post("/embed")
async def embed_file(request: Request, user: dict = Depends(get_current_user if USERNAME else lambda: {"username": "nobody"})):
    # Get the form data
    form = await request.form()
    filename = form['filename']
    tmp_filename = form['tmp_filename']
    title = form['title']
    collection = form['collection']
    chunk_size = int(form['chunk_size']) or MAX_CHUNK_SIZE
    llm_generated = form.get('llm_generated') == "1"
    auto_chunk = form.get('auto_chunk') == "1"
    # Prepare the document for ingest
    # TODO: llm_generated
    if not auto_chunk:
        chunk_size = 0
    collection = request.cookies.get('collection', collection)
    collection = validate_collection(collection)
    documents.add_file(collection, title, filename, tmp_filename, chunk_size=chunk_size)
    # Delete the temporary file
    if tmp_filename and os.path.exists(tmp_filename):
        os.remove(tmp_filename)
    # Redirect to the index page - send a 302 redirect
    return templates.TemplateResponse(request, "redirect.html")

@app.get("/view", response_class=HTMLResponse)
async def view_file(request: Request, user: dict = Depends(get_current_user if USERNAME else lambda: {"username": "nobody"})):
    filename = request.query_params.get('filename')
    collection = request.cookies.get('collection')
    collection = validate_collection(collection)
    # display view.html
    return templates.TemplateResponse(request, "view.html", {"request": request, 
                                                    "filename": filename, 
                                                    "collection": collection,
                                                    "version": VERSION})

@app.get("/view_chunk", response_class=HTMLResponse)
async def view_chunk(request: Request, user: dict = Depends(get_current_user if USERNAME else lambda: {"username": "nobody"})):
    zuuid = request.query_params.get('uuid')
    collection = request.cookies.get('collection')
    collection = validate_collection(collection)
    # Get the document from the database
    d = documents.get_documents(collection, uuid=zuuid)
    if d:
        d = d[0]
    filename = d.get('file', '')
    title = d.get('title', '')
    doc_type = d.get('doc_type', '')
    content = (d.get('content') or ' ').replace("\n", "\\n ")
    chunk = (d.get('chunk') or ' ').replace("\n", "\\n ")
    creation_time = d.get('creation_time') or "0"
    t = int(creation_time)
    creation_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))
    # display view_chunk.html
    return templates.TemplateResponse(request, "view_chunk.html", {"request": request,
                                                          "uuid": zuuid,
                                                          "filename": filename,
                                                          "title": title,
                                                          "doc_type": doc_type,
                                                          "content": content,
                                                          "creation_time": creation_time,
                                                          "chunk": chunk,
                                                          "collection": collection,
                                                          "version": VERSION})

#
# API Routes
#

@app.get("/get_uploaded_files", response_class=HTMLResponse)
async def get_uploaded_files(request: Request, user: dict = Depends(get_current_user if USERNAME else lambda: {"username": "nobody"})):
    collection = request.cookies.get('collection')
    collection = validate_collection(collection)
    file_entries = collection_files(collection)
    return json.dumps(file_entries)

@app.post("/select_collection")
async def select_collection(request: Request, user: dict = Depends(get_current_user if USERNAME else lambda: {"username": "nobody"})):
    form = await request.form()
    collection = form['collection']
    response = templates.TemplateResponse(request, "index.html", {"request": request})
    response.set_cookie(key="collection", value=collection)
    return response

@app.get("/get_collections", response_class=HTMLResponse)
async def get_collections(request: Request, user: dict = Depends(get_current_user if USERNAME else lambda: {"username": "nobody"})):
    collections = documents.all_collections()
    # Return the list of collections in alphabetical order
    return json.dumps(sorted(collections))

@app.post("/new_collection")
async def new_collection(request: Request, user: dict = Depends(get_current_user if USERNAME else lambda: {"username": "nobody"})):
    form = await request.form()
    collection = form['collection']
    # Verify user has access to create a collection
    if COLLECTIONS_ADMIN or (COLLECTIONS and collection in COLLECTIONS.split(",")):
        # Create the collection
        try:
            r = f"Collection {collection} already exists."
            if documents.create(collection):
                r = f"Collection {collection} created."
        except Exception as er:
            r = f"Failed to create collection: {collection}"
    else:
        r = "You do not have permission to create a collection."
    return r

@app.post("/delete_collection")
async def delete_collection(request: Request, user: dict = Depends(get_current_user if USERNAME else lambda: {"username": "nobody"})):
    form = await request.form()
    collection = form['collection']
    # Verify user has access to delete a collection
    if not COLLECTIONS_ADMIN or (COLLECTIONS and collection not in COLLECTIONS.split(",")):
        r = "You do not have permission to delete this collection."
    else:
        # Delete the collection
        try:
            r = f"Collection {collection} does not exist."
            if documents.delete(collection):
                r = f"Collection {collection} deleted."
        except Exception as er:
            r = f"Failed to delete collection: {collection}"
    return r

@app.post("/delete")
async def delete_file(request: Request, user: dict = Depends(get_current_user if USERNAME else lambda: {"username": "nobody"})):
    form = await request.form()
    filename = form['filename']
    collection = request.cookies.get('collection')
    collection = validate_collection(collection)
    # Delete the document from the database
    documents.delete_document(collection, filename=filename)
    return json.dumps({"status": "ok"})

# Version
@app.get("/version")
async def version():
    return {"version": VERSION}

# Display settings and stats
@app.get("/stats")
async def home(request: Request):
    global stats, client
    collection = request.cookies.get('collection')
    collection = validate_collection(collection)
    # Create a simple status page
    data = {
        "TinyLLM Chatbot Version": VERSION,
        "Start Time": datetime.datetime.fromtimestamp(stats["start_time"]).strftime("%Y-%m-%d %H:%M:%S"),
        "Uptime": str(datetime.timedelta(seconds=int(time.time() - stats["start_time"]))),
        "Collection": collection,
        "WEAVIATE_HOST": WEAVIATE_HOST,
        "WEAVIATE_GRPC_HOST": WEAVIATE_GRPC_HOST,
        "WEAVIATE_PORT": WEAVIATE_PORT,
        "WEAVIATE_GRPC_PORT": WEAVIATE_GRPC_PORT,
        "UPLOAD_FOLDER": UPLOAD_FOLDER,
        "MAX_CHUNK_SIZE": MAX_CHUNK_SIZE,
        "COLLECTIONS": COLLECTIONS,
        "PORT": PORT,
        "COLLECTIONS_ADMIN": COLLECTIONS_ADMIN,
        "MAXCLIENTS": MAXCLIENTS,
        "Session Count": len(client),
        "Connected": documents.is_connected(),
        "Collections": documents.all_collections(),
    }
    # Build a simple HTML page based on data facets
    html = "<html><head><title>TinyLLM Document Managher</title>"
    html += "<style>body { font-family: Helvetica, Arial, sans-serif; }</style>"
    html += "</head><body>"
    html += "<h1>TinyLLM Document Manager Status</h1>"
    # Provide link to project
    html += "<p>Settings and Current Status for <a href='https://github.com/jasonacox/TinyLLM/tree/main/chatbot'>TinyLLM DocMan</a></p>"
    html += "<table>"
    for key in data:
        html += f"<tr><td>{key}</td><td>{data[key]}</td></tr>"
    html += "</table>"
    # Add JS to refresh page every 5 seconds
    html += "<script>setTimeout(function(){location.reload()},5000);</script>"
    html += "</body></html>"
    return HTMLResponse(content=html, status_code=200)

# Serve static socket.io.js
@app.get("/socket.io.js")
def serve_socket_io_js():
    return FileResponse("docman/socket.io.js", media_type="application/javascript")

#
# SocketIO Events
#

app.mount("/", socket_app)  # Here we mount socket app to main fastapi app

# Client connected - start thread to send updates
@sio.on('connect')
async def handle_connect(session_id, env):
    await sio.emit('connected', {'data': f'Connected to TinyLLM Document Manager {VERSION}',
                                 'version': VERSION}, room=session_id)
    if session_id in client:
        log(f"Client reconnected: {session_id}")
    else:
        # New client connected
        log(f"Client connected: {session_id}")
        # Limit number of clients
        if len(client) > MAXCLIENTS:
            log(f"Too many clients connected: {len(client)}")
            await sio.emit('update', {'update': 'Too many clients connected. Try again later.', 'voice': 'user'},room=session_id)
            return
        # Create client session
        client[session_id]={}
        # Initialize context for this client
        client[session_id]["collection"] = None
        client[session_id]["stop_thread_flag"] = False

# Client disconnected
@sio.on('disconnect')
async def handle_disconnect(session_id):
    log(f"Client disconnected: {session_id}")
    # Remove client
    if session_id in client:
        # shutdown thread
        client[session_id]["stop_thread_flag"] = True
        client.pop(session_id)

# Client request for refreshCollections
@sio.on('message')
async def handle_message(session_id, data):
    log(f"Message from {session_id}: {data}")
    # Ensure client is connected
    if session_id not in client:
        # Add client
        client[session_id]={}
        client[session_id]["stop_thread_flag"] = False
    # Remember the collection
    collection = data.get("collection")
    collection = validate_collection(collection)
    client[session_id]["collection"] = collection
    # Process the message
    if data.get("request") == "refreshCollections":
        # Get the list of collections
        client[session_id]["stop_thread_flag"] = True
        collections = documents.all_collections()
        current_collection = client[session_id]["collection"]
        print(f"Current Collection: {current_collection}")
        # Check for valid collection
        if current_collection not in collections:
            # Set to first collection
            current_collection = collections[0] if len(collections) > 0 else ''
        await sio.emit('refreshCollections', {'collections': collections,
                                              'collection': current_collection}, room=session_id)
        client[session_id]["stop_thread_flag"] = False
    elif data.get("request") == "refreshUploadedDocuments":
        if not documents.is_connected():
            documents.connect()
        # Get the handle of the file list generator and send the files as they are generated
        file_entries = {}
        payload = []
        number_of_docs = 0
        d = documents.list_documents_stream(collection)
        for doc in d:
            if session_id not in client:
                break
            if client[session_id]["stop_thread_flag"]:
                client[session_id]["stop_thread_flag"] = False
                break
            number_of_docs += 1
            filename = doc["filename"]
            if filename not in file_entries:
                file_entries[filename] = 0
            file_entries[filename] += 1
            # Build payload
            payload = []
            for filename in file_entries:
                count = file_entries[filename]
                payload.append(({"filename": filename, "count": count}))
            # Send update on number of docs loaded
            await sio.emit('refreshUploadedDocuments', {'files': [], 'collection': collection, 'loading': number_of_docs}, room=session_id)
        # Send the list of documents
        await sio.emit('refreshUploadedDocuments', {'files': payload, 'collection': collection}, room=session_id)
        # Check for empty list and send empty list
        if not file_entries:
            await sio.emit('refreshUploadedDocuments', {'files': [], 'collection': collection}, room=session_id)

# Client request for loadDocuments
@sio.on('loadDocuments')
async def handle_load_documents(session_id, data):
    log(f"Load Documents from {session_id}: {data}")
    filename = data.get("filename")
    collection = data.get("collection")
    collection = validate_collection(collection)
    # Get the list of documents
    chunks = []
    num = 0
    dlist = documents.list_chunks_stream(collection, filename)
    for d in dlist:
        num += 1
        zuuid = d.get('uuid', '')
        title = d.get('title') or 'No Title'
        doc_type = d.get('doc_type', '')
        content = d.get('content') or ' '
        creation_time = d.get('creation_time') or "0"
        chunk_size = d.get('chunk_size') or 0
        chunks.append({"uuid": zuuid, "title": title, "doc_type": doc_type, 
                       "content": content, "creation_time": creation_time,
                       "chunk_size": chunk_size})
        await sio.emit('chunks', {'chunks': [], 'collection': collection,
                            'loading': num, 'filename': filename,
                            'creation_time': creation_time}, room=session_id)
    # Send the full list of documents
    await sio.emit('chunks', {'chunks': chunks, 'collection': collection,
                        'loading': 0, 'filename': filename,
                        'creation_time': creation_time}, room=session_id)

# Start the server
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=PORT)
    documents.close()

# To start the server, run the following command:
# uvicorn docman:app --reload
