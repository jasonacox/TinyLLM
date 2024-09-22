#!/usr/bin/python3
"""
TinyLLM Document Manager

This is a simple document manager that allows users to manage their collectsion
of documents in a vector dtatabase (weaviate).  It allows users to upload files
and view the contents of the files.  It is a simple web application that uses
FastAPI and Jinja2 for the web interface.

Environment Variables:
- MAX_CHUNK_SIZE: Maximum size of a chunk in bytes (default 1024)
- UPLOAD_FOLDER: Folder where uploaded files are stored (default uploads)
- HOST: Weaviate host (default localhost)
- COLLECTIONS: Comma separated list of collections allowed (default all)
- PORT: Port for the web server (default 8000)

Setup:
- pip install fastapi uvicorn jinja2 python-multipart requests beautifulsoup4
- pip install weaviate-client
- pip install pdfreader

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

from documents import Documents

import requests
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from pypdf import PdfReader

from documents import Documents

# TinyLLM Doc Manager Version
from version import VERSION

# Stats
stats = {"start_time": time.time()}

# Environment variables
MAX_CHUNK_SIZE = int(os.getenv('MAX_CHUNK_SIZE', "1024"))
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
HOST = os.getenv('HOST', 'localhost')
COLLECTIONS = os.getenv('COLLECTIONS', None)
PORT = os.getenv('PORT', 5001)

# Globals
client = {}

# Create a new instance of the Documents class to manage the database
documents = Documents(host=HOST)
try:
    if not documents.connect():
        print("Failed to connect to the vector database")
        sys.exit(1)
except Exception as e:
    print(f"Failed to connect to the vector database: {e}")
    sys.exit(1)
documents.close()

# Start the server
#log("Starting server...")
#sio = socketio.AsyncServer(async_mode="asgi")
#socket_app = socketio.ASGIApp(sio)
app = FastAPI()

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create list of documents in the collection
def collection_files(collection=None):
    if not collection:
        return []
    d = documents.list_documents(collection)
    documents.close()
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

# Validate access to collection
def validate_collection(collection):
    if COLLECTIONS:
        if collection not in COLLECTIONS.split(","):
            return False
    if collection not in documents.all_collections():
        return False
    return True

#
# FastAPI Routes
#

templates = Jinja2Templates(directory="docman")

# Display the main chatbot page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Check to see if the collection cookie is set
    collection = request.cookies.get('collection')
    if validate_collection(collection):
        collection = "test"
    return templates.TemplateResponse(request, "index.html",
                                      {"request": request,
                                       "collection": collection,
                                       "version": VERSION})

@app.post("/upload")
async def upload_file(request: Request):
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
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else url
    return templates.TemplateResponse(request, "embed.html", {"request": request,
                                                     "filename": filename,
                                                     "tmp_filename": tmp_filename,
                                                     "title": title,
                                                     "chunk_size": MAX_CHUNK_SIZE,
                                                     "collection": collection,
                                                     "version": VERSION})

@app.post("/embed")
async def embed_file(request: Request):
    # Get the form data
    form = await request.form()
    filename = form['filename']
    tmp_filename = form['tmp_filename']
    title = form['title']
    collection = form['collection']
    chunk_size = int(form['chunk_size']) or MAX_CHUNK_SIZE
    llm_generated = form.get('llm_generated') == "on"
    auto_chunk = form.get('auto_chunk') == "on"
    # Prepare the document for ingest
    # TODO: llm_generated
    documents.set_max_chunk_size(chunk_size)
    if not auto_chunk:
        documents.set_max_chunk_size(0)
    collection = request.cookies.get('collection', collection)
    if not validate_collection(collection):
        collection = 'test'
    documents.add_file(collection, title, filename, tmp_filename)
    documents.close()
    # Delete the temporary file
    if tmp_filename and os.path.exists(tmp_filename):
        os.remove(tmp_filename)
    # Redirect to the index page
    return templates.TemplateResponse(request, "index.html")

@app.get("/view", response_class=HTMLResponse)
async def view_file(request: Request):
    filename = request.query_params.get('filename')
    chunks = []
    collection = request.cookies.get('collection')
    if not validate_collection(collection):
        collection = 'test'
    # Get the document from the database
    dlist = documents.get_documents(collection, filename=filename)
    for d in dlist:
        zuuid = d.get('uuid', '')
        title = d.get('title', '')
        doc_type = d.get('doc_type', '')
        content = d.get('content', '')
        creation_time = d.get('creation_time') or "0"
        chunks.append({"uuid": zuuid, "title": title, "doc_type": doc_type, 
                       "content": content, "creation_time": creation_time})
    documents.close()
    # display view.html
    return templates.TemplateResponse(request, "view.html", {"request": request, 
                                                    "filename": filename, 
                                                    "chunks": chunks,
                                                    "collection": collection,
                                                    "version": VERSION})

@app.get("/view_chunk", response_class=HTMLResponse)
async def view_chunk(request: Request):
    zuuid = request.query_params.get('uuid')
    collection = request.cookies.get('collection')
    if not validate_collection(collection):
        collection = 'test'
    # Get the document from the database
    d = documents.get_documents(collection, uuid=zuuid)
    if d:
        d = d[0]
    filename = d.get('file', '')
    title = d.get('title', '')
    doc_type = d.get('doc_type', '')
    content = d.get('content', '')
    chunk = d.get('chunk', '')
    creation_time = d.get('creation_time') or "0"
    t = int(creation_time)
    creation_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))
    documents.close()
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
async def get_uploaded_files(request: Request):
    collection = request.cookies.get('collection')
    if not validate_collection(collection):
        collection = 'test'
    file_entries = collection_files(collection)
    return json.dumps(file_entries)

@app.post("/select_collection")
async def select_collection(request: Request):
    form = await request.form()
    collection = form['collection']
    response = templates.TemplateResponse("index.html", {"request": request})
    response.set_cookie(key="collection", value=collection)
    return response

@app.get("/get_collections", response_class=HTMLResponse)
async def get_collections(request: Request):
    collections = documents.all_collections()
    documents.close()
    return json.dumps(collections)

@app.post("/delete")
async def delete_file(request: Request):
    form = await request.form()
    filename = form['filename']
    collection = request.cookies.get('collection')
    if not validate_collection(collection):
        collection = 'test'
    # Delete the document from the database
    documents.delete_document(collection, filename=filename)
    documents.close()
    return json.dumps({"status": "ok"})

# Version
@app.get("/version")
async def version():
    return {"version": VERSION}

# Display settings and stats
@app.get("/stats")
async def home(request: Request):
    global stats
    collection = request.cookies.get('collection')
    if not validate_collection(collection):
        collection = 'test'
    # Create a simple status page
    data = {
        "TinyLLM Chatbot Version": VERSION,
        "Start Time": datetime.datetime.fromtimestamp(stats["start_time"]).strftime("%Y-%m-%d %H:%M:%S"),
        "Uptime": str(datetime.timedelta(seconds=int(time.time() - stats["start_time"]))),
        "Collection": collection,
        "HOST": HOST,
        "UPLOAD_FOLDER": UPLOAD_FOLDER,
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

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=PORT)

# To start the server, run the following command:
# uvicorn docman:app --reload
