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
WEAVIATE_HOST = os.getenv('WEAVIATE_HOST', 'localhost')
WEAVIATE_GRPC_HOST = os.getenv('WEAVIATE_GRPC_HOST', WEAVIATE_HOST)
WEAVIATE_PORT = os.getenv('WEAVIATE_PORT', '8080')
WEAVIATE_GRPC_PORT = os.getenv('WEAVIATE_GRPC_PORT', '50051')
COLLECTIONS = os.getenv('COLLECTIONS', None)
PORT = int(os.getenv('PORT', "5001"))
COLLECTIONS_ADMIN = os.environ.get("COLLECTIONS_ADMIN", "true").lower() == "true"

# Set up pandocs - Needed to convert documents to text
#from pypandoc.pandoc_download import download_pandoc
#download_pandoc()

# Create a new instance of the Documents class to manage the database
documents = Documents(host=WEAVIATE_HOST, grpc_host=WEAVIATE_GRPC_HOST, port=WEAVIATE_PORT, grpc_port=WEAVIATE_GRPC_PORT)
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

# Get a title from a URL
def get_title_from_url(url):
    response = requests.get(url, verify=False)
    if response.status_code != 200:
        return None
    content_type = response.headers.get('content-type', '')
    print(content_type)
    if 'pdf' in content_type:
        pdf_content = BytesIO(response.content)
        pdf = PdfReader(pdf_content)
        print(pdf.metadata)
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
            title = get_title_from_url(url)
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
    llm_generated = form.get('llm_generated') == "1"
    auto_chunk = form.get('auto_chunk') == "1"
    # Prepare the document for ingest
    # TODO: llm_generated
    if not auto_chunk:
        chunk_size = 0
    collection = request.cookies.get('collection', collection)
    if not validate_collection(collection):
        collection = 'test'
    documents.add_file(collection, title, filename, tmp_filename, chunk_size=chunk_size)
    documents.close()
    # Delete the temporary file
    if tmp_filename and os.path.exists(tmp_filename):
        os.remove(tmp_filename)
    # Redirect to the index page - send a 302 redirect
    return templates.TemplateResponse(request, "redirect.html")

@app.get("/view", response_class=HTMLResponse)
async def view_file(request: Request):
    filename = request.query_params.get('filename')
    chunks = []
    collection = request.cookies.get('collection')
    if not validate_collection(collection):
        collection = 'test'
    # Get the document from the database
    dlist = documents.get_documents(collection, filename=filename)
    # Sort based on creation_time if it exists
    # is dlist a list of dictionaries?
    if dlist and type(dlist) is dict:
        if dlist[0] and dlist[0].get('creation_time'):
            dlist = sorted(dlist, key=lambda x: (x.get('creation_time')) or 0, reverse=False)
        else:
            dlist = sorted(dlist, key=lambda x: x.get('title'), reverse=False)
    for d in dlist:
        zuuid = d.get('uuid', '')
        title = d.get('title') or 'No Title'
        doc_type = d.get('doc_type', '')
        content = d.get('content') or ' '
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
    content = (d.get('content') or ' ').replace("\n", "\\n ")
    chunk = (d.get('chunk') or ' ').replace("\n", "\\n ")
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
    response = templates.TemplateResponse(request, "index.html", {"request": request})
    response.set_cookie(key="collection", value=collection)
    return response

@app.get("/get_collections", response_class=HTMLResponse)
async def get_collections(request: Request):
    collections = documents.all_collections()
    documents.close()
    # Return the list of collections in alphabetical order
    return json.dumps(sorted(collections))

@app.post("/new_collection")
async def new_collection(request: Request):
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
        documents.close()
    else:
        r = "You do not have permission to create a collection."
    return r

@app.post("/delete_collection")
async def delete_collection(request: Request):
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
        documents.close()
    return r

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
        "WEAVIATE_HOST": WEAVIATE_HOST,
        "WEAVIATE_GRPC_HOST": WEAVIATE_GRPC_HOST,
        "WEAVIATE_PORT": WEAVIATE_PORT,
        "WEAVIATE_GRPC_PORT": WEAVIATE_GRPC_PORT,
        "UPLOAD_FOLDER": UPLOAD_FOLDER,
        "MAX_CHUNK_SIZE": MAX_CHUNK_SIZE,
        "COLLECTIONS": COLLECTIONS,
        "PORT": PORT,
        "COLLECTIONS_ADMIN": COLLECTIONS_ADMIN,
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
