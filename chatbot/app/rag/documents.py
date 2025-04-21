#!/usr/bin/python3
"""
Web based ChatBot - Documents Handling Module

This module is responsible for ingesting and managing documents in the
vector database, Weaviate. It uses the Weaviate Python client to interact
with the Weaviate instance.

Class Documents:
    create: Create a collection in Weaviate
    delete: Delete a collection in Weaviate
    connect: Connect to the Weaviate instance
    close: Close the Weaviate connection
    all_collections: List all collections in Weaviate
    list_documents: List all documents in a collection with file as the key
    list_documents_stream: List all documents in a collection and stream the results
    list_chunks_stream: List all documents in a collection and stream the results
    get_document: Get a document by its ID
    get_documents: Get documents by ID, query, or filename
    delete_document: Delete a document by its ID or filename
    add_document: Ingest a document into Weaviate
    update_document: Update a document in Weaviate by its ID
    add_file: Detect and convert document, filename argument
    add_url: Import URL document
    add_pdf: Add a PDF document
    add_docx: Add a DOCX document
    add_txt: Add a TXT document
    add_html: Add an HTML document
    add_json: Add a JSON document
    add_csv: Add a CSV document
    add_xml: Add an XML document
    add_xlsx: Add an XLSX document

Requirements:
    !pip install weaviate-client pdfreader bs4 pypandoc pypdf requests pandas openpyxl

Run Test:
    WEAVIATE_HOST=localhost python3 documents.py

Author: Jason A. Cox
16 September 2024
https://github.com/jasonacox/TinyLLM

"""
# pylint: disable=too-many-public-methods
# pylint: disable=redefined-outer-name

# Standard library imports
import io
import logging
import os
import sys
import time

# Third-party imports
import pandas as pd
import pypandoc
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import weaviate
import weaviate.classes as wvc # pylint: disable=unused-import
from weaviate.auth import AuthApiKey
from weaviate.classes.query import Filter
from weaviate.exceptions import WeaviateConnectionError

# optional - download pandoc
#from pypandoc.pandoc_download import download_pandoc
#download_pandoc()

# Logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def log(msg):
    logger.debug(msg)
    print(msg)

# Defaults
MAX_CHUNK_SIZE=256*4

# Data Schema
schema_properties=[
    {
        "name": "title",                     
        "description": "Title of Document",           
        "dataType": ["text"],
        "moduleConfig": {
            "text2vec-transformers": {
                "skip": False,
                "vectorizePropertyName": False,
            }
        },
    },
    {
        "name": "chunk",
        "dataType": ["text"],
        "description": "Chunk of Document",
        "moduleConfig": {
            "text2vec-transformers": {
                "skip": False,
                "vectorizePropertyName": False,
            }
        },
    },
    {
        "name": "doc_type",
        "dataType": ["text"],
        "description": "Document Type",
        "moduleConfig": {
            "text2vec-transformers": {
                "skip": True,
                "vectorizePropertyName": False,
            }
        },
    },
    {
        "name": "file",
        "dataType": ["text"],
        "description": "Document Filename",
        "moduleConfig": {
            "text2vec-transformers": {
                "skip": True,
                "vectorizePropertyName": False,
            }
        },
    },
    {
        "name": "content",
        "dataType": ["text"],
        "description": "Full Document Content",
        "moduleConfig": {
            "text2vec-transformers": {
                "skip": True,
                "vectorizePropertyName": False,
            }
        },
    },
    {
        "name": "creation_time",
        "dataType": ["number"],
        "description": "Document Creation Time",
        "moduleConfig": {
            "text2vec-transformers": {
                "skip": True,
                "vectorizePropertyName": False,
            }
        },
    },
]

# Document class
class Documents:
    """
    Documents class

    The Document class is responsible for managing documents in the vector
    database, weaviate. It uses the Weaviate Python client to interact with
    the weaviate instance.

    Attributes:
        client: Weaviate client object
    """

    def __init__(self, host="localhost", grpc_host=None, port=8080, grpc_port=50051, retry=3, filepath="/tmp",
                 cache_expire=60, auth_key=None, secure=False):
        """
        Initialize the Document class
        """
        # Weaviate client object
        self.host = host                        # Weaviate host IP address
        self.grpc_host = grpc_host              # Weaviate gRPC host IP address
        self.filepath = filepath                # File path for temporary document storage
        self.port = port                        # Weaviate port
        self.grpc_port = grpc_port              # Weaviate gRPC port
        self.client = None                      # Weaviate client object
        self.retry = retry                      # Number of times to retry connection
        self.cache = {}                         # Cache of documents
        self.cache_expire = cache_expire        # Cache expiration time
        self.auth_key = auth_key                # Weaviate API key
        self.secure = secure                    # Weaviate secure connection
        if not grpc_host:
            self.grpc_host = host
        # Verify file path
        if not os.path.exists(filepath):
            os.makedirs(filepath)

    def connect(self):
        """
        Connect to the weaviate instance
        """
        x = self.retry
        # Connect to Weaviate
        while x:
            weaviate_optional = {}
            if self.auth_key:
                weaviate_optional = {
                    'auth_credentials': AuthApiKey(self.auth_key)
                }
            try:
                self.client = weaviate.connect_to_custom(
                    http_host=self.host,
                    http_port=self.port,
                    http_secure=self.secure,
                    grpc_host=self.grpc_host,
                    grpc_port=self.grpc_port,
                    grpc_secure=self.secure,
                    additional_config=weaviate.config.AdditionalConfig(
                        timeout=(15, 115)
                    ),
                    **weaviate_optional
                )
                log(f"Connected to Weaviate at {self.host}")
                return True
            except WeaviateConnectionError as er:
                log(f"Connection error: {str(er)}")
                x -= 1
                time.sleep(1)
        if not x:
            raise WeaviateConnectionError(f"Unable to connect to Weaviate at {self.host}")
        return False

    def is_connected(self):
        """
        Check if the weaviate connection is active
        """
        if self.client:
            return True
        return False

    def close(self):
        """
        Close the weaviate connection
        """
        if self.client:
            self.client.close()
            log("Weaviate connection closed")
            self.client = None

    def all_collections(self):
        """
        List all collections in weaviate

        Returns:
            collections: List of collections
        """
        # Check cache
        if "collections" in self.cache and self.cache["collections"]["expires"] > time.time():
            return self.cache["collections"]["data"]
        if not self.client:
            self.connect()
        x = self.retry
        while x:
            try:
                c = []
                collections = self.client.collections.list_all(simple=True)
                for i in collections:
                    c.append(i)
                log(f"Collections: {c}")
                break
            except WeaviateConnectionError as er:
                log(f"Connection error: {str(er)}")
                self.connect()
                time.sleep(1)
                x -= 1
        if not x:
            raise WeaviateConnectionError("Unable to connect to Weaviate")
        # Cache the result
        self.cache["collections"] = {
            "data": c,
            "expires": time.time() + self.cache_expire
        }
        return c

    def create(self, collection):
        """
        Create a collection in weaviate
        """
        x = self.retry
        if not self.client:
            self.connect()
        # Verify it does not exist
        collections = self.all_collections()
        if collection.title() in collections:
            log(f"Collection already exists: {collection}")
            return False
        # Create a collection
        while x: # retry until success
            try:
                schema = {
                    "class": collection,
                    "description": "AutoCreated by TinyLLM",
                    "vectorizer": "text2vec-transformers",
                    "properties": schema_properties,
                }
                self.client.collections.create_from_dict(schema)
                #self.client.collections.create(
                #    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers())
                # Invalidate cache
                if "collections" in self.cache:
                    self.cache["collections"]["expires"] = 0
                log(f"Collection created: {collection}")
                break
            except WeaviateConnectionError as er:
                log(f"Connection error: {str(er)}")
                self.connect()
                time.sleep(1)
                x -= 1
        if not x:
            raise WeaviateConnectionError("Unable to connect to Weaviate")
        return True

    def delete(self, collection):
        """
        Delete a collection in weaviate
        """
        x = self.retry
        if not self.client:
            self.connect()
        # Verify it does not exist
        collections = self.all_collections()
        if collection.lower() not in [c.lower() for c in collections]:
            log(f"Collection does not exist: {collection}")
            return False
        # Delete a collection
        while x:
            try:
                self.client.collections.delete(collection)
                log(f"Collection deleted: {collection}")
                # Invalidate cache
                if "collections" in self.cache:
                    self.cache["collections"]["expires"] = 0
                break
            except WeaviateConnectionError as er:
                log(f"Connection error: {str(er)}")
                self.connect()
                x -= 1
                time.sleep(1)
        if not x:
            raise WeaviateConnectionError("Unable to connect to Weaviate")
        return True

    def list_documents(self, collection=None):
        """
        List all documents in collection with file as the key

        Args:
            collection: Collection name

        Returns:
            documents: Dictionary of documents with filename as the key
            {
                "filename.txt": {
                    "uuid": {
                        "title": "title", 
                        "doc_type": "doc_type",
                        "creation_time": creation_time
                    }
                }
            }
        """
        x = self.retry
        documents = {}
        # List all documents in collection
        if not self.client:
            self.connect()
        # Get list of documents in collection
        while x:
            try:
                collection = self.client.collections.get(collection)
                for o in collection.iterator():
                    p = o.properties
                    uuid = str(o.uuid)
                    filename = p.get("file")
                    title = p.get("title")
                    doc_type = p.get("doc_type")
                    creation_time = p.get("creation_time")
                    if filename not in documents:
                        documents[filename] = {}
                    documents[filename][uuid] = {
                        "title": title,
                        "doc_type": doc_type,    
                        "creation_time": creation_time            
                    }
                break
            except WeaviateConnectionError as er:
                log(f"Connection error: {str(er)}")
                self.connect()
                x -= 1
                time.sleep(1)
        if not x:
            raise WeaviateConnectionError("Unable to connect to Weaviate")
        return documents

    def list_documents_stream(self, collection=None):
        """
        List all documents in collection and stream the results

        Args:
            collection: Collection name
        """
        x = self.retry
        documents = {}
        # List all documents in collection
        if not self.client:
            self.connect()
        # Get list of documents in collection
        while x:
            try:
                collection = self.client.collections.get(collection)
                for o in collection.iterator():
                    p = o.properties
                    uuid = str(o.uuid)
                    filename = p.get("file")
                    title = p.get("title")
                    doc_type = p.get("doc_type")
                    creation_time = p.get("creation_time")
                    if filename not in documents:
                        documents[filename] = {}
                    documents[filename][uuid] = {
                        "title": title,
                        "doc_type": doc_type,    
                        "creation_time": creation_time            
                    }
                    yield { "filename": filename,
                        "uuid": uuid,
                        "title": title,
                        "doc_type": doc_type,
                        "creation_time": creation_time }
                break
            except WeaviateConnectionError as er:
                log(f"Connection error: {str(er)}")
                self.connect()
                x -= 1
                time.sleep(1)
        if not x:
            raise WeaviateConnectionError("Unable to connect to Weaviate")

    def list_chunks_stream(self, collection, filename=None):
        """
        List all documents in collection and stream the results

        Args:
            collection: Collection name
            filename: Filename to filter on
        """
        x = self.retry
        documents = []
        # List all documents in collection
        if not self.client:
            self.connect()
        # Get list of documents in collection
        while x:
            try:
                collection = self.client.collections.get(collection)
                for o in collection.iterator():
                    p = o.properties
                    fn = p.get("file")
                    if filename and fn != filename:
                        continue
                    uuid = str(o.uuid)
                    title = p.get("title")
                    doc_type = p.get("doc_type")
                    creation_time = p.get("creation_time")
                    chunk = p.get("chunk") or " "
                    chunk_size = len(chunk)
                    content_size = len(p.get("content"))
                    documents.append({
                        "title": title,
                        "doc_type": doc_type,    
                        "creation_time": creation_time,
                        "uuid": uuid,
                        "chunk_size": chunk_size,
                        "content_size": content_size
                    })
                    yield {
                        "title": title,
                        "doc_type": doc_type,    
                        "creation_time": creation_time,
                        "uuid": uuid,
                        "chunk_size": chunk_size,
                        "content_size": content_size
                    }
                break
            except WeaviateConnectionError as er:
                log(f"Connection error: {str(er)}")
                self.connect()
                x -= 1
                time.sleep(1)
        if not x:
            raise WeaviateConnectionError("Unable to connect to Weaviate")

    def get_document(self, collection, uuid):
        """
        Return a document by its ID
        """
        document = None
        x = self.retry
        if not self.client:
            self.connect()
        # Get a document by its ID - list fist element if list
        while x:
            try:
                c = self.client.collections.get(collection)
                udocs = c.query.fetch_objects(
                    filters=Filter.by_id().equal(uuid),
                )
                p = udocs.objects[0].properties
                document ={
                    "uuid": uuid,
                    "file": p.get("file"),
                    "title": p.get("title"),
                    "chunk": p.get("chunk"),
                    "doc_type": p.get("doc_type"),
                    "content": p.get("content"),
                    "creation_time": p.get("creation_time"),
                }
                break
            except WeaviateConnectionError as er:
                log(f"Connection error (retry {x}): {str(er)}")
                self.connect()
                x -= 1
                time.sleep(1)
        if not x:
            raise WeaviateConnectionError("Unable to connect to Weaviate")
        return document

    def get_documents(self, collection, uuid=None, query=None, filename=None, num_results=10):
        """
        Return a document by ID, query or filename
        """
        dd = []
        x = self.retry
        if not self.client:
            self.connect()
        if uuid:
            # Get a document by its ID
            dd = [self.get_document(collection, uuid)]
        if query:
            # Search by vector query
            while x:
                try:
                    qdocs = self.client.collections.get(collection)
                    r = qdocs.query.near_text(
                        query=query,
                        limit=num_results
                    )
                    for i in r.objects:
                        p = i.properties
                        uuid = str(i.uuid)
                        dd.append( {
                            "uuid": uuid,
                            "file": p.get("file"),
                            "title": p.get("title"),
                            "chunk": p.get("chunk"),
                            "doc_type": p.get("doc_type"),
                            "content": p.get("content"),
                            "creation_time": p.get("creation_time"),
                        })
                    break
                except WeaviateConnectionError as er:
                    log(f"Connection error (retry {x}): {str(er)}")
                    self.connect()
                    x -= 1
                    time.sleep(1)
            if not x:
                raise WeaviateConnectionError("Unable to connect to Weaviate")
        if filename:
            # Get a document by its filename
            log(f"Getting documents by filename: {filename}")
            r = self.list_documents(collection)
            log(f"Documents: {r}")
            for fn in r:
                log(f"Checking filename: {fn}")
                if fn == filename:
                    for doc_uuid in r[fn]:
                        log(f"Getting document by filename: {filename} - uuid: {doc_uuid}")
                        dd.append(self.get_document(collection, doc_uuid))
        return dd

    def delete_document(self, collection, uuid=None, filename=None):
        """
        Delete a document by its ID or filename
        """
        r = None
        x = self.retry
        # Delete a document by its ID
        if not self.client:
            self.connect()
        while x:
            try:
                c = self.client.collections.get(collection)
                if uuid:
                    r = c.data.delete_by_id(uuid)
                    log(f"Document deleted: {uuid}")
                elif filename:
                    # Delete a document by its filename
                    documents = self.list_documents(collection)
                    for f in documents:
                        if f == filename:
                            # delete all UUIDs for this filename
                            for u in documents[f]:
                                r = c.data.delete_by_id(u)
                                log(f"Document deleted: {filename} - uuid: {u}")
                else:
                    raise ValueError('Missing document ID or filename')
                break
            except WeaviateConnectionError as er:
                log(f"Connection error (retry {x}): {str(er)}")
                self.connect()
                x -= 1
                time.sleep(1)
        if not x:
            raise WeaviateConnectionError("Unable to connect to Weaviate")
        return r

    def add_document(self, collection, title, doc_type, filename, chunk=None, content=None, chunk_size=MAX_CHUNK_SIZE):
        """
        Add a document into weaviate

        Inputs:
            collection: Collection name
            title: Document title
            doc_type: Document type
            filename: Document filename
            chunk: Document chunk - Part of the document
            content: Document content - Full text of the document
        """
        log(f"Adding document: {filename} - {title} - {doc_type} - {chunk} - {content} - {chunk_size}")
        log(f"Collection: {collection} - Doc size: {len(content)}")
        r = None
        dd = []
        if not chunk and not content:
            raise ValueError('Missing document content')
        if not content:
            content = chunk
        if not (title and doc_type and filename and content):
            raise ValueError('Missing document properties')
        if not chunk and chunk_size > 0:
            # Auto break up content into chunks
            chunks = break_up_content(content, chunk_size)
            ci = 0
            total_chunks = len(chunks)
            for new_chunk in chunks:
                ci = ci + 1
                log(f"Creating chunk {ci} of {total_chunks}")
                if total_chunks > 1:
                    suffix = f" - Section {ci} of {total_chunks}"
                else:
                    suffix = ""
                dd.append({
                    "title": title + suffix,
                    "chunk": new_chunk,
                    "doc_type": doc_type,
                    "file": filename,
                    "content": content,
                    "creation_time": time.time()
                })
        else :
            dd.append({
                "title": title,
                "chunk": chunk,
                "doc_type": doc_type,
                "file": filename,
                "content": content,
                "creation_time": time.time()
            })
        x = self.retry
        if not self.client:
            self.connect()
        while x:
            try:
                c = self.client.collections.get(collection)
                # Do batches of 10 of dd at a time
                for i in range(0, len(dd), 10):
                    log(f"Embedding document batch: {i} to {i+10}")
                    r = c.data.insert_many(dd[i:i+10])
                log(f"Document added: {filename}")
                break
            except WeaviateConnectionError as er:
                log(f"Connection error (retry {x}): {str(er)}")
                self.connect()
                x -= 1
                time.sleep(1)
        if not x:
            raise WeaviateConnectionError("Unable to connect to Weaviate")
        return r

    def update_document(self, collection, uuid, title, doc_type, filename, chunk=None, content=None):
        """
        Update a document in weaviate by its ID
        """
        # Delete and re-add document
        x = self.retry
        r = None
        if not self.client:
            self.connect()
        while x:
            try:
                self.delete_document(collection, uuid)
                r = self.add_document(collection, title, doc_type, filename, chunk, content)
                log(f"Document updated: {uuid}")
                break
            except WeaviateConnectionError as er:
                log(f"Connection error (retry {x}): {str(er)}")
                self.connect()
                x -= 1
                time.sleep(1)
        if not x:
            raise WeaviateConnectionError("Unable to connect to Weaviate")
        return r

    def add_file(self, collection, title, filename, tmp_file=None, chunk_size=None):
        """
        Detect and convert document into weaviate
        """
        # is filename a URL?
        if filename.startswith("http"):
            # TODO: Break into chunks
            return self.add_url(collection, title, filename, chunk_size)
        else:
            # Detect what type of file (case insensitive)
            if filename.lower().endswith('.pdf'):
                # PDF document
                return self.add_pdf(collection, title, filename, tmp_file, chunk_size)
            elif filename.lower().endswith('.docx'):
                # DOCX document
                return self.add_docx(collection, title, filename, tmp_file, chunk_size)
            elif filename.lower().endswith('.txt'):
                # TXT document
                return self.add_txt(collection, title, filename, tmp_file, chunk_size)
            elif filename.lower().endswith('.html'):
                # HTML document
                return self.add_html(collection, title, filename, tmp_file, chunk_size)
            elif filename.lower().endswith('.json'):
                # JSON document
                return self.add_json(collection, title, filename, tmp_file, chunk_size)
            elif filename.lower().endswith('.csv'):
                # CSV document
                return self.add_csv(collection, title, filename, tmp_file, chunk_size)
            elif filename.lower().endswith('.xml'):
                # XML document
                return self.add_xml(collection, title, filename, tmp_file, chunk_size)
            elif filename.lower().endswith('.xlsx') or filename.lower().endswith('.xls'):
                # XLSX document
                return self.add_xlsx(collection, title, filename, tmp_file, chunk_size)
            else:
                # Unsupported document
                return False

    def add_url(self, collection, title, url, chunk_size=None):
        """
        Import URL document 
        """
        content = extract_from_url(url, title)
        if content:
            for i in range(len(content["page"])):
                self.add_document(collection, content["title"][i], "URL", url, content=content["page"][i], chunk_size=chunk_size)
            return True
        return False

    def add_pdf(self, collection, title, filename, tmp_file, chunk_size=None):
        """
        Add a PDF document from a local file
        """
        # Convert PDF to text document
        with open(tmp_file, 'rb') as file:
            pdf_content = file.read()
        pdf2text = ""
        pdf_file = io.BytesIO(pdf_content)
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            pdf2text = page.extract_text() + "\n"
            section = title + " - Page " + str(page.page_number+1)
            r = self.add_document(collection, section, "PDF", filename, content=pdf2text, chunk_size=chunk_size)
        return r

    def add_docx(self, collection, title, filename, tmp_file, chunk_size=None):
        """
        Add a DOCX document
        """
        # Convert DOCX file to text document
        docx2text = pypandoc.convert_file(tmp_file, 'plain', format='docx')
        # TODO: Break into pages
        r = self.add_document(collection, title, "DOCX", filename, content=docx2text, chunk_size=chunk_size)
        return r

    def add_txt(self, collection, title, filename, tmp_file, chunk_size=None):
        """
        Add a TXT document
        """
        # Read text from TXT file
        with open(tmp_file, 'r') as f:
            txt2text = f.read()
        r = self.add_document(collection, title, "TXT", filename, content=txt2text, chunk_size=chunk_size)
        return r

    def add_html(self, collection, title, filename, tmp_file, chunk_size=None):
        """
        Add a HTML document
        """
        # Read and convert html to text
        with open(tmp_file, 'r') as f:
            html2text = f.read()
        soup = BeautifulSoup(html2text, 'html.parser')
        title = soup.title.string
        paragraphs = soup.find_all(['p', 'code', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre', 'ol'])
        website_text = f"Document Title: {title}\nDocument Content:\n" + '\n\n'.join([p.get_text() for p in paragraphs])
        r = self.add_document(collection, title, "HTML", filename, content=website_text, chunk_size=chunk_size)
        return r

    def add_json(self, collection, title, filename, tmp_file, chunk_size=None):
        """
        Add a JSON document
        """
        # Read text from JSON file
        with open(tmp_file, 'r') as f:
            json2text = f.read()
        r = self.add_document(collection, title, "JSON", filename, content=json2text, chunk_size=chunk_size)
        return r

    def add_csv(self, collection, title, filename, tmp_file, chunk_size=None):
        """
        Add a CSV document
        """
        # Read text from CSV file
        with open(tmp_file, 'r') as f:
            csv2text = f.read()
        r = self.add_document(collection, title, "CSV", filename, content=csv2text, chunk_size=chunk_size)
        return r

    def add_xml(self, collection, title, filename, tmp_file, chunk_size=None):
        """
        Add a XML document
        """
        # Read text from XML file
        with open(tmp_file, 'r') as f:
            xml2text = f.read()
        r = self.add_document(collection, title, "XML", filename, content=xml2text, chunk_size=chunk_size)
        return r

    def add_xlsx(self, collection, title, filename, tmp_file, chunk_size=None):
        """
        Add a XLSX document - Spreadsheet
        """
        # Read all sheets into a dictionary of DataFrames
        sheets_dict = pd.read_excel(tmp_file, sheet_name=None)
        # Iterate through each sheet
        for sheet_name, df in sheets_dict.items():
            # Convert the DataFrame to JSON
            json_output = df.to_json(orient='records', indent=4)
            title_sheet = title + " - " + sheet_name
            r = self.add_document(collection, title_sheet, "XLSX", filename, content=json_output, chunk_size=chunk_size)
        return r

# End of document class

# Utility functions

# Function to break up content into chunks
def break_up_content(text, max_size):
    """Break up text into chunks of max_size."""
    if len(text) > max_size:
        # Break up text into lines and then into chunks
        lines = text.splitlines()
        result = []
        current_chunk = ""
        for line in lines:
            if len(current_chunk) + len(line) > max_size:
                result.append(current_chunk)
                current_chunk = ""
            current_chunk = current_chunk + line + "\n"
        result.append(current_chunk)
        return result
    return [text]

def extract_from_url(url, title):
    """
    Extract text from a URL and return the content
    """
    try:
        response = requests.get(url, allow_redirects=True, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        m = f"Failed to fetch the webpage. Error: {str(e)}"
        log(m)
        return None
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
        return content_handlers[content_type](response, title)
    else:
        return None

# Function - Extract text from PDF
def extract_text_from_pdf(response, title):
    # Convert PDF to text
    pdf_content = response.content
    chunked = {
        "source": response.url,
        "doc_type": "PDF",
        "page": [],
        "title": [],
    }
    pdf_f = io.BytesIO(pdf_content)
    reader = PdfReader(pdf_f)
    if not title:
        title = "PDF Document {response.url}"
    # Extract text from each page
    for page in reader.pages:
        page_text = page.extract_text()
        title_prefix = f"{title} - Page {page.page_number+1}"
        chunked["page"].append(page_text)
        chunked["title"].append(title_prefix)
    pdf_f.close()
    return chunked

# Function - Extract text from text
def extract_text_from_text(response, title):
    chunked = {
        "source": response.url,
        "doc_type": "TXT",
        "page": [],
        "title": [],
    }
    chunked["page"].append(response.text)
    chunked["title"].append(title)
    return chunked

# Function - Extract text from HTML
def extract_text_from_html(response, title):
    chunked = {
        "source": response.url,
        "doc_type": "HTML",
        "page": [],
        "title": [],
    }
    html_content = response.text
    # get title of page from html
    source = "Document Source: " + str(response.url)
    soup = BeautifulSoup(html_content, 'html.parser')
    if not title:
        title = ("Document Title: " + soup.title.string + "\n") if soup.title else ""
    paragraphs = soup.find_all(['p', 'code', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre', 'ol'])
    website_text = f"{title}{source}\nDocument Content:\n" + '\n\n'.join([p.get_text() for p in paragraphs])
    chunked["page"].append(website_text)
    chunked["title"].append(title)
    return chunked

# Main - Test
if __name__ == "__main__":
    print("Testing the document module")
    print("---------------------------")
    HOST = os.getenv("WEAVIATE_HOST", "localhost")
    # Test the document module
    print("Testing the document module")
    docs = Documents(host=HOST)
    print("Connecting to Weaviate")
    if not docs.connect():
        sys.exit(1)
    # Remove test collection
    print("Deleting test collection")
    docs.delete("test")
    print("Creating test collection")
    docs.create("test")
    # URL Test
    print("Adding a URL document")
    docs.add_file("test", "Twinkle, Twinkle, Little Star", "https://www.jasonacox.com/wordpress/archives/2141")
    # PDF Test
    print("Adding a PDF document")
    docs.add_file("test", "Wind the Clock", "test.pdf", "/tmp/tinyllm/test.pdf")
    # DOCX Test
    print("Adding a DOCX document")
    docs.add_file("test", "Wiring for Outcomes", "test.docx", "/tmp/tinyllm/test.docx")
    # TXT Test
    print("Adding a TXT document")
    docs.add_file("test", "Grid Bugs", "test.txt", "/tmp/tinyllm/test.txt")
    # List documents
    print("Listing documents")
    documents = docs.list_documents("test")
    print(f"   Number of files: {len(documents)}")
    for f in documents:
        print(f"{f}: {documents[f]}")
    # Get document
    print("Getting document with query: time")
    results = docs.get_documents("test", query="time", num_results=5)
    print(f"   Number: {len(results)}")
    uuid = []
    for d in results:
        print("   " + d["uuid"] + " - " + d["title"] + " - " + d["file"])
        if d["doc_type"] != "PDF":
            uuid.append(d["uuid"])
    # Update document
    print("Updating document")
    docs.update_document("test", uuid[0], "Replace Title", "TXT", "updated.txt", "Updated content")
    # Delete document by ID
    print("Deleting document by ID")
    docs.delete_document("test", uuid[1])
    # Delete document by filename
    print("Deleting document by filename: test.docx")
    docs.delete_document("test", filename="test.pdf")
    # Get number of documents
    print("Getting number of documents")
    documents = docs.list_documents("test")
    print(f"   Number: {len(documents)}")
    # Delete collection
    print("Deleting test collection")
    docs.delete("test")
    # Close connection
    docs.close()
    log("Test complete")
