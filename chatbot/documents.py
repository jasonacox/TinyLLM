#!/usr/bin/python3
"""
Web based ChatBot - Documents Handling Module

This module is responsible for ingesting and managing documents in the
vector database, weaviate. It uses the Weaviate Python client to interact
with the weaviate instance.

Class Documents:
    create: Create a collection in weaviate
    delete: Delete a collection in weaviate
    connect: Connect to the weaviate instance
    close: Close the weaviate connection
    list_documents: List all documents in collection with file as the key
    get_document: Get a document by its ID
    get_documents: Get a document by ID or query
    delete_document: Delete a document by its ID
    add_document: Ingest a document into weaviate
    update_document: Update a document in weaviate by its ID
    add_file: Detect and convert document, filename argument
    add_url: Import URL document
    add_pdf: Add a PDF document
    add_docx: Add a DOCX document
    add_txt: Add a TXT document

Utility Functions:
    extract_text_from_url: Extract text from a URL
    extract_text_from_pdf: Extract text from a PDF
    extract_text_from_text: Extract text from a text file
    extract_text_from_html: Extract text from an HTML file

Requirements:
    !pip install weaviate-client 

Author: Jason A. Cox
16 September 2024
https://github.com/jasonacox/TinyLLM

"""

# Imports
import os
import io
import logging

import weaviate.classes as wvc
import weaviate
import requests
from pdfreader import PdfReader
from bs4 import BeautifulSoup
import pypandoc

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log(msg):
    logger.info(msg)

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

    def __init__(self, host, filepath="/tmp", port=8080, grpc_port=50051):
        """
        Initialize the Document class
        """
        # Weaviate client object
        self.host = host
        self.filepath = filepath
        self.port = port
        self.grpc_port = grpc_port
        self.client = None
        # Verify file path
        if not os.path.exists(filepath):
            raise Exception('File path does not exist')

    def create(self, collection):
        """
        Create a collection in weaviate
        """
        if not self.client:
            self.connect()
        # Create a collection
        r = self.client.collections.create(
            name=collection,
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers()
        )
        log(f"Collection created: {collection}")
        return r

    def delete(self, collection):
        """
        Delete a collection in weaviate
        """
        if not self.client:
            self.connect()
        # Delete a collection
        r = self.client.collections.delete(collection)
        log(f"Collection deleted: {collection}")
        return r

    def connect(self):
        """
        Connect to the weaviate instance
        """
        try:
            self.client = weaviate.connect_to_local(
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port,
                additional_config=weaviate.config.AdditionalConfig(timeout=(15, 115))
            )
            log(f"Connected to Weaviate at {self.host}")
            return True
        except Exception as er:
            log(f"Unable to connect to Weaviate at {self.host}: {str(er)}")
            return False

    def close(self):
        """
        Close the weaviate connection
        """
        if self.client:
            self.client.close()

    def list_documents(self, collection=None):
        """
        List all documents in collection with file as the key
        """
        # List all documents in collection
        if not self.client:
            self.connect()
        # Get list of documents in collection
        collection = self.client.collections.get(collection)
        documents = {}
        for o in collection.iterator():
            p = o.properties
            uuid = str(o.uuid)
            filename = p.get("file")
            title = p.get("title")
            doc_type = p.get("doc_type")
            if filename not in documents:
                documents[filename] = {}
            documents[filename][uuid] = {
                "title": title,
                "doc_type": doc_type,                
            }
        return documents

    def get_document(self, collection, uuid):
        """
        Return a document by its ID
        """
        # Get a document by its ID - list fist element if list
        document = self.client.documents.get(collection, uuid) 
        if document and isinstance(document, list):
            document = document[0]
        return document

    def get_documents(self, collection, uuid=None, query=None, num_results=10):
        """
        Return a document by ID or query
        """
        docs = []
        if not self.client:
            self.connect()
        if uuid:
            # Get a document by its ID
            response = collection.query.fetch_object_by_id(uuid)
            p = response.properties
            docs.append({
                "uuid": uuid,
                "file": p.get("file"),
                "title": p.get("title"),
                "doc_type": p.get("doc_type"),
                'content': p.get("content"),
            })
        if query:
            # Search by vector query
            query = "List facts about solar."
            num_results = 5
            docs = self.client.collections.get(collection)
            results = docs.query.near_text(
                query=query,
                limit=num_results
            )
            for i in results.objects:
                p = i.properties
                uuid = str(i.uuid)
                docs.append( {
                    "uuid": uuid,
                    "file": p.get("file"),
                    "title": p.get("title"),
                    "doc_type": p.get("doc_type"),
                    'content': p.get("content"),
                })
        return docs

    def delete_document(self, collection, uuid):
        """
        Delete a document by its ID
        """
        # Delete a document by its ID
        if not self.client:
            self.connect()
        collection = self.client.collections.get(collection)
        return collection.data.delete_by_id(uuid)

    def add_document(self, collection, title, doc_type, filename, content):
        """
        Add a document into weaviate
        """
        if not self.client:
            self.connect()
        if not (title and doc_type and filename and content):
            raise ValueError('Missing document properties')
        # Ingest a document
        documents = [{
            "title": title,
            "doc_type": doc_type,
            "file": filename,
            "content": content
        }]
        c = self.client.collections.get(collection)
        r = c.data.insert_many(documents)
        log(f"Document added: {filename}")
        return r

    def update_document(self, collection, uuid, title, doc_type, filename, content):
        """
        Update a document in weaviate by its ID
        """
        self.delete_document(collection, uuid)
        return self.add_document(collection, title, doc_type, filename, content)

    def add_file(self, collection, title, filename, tmp_file=None):
        """
        Detect and convert document into weaviate
        """
        # is filename a URL?
        if filename.startswith("http"):
            return self.add_url(collection, title, filename)
        else:
            # Detect what type of file (case insensitive)
            if filename.lower().endswith('.pdf'):
                # PDF document
                return self.add_pdf(collection, title, filename, tmp_file)
            elif filename.lower().endswith('.docx'):
                # DOCX document
                return self.add_docx(collection, title, filename, tmp_file)
            elif filename.lower().endswith('.txt'):
                # TXT document
                return self.add_txt(collection, title, filename, tmp_file)
            else:
                # Unsupported document
                raise ValueError('Unsupported document format')

    def add_url(self, collection, title, url):
        """
        Import URL document
        """
        content = extract_from_url(url)
        if content:
            return self.add_document(collection, title, "URL", url, content)
        return None

    def add_pdf(self, collection, title, filename, tmp_file):
        """
        Add a PDF document from a local file
        """
        # Convert PDF to text document
        pdf_content = open(tmp_file, 'rb').read()
        pdf2text = ""
        f = io.BytesIO(pdf_content)
        reader = PdfReader(f)
        for page in reader.pages:
            pdf2text = page.extract_text() + "\n"
            # TODO: Break into chunks
            section = title + " - Page " + str(page.page_number)
            r = self.add_document(collection, section, "PDF", filename, pdf2text)
        return r

    def add_docx(self, collection, title, filename, tmp_file):
        """
        Add a DOCX document
        """
        # Convert DOCX file to text document
        docx2text = pypandoc.convert_file(tmp_file, 'plain', format='docx')
        # TODO: Break into chunks
        r = self.add_document(collection, title, "DOCX", filename, docx2text)
        return r

    def add_txt(self, collection, title, filename, tmp_file):
        """
        Add a TXT document
        """
        # Read text from TXT file
        with open(tmp_file, 'r') as f:
            txt2text = f.read()
        r = self.add_document(collection, title, "TXT", filename, txt2text)
        return r

# Utility functions

def extract_from_url(url):
    """
    Extract text from a URL and return the content
    """
    # Function - Extract text from PDF
    def extract_text_from_pdf(response):
        # Convert PDF to text
        pdf_content = response.read()
        pdf2text = ""
        f = io.BytesIO(pdf_content)
        reader = PdfReader(f)
        for page in reader.pages:
            pdf2text = pdf2text + page.extract_text() + "\n"
        return pdf2text

    # Function - Extract text from text
    def extract_text_from_text(response):
        return response.text()

    # Function - Extract text from HTML
    def extract_text_from_html(response):
        html_content = response.text()
        # get title of page from html
        source = "Document Source: " + str(response.url)
        soup = BeautifulSoup(html_content, 'html.parser')
        title = ("Document Title: " + soup.title.string + "\n") if soup.title else ""
        paragraphs = soup.find_all(['p', 'code', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre', 'ol'])
        website_text = f"{title}{source}\nDocument Content:\n" + '\n\n'.join([p.get_text() for p in paragraphs])
        return website_text

    try:
        with requests.Session() as session:
            response = session.get(url, allow_redirects=True)
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
                    "text/html": extract_text_from_html,
                    "application/xml": extract_text_from_text,
                }
                if content_type in content_handlers:
                    return content_handlers[content_type](response)
                else:
                    return "Unsupported content type"
            else:
                m = f"Failed to fetch the webpage. Status code: {response.status_code}"
                log(m)
                return m
    except Exception as err:
        log(f"An error occurred: {str(err)}")
        return None
