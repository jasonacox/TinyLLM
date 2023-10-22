#!/usr/bin/python3
"""
Fetch blog data from jasonacox.com and embed into a qdrant vector database. 
This uses a sentence transformer for the embedding calculations.

BATCH DOCUMENT VERSION - This version uploads a batch of documents at a time.

Author: Jason A. Cox
10 October 2023
https://github.com/jasonacox/TinyLLM/

Requirements:
    * pip install qdrant-client sentence-transformers 

Credits:
    * Jacob Marks - How I Turned My Company’s Docs into a Searchable Database with OpenAI
      https://towardsdatascience.com/how-i-turned-my-companys-docs-into-a-searchable-database-with-openai-4f2d34bd8736
    * Jason Fan - How to connect Llama 2 to your own data, privately
      https://jfan001.medium.com/how-to-connect-llama-2-to-your-own-data-privately-3e14a73e82a2

"""
import os
import re
import string
import uuid
from html import unescape

import httpx
import qdrant_client as qc
import qdrant_client.http.models as qmodels

from sentence_transformers import SentenceTransformer

# Configuration Settings
MODEL = os.environ.get("MY_MODEL", "all-MiniLM-L6-v2")
DEBUG = os.environ.get("DEBUG", "False") == "True"
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "mylibrary") 
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
RESULTS = 5

# Sentence Transformer Setup
print("Sentence Transformer starting...")
model = SentenceTransformer(MODEL, device="cuda") 

# Qdrant Setup
print("Connecting to Qdrant DB...")
client = qc.QdrantClient(url=QDRANT_HOST)
METRIC = qmodels.Distance.DOT
DIMENSION = model.get_sentence_embedding_dimension()

# Create embeddings for text
def embed_text(text):
    embeddings = model.encode(text, convert_to_tensor=True)
    return embeddings

# Initialize qdrant collection (will erase!)
def create_index():
    client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config = qmodels.VectorParams(
            size=DIMENSION,
            distance=METRIC,
        )
    )

# Creates vector for content with attributes
def create_vector(content, title, page_url, doc_type="text"):
    vector = embed_text(content)
    uid = str(uuid.uuid1().int)[:32]
    # Document attributes
    payload = {
        "text": content,
        "title": title,
        "url": page_url,
        "doc_type": doc_type
    }
    return uid, vector, payload

# Find document closely related to query
def query_index(query, top_k=5):
    vector = embed_text(query)
    results = client.search(
        collection_name=COLLECTION_NAME,
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

#
# Main - Index Blog Articles
#
tag_re = re.compile('<.*?>') # regex to remove html tags

# blog address - rss feed in json format
feed = "https://www.jasonacox.com/wordpress/feed/json"

# pull blog content
print(f"Pulling blog json feed content from {feed}...")
data = httpx.get(feed, timeout=None).json()

# First time - create index and import data
create_index()

# Loop to read in all articles - ignore any errors
print("Indexing blog articles...")  
n = 1
ids = []
vectors = []
payloads = []
for item in data["items"]:
    title = item["title"]
    url = item["url"]
    body = tag_re.sub('', item["content_html"])
    body = unescape(body)
    body = ''.join(char for char in body if char in string.printable)
    try:
        print(f"Adding: {n} : {title} [size={len(body)}]")
        uid, vector, payload = create_vector(body, title, url, doc_type="text")
        ids.append(uid)
        vectors.append(vector)
        payloads.append(payload)
    except:
        print(" - EMBEDDING ERROR: Ignoring")
    n = n + 1
    
## Add vectors to collection
print("Adding vectors to collection...")
client.upsert(
    collection_name=COLLECTION_NAME,
    points=qmodels.Batch(
        ids = ids,
        vectors=vectors,
        payloads=payloads
    ),
)

# Query the collection - TEST
prompt = "Give me some facts about solar."
query_result = query_index(prompt, top_k=RESULTS)

# Print results
print("")
print("Prompt: " + prompt)
print(f"Top {RESULTS} Documents found:")
for result in query_result:
    print(" * " + result['title'])

# Done