"""
Fetch blog data from jasonacox.com and embed into the chroma vector database. 
This uses the built-in sentence transformer for the embedding calculations.

Author: Jason A. Cox
10 October 2023
https://github.com/jasonacox/TinyLLM/

Requirements:
    * pip install sentence-transformers chromadb

"""
import os
import re
import string
from html import unescape
import uuid
import httpx
import chromadb
from chromadb.config import Settings

# ChromaDB Configuration Settings for Data
print("ChromaDB starting...")
DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR, 'data')
RESULTS = 5
chroma_client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(allow_reset=True, anonymized_telemetry=False))
sample_collection = chroma_client.get_or_create_collection(name="jasonacox")

# Initialize arrays
documents = []
metadatas = []
ids = []

# Read in blog data from jasonacox.com
tag_re = re.compile('<.*?>') # regex to remove html tags
feed = "https://www.jasonacox.com/wordpress/feed/json"
print(f"Pulling blog json feed content from {feed}...")
data = httpx.get(feed, timeout=None).json()

# Loop to read in all articles - ignore any errors
print("Indexing blog articles...")
n = 1
for item in data["items"]:
    uid = str(uuid.uuid1().int)[:32]
    title = item["title"]
    url = item["url"]
    meta = {'title': title, 'url': url}
    body = tag_re.sub('', item["content_html"])
    body = unescape(body)
    body = ''.join(char for char in body if char in string.printable)
    documents.append(body)
    metadatas.append(meta)
    ids.append(uid)
    n = n + 1

# Add vectors to collection
sample_collection.add(documents=documents, metadatas=metadatas, ids=ids)

# Query the collection - TEST
prompt = "Give me some facts about solar."
query_result = sample_collection.query(query_texts=prompt, n_results=RESULTS)

# Print results
print("")
print("Prompt: " + prompt)
print(f"Top {RESULTS} Documents found:")
for result in query_result['metadatas'][0]:
    print(" * " + result['title'])


