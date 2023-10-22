#!/usr/bin/python3
"""
Search qdrant vector database. 
This uses a sentence transformer for the embedding calculations.

Author: Jason A. Cox
11 October 2023
https://github.com/jasonacox/TinyLLM/

Requirements:
    * pip install qdrant-client sentence-transformers pydantic~=2.4.2
"""
import os
from html import unescape
import qdrant_client as qc
import qdrant_client.http.models as qmodels
from sentence_transformers import SentenceTransformer

# Configuration Settings
MODEL = os.environ.get("MY_MODEL", "all-MiniLM-L6-v2")
DEBUG = os.environ.get("DEBUG", "False") == "True"
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "mylibrary") 
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
DEVICE = os.environ.get("DEVICE", "cuda")
RESULTS = 5

# Sentence Transformer Setup
print("Sentence Transformer starting...")
model = SentenceTransformer(MODEL, device=DEVICE) 

# Qdrant Setup
print("Connecting to Qdrant DB...")
client = qc.QdrantClient(url=QDRANT_HOST)

# Create embeddings for text
def embed_text(text):
    embeddings = model.encode(text, convert_to_tensor=True)
    return embeddings

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
# Main - User Loop
#

# Query the collection - TEST
print("Connected to Vector Database\n")
print("Enter a query or 'q' to quit.")
while True:
    prompt = input("\nQuery: ")
    if prompt in ["q", "Q", ""]:
        break

    # Query the index
    query_result = query_index(prompt, top_k=RESULTS)

    # Print results
    print("")
    print("Prompt: " + prompt)
    print(f"Top {RESULTS} Documents found:")
    for result in query_result:
        print(" * " + result['title'])

print("")
print("Done.")
# Done