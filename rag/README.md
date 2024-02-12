# Retrieval Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is an architecture that combines the strengths of retrieval-based and generation-based language models. The basic idea is to use a retrieval model to generate high-quality text, and then augment this text with additional information from a generative model. This allows RAG models to generate more accurate and informative text than either retrieval or generation models alone.

This will explore the following vector databases:

* [Qdrant](https://qdrant.tech/) (pronounced quadrant) - vector database & vector similarity search engine
* [Chroma](https://www.trychroma.com/) - the AI-native open-source embedding database

These are both an open-source vector database systems that store and retrieve vector embeddings. They are designed to store embeddings with associated metadata for use by large language models. Additional metadata (e.g. url, title, doc_type) can be stored along with the embedded payload.

## Qdrant

Qdrant server can be started by using the [setup.sh](setup.sh) script.

```bash
# Start qdrant container
cd qdrant
docker run -p 6333:6333 \
    -d \
    --name qdrant \
    -v $PWD/storage:/qdrant/storage \
    -v $PWD/config/config.yaml:/qdrant/config/production.yaml \
    qdrant/qdrant

# Install python libraries
pip install qdrant-client sentence-transformers 
```

The example script `qdrant.py` demonstrates how to embed text documents into a vector database. This example uses blog posts, embeds the articles using a sentence transformer and stores them in the qdrant database.

Example Run

<img width="646" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/db92f2b0-033c-4743-84d0-f7a70d629348">

```txt
Prompt: Give me some facts about solar.
Top 5 Documents found:
 * California Solar and Net Metering
 * Halfway Out of the Dark
 * Richard Feynman on Light
 * Solar Powered WiFi Weather Station
 * Make a Difference
```

## Chroma

```bash
# Install python libraries
pip install sentence-transformers chromadb
```

The example script `chroma.py` demonstrates how to embed text documents into a vector database. This example uses blog posts, embeds the articles using a sentence transformer and stores them in the qdrant database.

## TinyLLM Chatbot Integration

External text files need to be processed, embedded and stored in the vector database. An example script on how to do that is in [qdrant-single.py](./qdrant-single.py), with a snip below:

```python
# Create embeddings for text
def embed_text(text):
    embeddings = model.encode(text, convert_to_tensor=True)
    return embeddings

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

# Adds document vector to qdrant database
def add_doc_to_index(text, title, url, doc_type="text"):
    ids = []
    vectors = []
    payloads = []
    uid, vector, payload = create_vector(text,title,url, doc_type)
    ids.append(uid)
    vectors.append(vector)
    payloads.append(payload)
    ## Add vectors to collection
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=qmodels.Batch(
            ids = ids,
            vectors=vectors,
            payloads=payloads
        ),
    )
```

Once those documents are embedded and stored in the Qdrant vector database, the TinyLLM Chatbot can be set up to use that for `/rag <library> <prompt>` command responses.

### TinyLLM Chatbot with Qdrant Support

Run `jasonacox/chatbot:latest-rag` which includes the Sentance-Transformer and Qdrant client code.

```bash
# Create placeholder prompts.json
touch prompts.json

# Start the chatbot container
docker run \
    -d \
    -p 5000:5000 \
    -e PORT=5000 \
    -e OPENAI_API_BASE="http://localhost:8000/v1" \
    -e LLM_MODEL="tinyllm" \
    -e USE_SYSTEM="false" \
    -e QDRANT_HOST="localhost" \
    -e RESULTS=1 \
    -e SENTENCE_TRANSFORMERS_HOME=/app/.tinyllm \
    -v $PWD/.tinyllm:/app/.tinyllm \
    --name chatbot \
    --restart unless-stopped \
    jasonacox/chatbot:latest-rag
```

When a user runs a command like, `/rag mylibrary List some facts about solar`, the Chatbot will first search the vector database for semantically similar documents and will present the top `RESULTS` of those as context for the prompt to the LLM. This will allow the LLM to have relevant local data to answer. The RAG prompt can guide the LLM to only answer based on the context or to use its extended model knowledge to fill in answer. By default it will use the following RAG prompt to structure the response:


```
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Back up your answer using bullet points and facts from the context.
Context: {context_str}
Question: {prompt}
Answer:
```

```python
  context_str = ""
  
  for doc in documents:
    context_str += f"{doc.title}: {doc.content}\\n\\n"

  # Prompt - This is a good one for most LLMs
  prompt = (
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
    f"Question: {prompt}"
    f"Context: {context_str}"
    "Answer:"
  )

  # Prompt alternative
  prompt2 = (
    "Context: \\n"
    "---------------------\\n"
    f"{context_str}"
    "\\n---------------------\\n"
    f"Given the above context and no other information, answer the question: {question}\\n"
  )

  data = {"prompt": prompt}
  res = requests.post(f"{base_url}:8080/v1/models/model:predict", json=data)
  res_json = res.json()
  return res_json['data']['generated_text']
  ```

## References

* Jacob Marks - How I Turned My Company’s Docs into a Searchable Database with OpenAI - https://towardsdatascience.com/how-i-turned-my-companys-docs-into-a-searchable-database-with-openai-4f2d34bd8736
 * Jason Fan - How to connect Llama 2 to your own data, privately - https://jfan001.medium.com/how-to-connect-llama-2-to-your-own-data-privately-3e14a73e82a2
 * Sentence Transformers - https://www.sbert.net/index.html
