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
docker run -p 6333:6333 \
    -d \
    --name qdrant \
    -v $PWD/storage:/qdrant/storage \
    -v $PWD/config/config.yaml:/qdrant/config/production.yaml \
    qdrant/qdrant

# Install python libraries
ip install qdrant-client sentence-transformers 
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

## TinyLLM Integration

External text files are processed, embedded and stored in the vector database. When a user query comes in for TinyLLM, it will first search the vector database and will present that as context in the prompt to the LLM. This will allow the LLM to have relevant local data to answer. The system prompt can guide the LLM to only answer based on the context or to use its extended model knowledge to fill in answer.

1. Embed text documents into vector database.
2. Build chatbot to augment user prompts with relevant context.
3. Call LLM for text generation.

```python
  context_str = ""
  
  for doc in documents:
    context_str += f"{doc.title}: {doc.content}\\n\\n"

  prompt = (
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
