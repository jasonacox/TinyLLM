# Retrieval Augmented Generation (RAG)

** This is a work in progress **

Retrieval-Augmented Generation (RAG) is an architecture that combines the strengths of retrieval-based and generation-based language models. The basic idea is to use a retrieval model to generate high-quality text, and then augment this text with additional information from a generative model. This allows RAG models to generate more accurate and informative text than either retrieval or generation models alone.

For TinyLLM we are going to use Qdrant (pronounced quadrant) as a vector databases to store our local corpus or data that we want to search and use for our LLM.

## Background

External text files are processed, embedded and stored in the Qdrant vector database. When a user query comes in for TinyLLM, it will first search the vector database and will present that as context in the prompt to the LLM. This will allow the LLM to have relevant local data to answer. The system prompt can guide the LLM to only answer based on the context or to use its extended model knowledge to fill in answer.

## Steps

1. Embed text documents into Qdrant vector database
2. Build chatbot to augment user prompts with relevant context.
3. Call LLM for text generation.

## Setup

The following assumptions:
* llmserver is already running on port 

### Qdrant

Qdrant server can be started by using the [setup.sh](setup.sh) script.

```bash
docker run -p 6333:6333 \
    -d \
    --name qdrant \
    -v $PWD/storage:/qdrant/storage \
    -v $PWD/config/config.yaml:/qdrant/config/production.yaml \
    qdrant/qdrant
```

### Import Documents

See the `import.py` script...

### Query LLM with Local Documents Context

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
