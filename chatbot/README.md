# TinyLLM Web Based Chatbot and Document Manager

Chatbot: ![Chatbot](https://img.shields.io/docker/pulls/jasonacox/chatbot) DocMan: ![DocMan](https://img.shields.io/docker/pulls/jasonacox/docman)

The TinyLLM Chatbot is a web based python flask app that allows you to chat with a LLM using the OpenAI API.

The intent of this project is to build and interact with a locally hosted LLM using consumer grade hardware. With the Chatbot, we explore stitching context through conversational threads, rendering responses via realtime token streaming from LLM, and using external data to provide context for the LLM response (Retrieval Augmented Generation). With the Document Manager, we explore uploading documents to a Vector Database to use in retrieval augmented generation, allowing our Chatbot to produce answers grounded in knowledge that we provide.

Below are steps to get the Chatbot and Document Manager running.

## Chatbot

The Chatbot can be launched as a Docker container or via command line.

### Docker

```bash
# Create placeholder prompts.json
touch prompts.json

# Run Chatbot via Container - see run.sh for additional settings
docker run \
    -d \
    -p 5000:5000 \
    -e PORT=5000 \
    -e OPENAI_API_BASE="http://localhost:8000/v1" \
    -e TZ="America/Los_Angeles" \
    -v $PWD/.tinyllm:/app/.tinyllm \
    --name chatbot \
    --restart unless-stopped \
    jasonacox/chatbot
```

### Command Line

```bash
# Install required packages
pip install -r requirements.txt

# Run the chatbot web server - change the base URL to be where you host your llmserver
OPENAI_API_BASE="http://localhost:8000/v1" python3 server.py
```

### Chat Commands and Retrieval Augmented Generation (RAG)

Some RAG (Retrieval Augmented Generation) features including:

* Summarizing external websites and PDFs (paste a URL in chat window)
* If a Weaviate host is specified, the chatbot can use the vector database information to respond. See [rag](../rag/weaviate/) for details on how to set up Weaviate.
* Perform chain of thought (CoT) reasoning with `/think on` command (see [reasoning](./reasoning.md) for more details).
* Command - There are information commands using `/`

```
/reset                                  # Reset session
/version                                # Display chatbot version
/sessions                               # Display current sessions
/news                                   # List top 10 headlines from current new
/stock [company]                        # Display stock symbol and current price
/weather [location]                     # Provide current weather conditions
/rag on [library] [opt:number]          # Route all prompts through RAG using specified library
/rag off                                #   Disable
/think on                               # Perform Chain of Thought thinking on relevant prompts
/think off                              #   Disable
```

See the [rag](../rag/) for more details about RAG.

### Example Session

The examples below use a Llama 2 7B model served up with the OpenAI API compatible [llmserver](https://github.com/jasonacox/TinyLLM/tree/main/llmserver) on an Intel i5 systems with an Nvidia GeForce GTX 1060 GPU.

#### Chatbot

Open http://127.0.0.1:5000 - Example session:

<img width="946" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/08097e39-9c00-4f75-8c9a-d329c886b148">

#### Read URL

If a URL is pasted in the text box, the chatbot will read and summarize it.

<img width="810" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/44d8a2f7-54c1-4b1c-8471-fdf13439be3b">

#### Current News

The `/news` command will fetch the latest news and have the LLM summarize the top ten headlines. It will store the raw feed in the context prompt to allow follow-up questions.

<img width="930" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/2732fe07-99ee-4795-a8ac-42d9a9712f6b">


## Document Manager (Weaviate)

The document manager allows you to manage the collections and documents in the Weaviate vector database. It provides an easy way for you to upload and ingest the content from files or URL. It performs simple chunking (if requested). The simple UI let's you navigate through the collections and documents.

### Environment Variables

- MAX_CHUNK_SIZE: Maximum size of a chunk in bytes (default 1024)
- UPLOAD_FOLDER: Folder where uploaded files are stored (default uploads)
- HOST: Weaviate host (default localhost)
- COLLECTIONS: Comma separated list of collections allowed (default all)
- PORT: Port for the web server (default 8000)
- COLLECTIONS_ADMIN: Allow users to create and delete collections (default True)

### Docker Setup

The Document Manager uses a vector database to store the uploaded content. Set up the Weaviate vector database using this docker compose and the included [docker-compose.yml](docker-compose.yml) file.

```bash
# Setup and run Weaviate vector database on port 8080

docker compose up -d
```

To run the Document Manager, run the following and adjust as needed. Once running, the document manager will be available at http://localhost:5001

```bash
docker run \
    -d \
    -p 5001:5001 \
    -e PORT="5001" \
    -e WEAVIATE_HOST="localhost" \
    -e WEAVIATE_GRPC_HOST="localhost" \
    -e WEAVIATE_PORT="8080" \
    -e WEAVIATE_GRPC_PORT="50051" \
    -e MAX_CHUNK_SIZE="1024" \
    -e UPLOAD_FOLDER="uploads" \
    -e COLLECTIONS_ADMIN="true" \
    --name docman \
    --restart unless-stopped \
    jasonacox/docman
```
Note - You can restrict collections by providing the environmental variable `COLLECTIONS` to a string of comma separated collection names.

### Usage

You can now create collections (libraries of content) and upload files and URLs to be stored into the vector database for the Chatbot to reference.

<img width="1035" alt="image" src="https://github.com/user-attachments/assets/544c75d4-a1a3-4c32-a95f-7f12ff11a450">

<img width="1035" alt="image" src="https://github.com/user-attachments/assets/4b15ef87-8f25-4d29-9214-801a326b406f">

The Chatbot can use this information if you send the prompt command:

```bash
# Usage: /rag {library} {opt:number} {prompt}

# Examples:
/rag records How much did we donate to charity in 2022?
/rag blog 5 List some facts about solar energy.
```
