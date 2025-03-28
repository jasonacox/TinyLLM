# TinyLLM Web Based Chatbot and Document Manager

Chatbot: ![Chatbot](https://img.shields.io/docker/pulls/jasonacox/chatbot) DocMan: ![DocMan](https://img.shields.io/docker/pulls/jasonacox/docman)

The TinyLLM Chatbot is a web based python flask app that allows you to chat with a LLM using the OpenAI API.

The intent of this project is to build and interact with a locally hosted LLM using consumer grade hardware. With the Chatbot, we explore stitching context through conversational threads, rendering responses via realtime token streaming from LLM, and using external data to provide context for the LLM response (Retrieval Augmented Generation). With the Document Manager, we explore uploading documents to a Vector Database to use in retrieval augmented generation, allowing our Chatbot to produce answers grounded in knowledge that we provide.

Below are steps to get the Chatbot and Document Manager running.

## Chatbot

The Chatbot can be launched as a Docker container or via command line.

### Method 1: Docker Compose

A quickstart method is located in the [litellm](./litellm/) folder. This setup will launch the Chatbot + LiteLLM and PostgreSQL. This works on Mac and Linux (or WSL) systems.

```bash
cd litellm

# Edit compose.yaml and config.yaml for your setup.
nano compose.yaml
nano config.yaml

# Launch
docker compose up -d
```

The containers will download and launch. The database will be set up in the `./db` folder.
- The Chatbot will be available at http://localhost:5000
- The LiteLLM usage dashboard will be available at http://localhost:4000/ui

### Method 2: Docker

```bash
# Create placeholder prompts.json
touch prompts.json

# Run Chatbot - see run.sh for additional settings
docker run \
    -d \
    -p 5000:5000 \
    -e PORT=5000 \
    -e OPENAI_API_BASE="http://localhost:8000/v1" \
    -e INTENT_ROUTER=false \
    -e TZ="America/Los_Angeles" \
    -v $PWD/.tinyllm:/app/.tinyllm \
    --name chatbot \
    --restart unless-stopped \
    jasonacox/chatbot
```

#### LiteLLM Proxy Option

You can optionally set up LiteLLM to proxy multiple LLM backends (e.g. local vLLM, AWS Bedrock, OpenAI, Azure, Anthropic). See [LiteLLM documentation](https://docs.litellm.ai/docs/) for more information.

First, define your LLM connections in the local `config.yaml` file (see [LiteLLM options](https://docs.litellm.ai/docs/providers)). Note, if you are using a cloud provider service like AWS Bedrock or Azure, make sure you set up access first.

```yaml
model_list:

  - model_name: local-pixtral
    litellm_params:
      model: openai/mistralai/Pixtral-12B-2409
      api_base: http://localhost:8000/v1
      api_key: myAPIkey

  - model_name: bedrock-titan
    litellm_params:
      model: bedrock/amazon.titan-text-premier-v1:0
      aws_access_key_id: os.environ/CUSTOM_AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/CUSTOM_AWS_SECRET_ACCESS_KEY
      aws_region_name: os.environ/CUSTOM_AWS_REGION_NAME

  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY
```

Now,run the LiteLLM container. Edit this script to include your AWS and/or OpenAI keys for the models you want.

```bash
# Run LiteLLM Proxy - see
docker run \
    -d \
    -v $(pwd)/config.yaml:/app/config.yaml \
    -e CUSTOM_AWS_ACCESS_KEY_ID=your_AWS_key_here \
    -e CUSTOM_AWS_SECRET_ACCESS_KEY=your_AWS_secret_here \
    -e CUSTOM_AWS_REGION_NAME=us-east-1 \
    -e OPENAI_API_KEY=your_OpenAI_key_option \
    -p 4000:4000 \
    --name $CONTAINER \
    --restart unless-stopped \
    ghcr.io/berriai/litellm:main-latest \
    --config /app/config.yaml 
```

Finally, set up the chatbot to use LiteLLM:

```bash
# Run Chatbot - see run.sh for additional settings
docker run \
    -d \
    -p 5000:5000 \
    -e PORT=5000 \
    -e LITELLM_PROXY="http://localhost:4000/v1" \
    -e LITELLM_KEY="sk-mykey" \
    -e LLM_MODEL="local-pixtral" \
    -e TZ="America/Los_Angeles" \
    -v $PWD/.tinyllm:/app/.tinyllm \
    --name chatbot \
    --restart unless-stopped \
    jasonacox/chatbot
```

The Chatbot will try to use the specified model (`LLM_MODEL`) but if it is not available, it will select another available model. You can list and change the models inside the chatbot using the `/model` commands.

View the chatbot at http://localhost:5000

### Method 3: Command Line

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
/think filter [on|off]                  # Have chatbot filter out <think></think> content
/model [LLM_name]                       # Display or select LLM model to use (dialogue popup)
/search [opt:number] [prompt]           # Search the web to help answer the prompt
/intent [on|off]                        # Activate intent router to automatically run above functions
```

See the [rag](../rag/) for more details about RAG.

### Example Session

The examples below use a Llama 2 7B model served up with the OpenAI API compatible [llmserver](https://github.com/jasonacox/TinyLLM/tree/main/llmserver) on an Intel i5 systems with an Nvidia GeForce GTX 1060 GPU.

#### Chatbot

Open http://127.0.0.1:5000 - Example session:

<img width="800" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/08097e39-9c00-4f75-8c9a-d329c886b148">

#### Read URL

If a URL is pasted in the text box, the chatbot will read and summarize it.

<img width="800" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/44d8a2f7-54c1-4b1c-8471-fdf13439be3b">

#### Current News

The `/news` command will fetch the latest news and have the LLM summarize the top ten headlines. It will store the raw feed in the context prompt to allow follow-up questions.

<img width="800" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/2732fe07-99ee-4795-a8ac-42d9a9712f6b">

#### Model Selection

The `/model` command will popup the list of available models. Use the dropdown to select your model. Alternatively, specify the model with the command (e.g. `/model mixtral`) to select it immediately without the popup.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/e21ad350-6ae0-47de-b7ee-135176d66fe7" />

#### Search the Web

The `/search` command will allow the chatbot to search the web to help answer your prompt. This requires a [SearXNG](https://docs.searxng.org/) server which is started as part of the docker compose setup or can be run using:

```bash
echo "Starting $container container..."
docker run \
     -d \
     -p 8080:8080 \
     -v "${PWD}/litellm/searxng:/etc/searxng:rw" \
     -e "BASE_URL=http://localhost:8080/" \
     -e "INSTANCE_NAME=my-instance" \
     --name $container \
     --restart unless-stopped \
     searxng/searxng
```

The [settings.yml](./litellm/searxng/settings.yml) file needs to be edited to allow the json format. 

The chatbot looks for the environmental variable `SEARXNG` to set the URL of the search service, otherwise it uses http://localhost:8080. You can activate it by using the prompt command like this: `/search What is the cost of gas in Texas?`

<img width="800" alt="image" src="https://github.com/user-attachments/assets/8ee65216-6b11-4590-bf19-695e5b6e9a63" />

#### Intent Router

The chatbot now has the ability to read prompts and determine if a function call would help provide a grounded answer. 

It can be activated by setting the `INTENT_ROUTER=true` environmental variable or using the prompt command `/intent on`.  It will use things like /search to find current data on things that tend to change frequently (e.g. cost of eggs). It will also use /weather, /stock and /news. The heuristics it is using is based on LLM calls so the performance will vary based on the LLM you are using. It was tuned for use with the Llama-3.2-11B-Vision model.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/422a7c13-ec6f-43bb-a959-e37d0bb709ec" />

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

<img width="800" alt="image" src="https://github.com/user-attachments/assets/544c75d4-a1a3-4c32-a95f-7f12ff11a450">

<img width="800" alt="image" src="https://github.com/user-attachments/assets/4b15ef87-8f25-4d29-9214-801a326b406f">

The Chatbot can use this information if you send the prompt command:

```bash
# Usage: /rag {library} {opt:number} {prompt}

# Examples:
/rag records How much did we donate to charity in 2022?
/rag blog 5 List some facts about solar energy.
```
