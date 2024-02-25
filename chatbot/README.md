# TinyLLM Web Based Chatbot

The TinyLLM Chatbot is a web based python flask app that allows you to chat with a LLM using the OpenAI API.

The intent of this project is to build and interact with a locally hosted LLM using consumer grade hardware. With the Chatbot, we explore stitching context through conversational threads, rendering responses via realtime token streaming from LLM, and using external data to provide context for the LLM response (Retrieval Augemented Generation).

The Chatbot can be launched as a Docker container or via command line.

## Docker

```bash
# Create placeholder prompts.json
touch prompts.json

# Run Chatbot via Container - see run.sh for additional settings
docker run \
    -d \
    -p 5000:5000 \
    -e PORT=5000 \
    -e OPENAI_API_BASE="http://localhost:8000/v1" \
    -v $PWD/.tinyllm:/app/.tinyllm \
    --name chatbot \
    --restart unless-stopped \
    jasonacox/chatbot
```

## Command Line

```bash
# Install required packages
pip install fastapi uvicorn python-socketio jinja2 openai bs4 pypdf requests lxml aiohttp weaviate-client

# Run the chatbot web server - change the base URL to be where you host your llmserver
OPENAI_API_BASE="http://localhost:8000/v1" python3 server.py
```

## Chat Commands and Retrieval Augmented Generation (RAG)

Some RAG (Retrieval Augmented Generation) features including:

* Summarizing external websites and PDFs (paste a URL in chat window)
* If a Weaviate host is specified, the chatbot can use the vector database information to respond. See [rag](../rag/weaviate/) for details on how to set up Weaviate.
* Command - There are information commands using `/`

```
/reset                                  # Reset session
/version                                # Display chatbot version
/sessions                               # Display current sessions
/news                                   # List top 10 headlines from current new
/stock [company]                        # Display stock symbol and current price
/weather [location]                     # Provide current weather conditions
/rag [library] [opt:number] [prompt]    # Answer prompt based on response from Qdrant collection
```

See the [rag](../rag/) for more details about RAG.

## Example Session

The examples below use a Llama 2 7B model served up with the OpenAI API compatible [llmserver](https://github.com/jasonacox/TinyLLM/tree/main/llmserver) on an Intel i5 systems with an Nvidia GeForce GTX 1060 GPU.

### Chatbot

Open http://127.0.0.1:5000 - Example session:

<img width="946" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/08097e39-9c00-4f75-8c9a-d329c886b148">

### Read URL

If a URL is pasted in the text box, the chatbot will read and summarize it.

<img width="810" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/44d8a2f7-54c1-4b1c-8471-fdf13439be3b">

### Current News

The `/news` command will fetch the latest news and have the LLM summarize the top ten headlines. It will store the raw feed in the context prompt to allow follow-up questions.

<img width="930" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/2732fe07-99ee-4795-a8ac-42d9a9712f6b">

## Alternative System Prompts

* A Hacker’s Guide to Language Models - Jeremy Howard [[link](https://www.youtube.com/watch?v=jkrNMKz9pWU&ab_channel=JeremyHoward)]

You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. If you think there might not be a correct answer, you say so. Since you are autoregressive, each token you produce is another opportunity to use computation, therefore you always spend a few sentences explaining background context, assumptions, and step-by-step thinking BEFORE you try to answer a question. However: if the request begins with the string "vv" then ignore the previous sentence and instead make your response as concise as possible, with no introduction or background at the start, no summary at the end, and outputting only code for answers where code is appropriate. Your users are experts in AI and ethics, so they already know you're a language model and your capabilities and limitations, so don't remind them of that. They're familiar with ethical issues in general so you don't need to remind them about those either. Don't be verbose in your answers, but do provide details and examples where it might help the explanation. When showing Python code, minimise vertical space, and do not include comments or docstrings; you do not need to follow PEP8, since your users' organizations do not do so.
