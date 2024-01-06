# Web Based Chatbot

This is a web based python flask app that allows you to chat with a LLM using the OpenAI API.

The intent of this project is to build and interact with a locally hosted LLM using consumer grade hardware. The examples below use a Llama 2 7B model served up with the OpenAI API compatible [llmserver](https://github.com/jasonacox/TinyLLM/tree/main/llmserver) on an Intel i5 systems with an Nvidia GeForce GTX 1060 GPU.

# Docker

```bash
# Build container
./build.sh

# Run container
./run.sh
```

# Manual

```bash
# Install required packages
pip install openai flask flask-socketio bs4

# Run the chatbot web server - change the base URL to be where you host your llmserver
OPENAI_API_BASE="http://localhost:8000/v1" python3 server.py
```

## Chat Commands and Retrieval Augmented Generation (RAG)

* Summarize External Site - If a URL is pasted in the prompt, the chatbot will read and summarize it.
* RAG - If a Qdrant host is specified, the chatbot will use the vector database information to respond.
* Command - There are information commands using `/`.

```
@library [opt:number] [prompt] # RAG - answer prompt based on response from qdrant library
#library [opt:number] [prompt] # RAG - import from qdrant library and summarize
/reset                         # Reset session
/version                       # Display chatbot version
/sessions                      # Display nmber of sessions
```

## Example Session

Open http://127.0.0.1:5000 - Example session:

<img width="946" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/08097e39-9c00-4f75-8c9a-d329c886b148">

## Read URL

If a URL is pasted in the text box, the chatbot will read and summarize it.

<img width="810" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/44d8a2f7-54c1-4b1c-8471-fdf13439be3b">

## Current News

If the `/news` command is used, the chatbot will attempt to fetch the latest news and have the LLM summarize it for you.

<img width="930" alt="image" src="https://github.com/jasonacox/TinyLLM/assets/836718/2732fe07-99ee-4795-a8ac-42d9a9712f6b">

## Alternative System Prompts

* A Hacker’s Guide to Language Models - Jeremy Howard [[link](https://www.youtube.com/watch?v=jkrNMKz9pWU&ab_channel=JeremyHoward)]

You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. If you think there might not be a correct answer, you say so. Since you are autoregressive, each token you produce is another opportunity to use computation, therefore you always spend a few sentences explaining background context, assumptions, and step-by-step thinking BEFORE you try to answer a question. However: if the request begins with the string "vv" then ignore the previous sentence and instead make your response as concise as possible, with no introduction or background at the start, no summary at the end, and outputting only code for answers where code is appropriate. Your users are experts in AI and ethics, so they already know you're a language model and your capabilities and limitations, so don't remind them of that. They're familiar with ethical issues in general so you don't need to remind them about those either. Don't be verbose in your answers, but do provide details and examples where it might help the explanation. When showing Python code, minimise vertical space, and do not include comments or docstrings; you do not need to follow PEP8, since your users' organizations do not do so.
