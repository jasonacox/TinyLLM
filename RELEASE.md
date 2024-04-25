# Releases

## 0.14.4 - Llama-3 Support

* Add chatbot workaround for Meta Llama-3 support via stop token addition.
* Add logic to better handle model maximum context length errors with automated downsizing.
* Error handling and auto-retry for model changes on LLM.

## 0.14.3 - Resize Control

* Add intuitive UI control at top of user input area to allow user to resize text input box.

## 0.14.2 - Chatbot Stock RAG

* Add error checking and help for `/stock {company}` command.
* Allow user input textarea to be resized vertically.

## 0.14.1 - Chatbot Baseprompt

* Fixed bug with baseprompt updates to respond to saved Settings or new sessions.
* Updated baseprompt to include date and guidance for complex and open-ended questions.
* Add `TZ` local timezone environmental variable to ensure correct date in baseprompt.

## 0.14.0 - Chatbot Controls

* Added ability to change LLM Temperature and MaxTokens in settings.
* Added optional prompt settings read-only options to allow viewing but prevent changes (`PROMPT_RO=true`).

## 0.13.0 - Use Weaviate for RAG

* Moved from Qdrant to Weaviate - This externalizes the sentence transformation work and lets the chatbot run as a smaller service. Activate by setting `WEAVIATE_HOST` to the address of the DB.
* Added "References" text to output from `/rag` queries.
* Added `ONESHOT` environmental variable that if `True` will remove conversation threading allowing each query to be answered as a standalone sessions.
* Added `RAG_ONLY` environmental variable that if `True` will assume all queries should be directed to the default RAG database as set by `WEAVIATE_LIBRARY`.
* See https://github.com/jasonacox/TinyLLM/pull/5

```bash
docker run \
    -d \
    -p 5000:5000 \
    -e PORT=5000 \
    -e OPENAI_API_BASE="http://localhost:8000/v1" \
    -e ONESHOT="true" \
    -e RAG_ONLY="false" \
    -e WEAVIATE_HOST="localhost" \
    -e WEAVIATE_LIBRARY="tinyllm" \
    -v $PWD/.tinyllm:/app/.tinyllm \
    --name chatbot \
    --restart unless-stopped \
    jasonacox/chatbot
```

## 0.12.6 - CUDA Support

* Add CUDA support for sentence transformers.
* Improve web page import function `extract_text_from_html()` for better RAG formatting.
* Add RAG instructions for Weaviate Vector DB

## 0.12.5 - Chatbot LLM Model

* Added logic to poll LLM for model list. If only one model is available, use that. Otherwise verify the user requested model is available.
* Chatbot UI now shows model name and adds responsive elements to better display on mobile devices.

* Add encoding user prompts to correctly display html code in Chatbot.
* Fix `chat.py` CLI chatbot to handle user/assistant prompts for vLLM.

## 0.12.3 - Extract from URL

* Bug fix for `handle_url_prompt()` to extract text from URL.

## 0.12.2 - Misc Improvements 

* Speed up command functions using async, using `aiohttp`. 
* Fix prompt_expand for rag command.
* Added topic option to `/news` command.

## 0.12.1 - Performance Improvements

* Speed up user prompt echo. Immediately send to chat windows instead of waiting for LLM stream to start.
* Optimize message handling dispatching using async.
* Use AsyncOpenAI for non-streamed queries.

## 0.12.0 - FastAPI and Uvicorn

* Ported Chatbot to the async FastAPI and Uvicorn ASGI high speed web server implementation (https://github.com/jasonacox/TinyLLM/issues/3).
* Added /stats page to display configuration settings and current stats (optional `?format=json`)
* UI updated to help enforce focus on text entry box.
* Moved `prompts.json` and Sentence Transformer model location to a `./.tinyllm` for Docker support. 

## 0.11.4 - Stats Page

* Add `/stats` URL to Chatbot for settings and current status information.
* Update Chatbot HTML to set focus on user textbox.
* Move `prompts.json` and Sentence Transformer models into `.tinyllm` directory.

## 0.11.3 - Optimize for Docker

* Improve Chatbot for Docker
* Added admin alert broadcast feature (`POST /alert`)

## 0.11.0 - Chatbot Updates

* Add multi-line entry to prompt input using Shift-Enter.
* Fix HTML and CSS to support windows resize for settings dialogue box.
* Bug fix and Simplify RAG commands using slash prompts.

```
    Commands: /reset /version /sessions /rag /news /weather /stock
```

## 0.10.5 - vLLM Support

* vLLM provides a faster inference engine capable of handling multiple session simultaneously. It also runs well in Nvidia Docker containers. The llama-cpp-python implementation suffers from being single threaded and being fragile in containers (segment faults and core dumps). TODO: vLLM does not support older Nvidia cards by default. TODO: Provide instructions on modifying vLLM to run on Pascal based GPUs (e.g. Nvidia GTX 1060, Quadro P6000 or Tesla P100).
* Chatbot: System prompts are not needed by vLLM as it does the translation based on the model being used. Using system prompts is now a configuration toggle in chatbot.

## 0.10.1 - Misc Updates

* Updated default prompts.
* Minor formatting updates

## 0.10.0 - Chat Prompt Settings

* Settings button allows user to update base and query prompts for the chatbot.

## 0.9.3 - Chat Format and News

* LLMserver: Added chat format parameters to llama-cpp-python startup to ensure correct chat prompts are given to LLM based on model. See https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama_chat_format.py and consolidated list: https://github.com/jasonacox/TinyLLM/blob/main/llmserver/models/services/chatformats
* LLMserver: Updated [tinyllm](https://github.com/jasonacox/TinyLLM/blob/main/llmserver/tinyllm) startup script to include `restart` command.
* Chatbot: Added `/news` RAG command to chatbot which will cause it to attempt to fetch the latest news and have the LLM summarize it for you.
 
## 0.9.0 - Classifier
* Chatbot: Added `:` commands that will run a classifier on the prompt to determine RAG method to inform the LLM with current data to provide the response.

## 0.7.1 - Markdown Formatting
* Chatbot: Added "Copy code" button to code excerpts in LLM response.

## 0.7.0 - RAG Features
* Chatbot: Added `@` and `!` commands to pull prompt data documents from vector databse for RAG responses.

## 0.1.0 - Initial Release