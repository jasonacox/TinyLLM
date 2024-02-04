# Releases

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