# Releases

## 0.10.0 - Chat Prompt Settings

* Settings button allows user to update base and query fonts for the chatbot.

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