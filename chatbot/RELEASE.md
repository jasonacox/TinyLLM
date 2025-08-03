# TinyLLM Chatbot Releases

## Container

The TinyLLM Chatbot docker container is available at: [jasonacox/chatbot](https://hub.docker.com/r/jasonacox/chatbot).

## 0.16.3 - OpenAI Image Generation Support

* Added OpenAI DALL-E image generation support with dedicated environment variables.
* Added separate configuration for OpenAI image generation: `OPENAI_IMAGE_API_KEY`, `OPENAI_IMAGE_API_BASE`, `OPENAI_IMAGE_MODEL`, and `OPENAI_IMAGE_SIZE`.
* Fixed "/image route not enabled" error by separating OpenAI image settings from LiteLLM proxy configuration.
* Improved image generation route handling with automatic format detection for different providers (SwarmUI/OpenAI).

## 0.16.2 - Repetition Filter Settings

* Added environment variables for repetition filter: `REPEAT_WINDOW` and `REPEAT_COUNT`.
* These control the window size and repeat count for the LLM output repetition filter. Users can now adjust these via environment variables to fine-tune anti-repetition behavior.
* Documentation: Updated Chatbot environment variable sections in the README to use consistent tables.
* Documentation: Improved clarity, formatting, and consistency in README and environment variable documentation.

## 0.16.1 - Image Generation

* Added environment variables for image generation settings: `IMAGE_MODEL`, `IMAGE_CFGSCALE`, `IMAGE_STEPS`, `IMAGE_SEED`, `IMAGE_TIMEOUT`, `IMAGE_WIDTH`, and `IMAGE_HEIGHT`.

## 0.16.0 - Refactor Chatbot

* Refactor - To improve maintainability and prepare for future expansions, the chatbot code has been modularized.
* Image Generation - The chatbot now uses the SwarmUI APIs to generate images. Users can request image generation using the explicit `/image {prompt}` command or indirectly by using the intent router. Intent router will allow for user dialogue to adjust the image via prompt updates.
* Upgrade note: This is a major version upgrade. You will need to reset your prompts for the image intent to work correctly.

## 0.15.25 - Improvements

* Chatbot - Improve intent routing for news.

## 0.15.24 - Intent Improvements

* Chatbot - Improve the conversation context for intent routing. This allows conversational intent following.

## 0.15.23 - Intent Hotfix

* Chatbot - Fix intent router to work with multi-modal (image) context. Updated internet search RAG to better handle conversation thread context answering.

## 0.15.22 - Intent Router

* Chatbot - The chatbot now has simple heuristics (based on LLM queries) to determine the users intent and to route to the built-in functions to ground the response. This includes internet search functions `/news`, `/stock`, `/weather` and `/search`.
* To activate set environmental variable `INTENT_ROUTER=true` or use the prompt command `/intent on`.

## 0.15.21 - Internet Search

* Chatbot - The `/search` command has been added to allow the chatbot to search the internet to help answer your prompt. By default, the first 5 most relevant internet search results are added to the context.

Usage: /search {opt:number} {query}

## 0.15.20 - Chatbot Model

* Model selection behavior improved. Session will persist with selected model in that session even when refreshed. Last model selection will be remembered and auto-selected for any new sessions but will not impact existing sessions.

## 0.15.19 - Chatbot Docs

* Update URL reader to display reading status to user while processing document.

## 0.15.18 - Chatbot Updates

* Model selection will now be stored as a cookie to allow it to persist between sessions.
* Image handling has been updated to recover when switching between vision models and language models. A new `MAX_IMAGES` setting has been added to allow persisting more than one image in the same conversation context (must be supported by model or the images will be pruned by chatbot)
* Model selection option `/model list` will display list of available models in chat windows.

## 0.15.17 - Model Selector

* Chatbot - The `/model` command will now initiate a UI popup window and dropdown to allow the user to select a model from the list of available models. Alternatively, the user can specify the model with the command (e.g. `/model mixtral`) to select it immediately without the popup.

## 0.15.16 - Think Tags

* Chatbot - Add `/think filter` command and `THINK_FILTER` environmental setting to have chatbot filter out (no display) the \<think>\</think> content from models that have built in CoT reasoning like Deepseek R1.

## 0.15.15 - Docker Compose

* Quick Start using Docker compose for Chatbot.
* Chatbot - Bug Fix: Remove token limit on response. The `MAXTOKENS` setting is used to prune content sent to LLM. If not set, no pruning will happen.
* Chatbot - Added additional LiteLLM support with the environmental settings `LITELLM_PROXY` and `LITELLM_KEY`. If set, these will override the OpenAI API settings to use LiteLLM and will remove `EXTRA_BODY` defaults that conflict with LiteLLM.

## 0.15.14 - Multi-model Support

* Chatbot - Add `/model` command to list available models and dynamically set models during the session.

## 0.15.13 - Resource Fix

* Chatbot - Add LLM connection closures for non-streaming ad-hoc calls (e.g. CoT calls). This has removed the resource warning. Improved debug messages.
* Chatbot Documentation - Updated CoT prompts and added reasoning.md for additional prompt options.

## 0.15.12 - Update CoT

* Chatbot - Update Chain of Thought (CoT) to check request before routing all prompts through the CoT process. Using `/think always` will force CoT for all requests. Additionally, CoT prompts updated for better responses.

## 0.15.11 - Chain of Thought

* Chatbot - Add Chain of Thought (CoT) thinking option using the `/think on` or `/think off` toggles to the UI. When activated, queries will be passed through an out-of-band CoT loop to allow the LLM to thoughtfully explore answer and then provide a conclusion summary to the user. Set environmental variable "THINKING" to "true" to default all conversations to CoT mode.

## 0.15.10 - Bug Fix

* Chatbot - Fix error handling bug used to auto-detect max content length of LLM. Updated user input UI rendering to better handle indentation.

## 0.15.9 - DocMan Auth

* DocMan - Add basic authentication and secure connection options to Weaviate.

## 0.15.8 - Enhance Image Processing

* Chatbot - Add support for HEIC file type and resize all images to max dimensions of 1024. Handle image pasting into input field. Remove previous images from context thread.
* Chatbot - Clean up logging: non-critical logs are moved to DEBUG level.

## 0.15.7 - Vision Model Support

* Chatbot - Allows user to drag and drop images into the context window for multi-modal vision LLMs.

## 0.15.6 - Progressive Loading

* Chatbot - Updated /rag commands to allow turning auto-RAG on and off, setting the collection and result number.

## 0.15.5 - Async and SocketIO

* Chatbot - Switch to async and socket communication for more responsive UI. Bug fixes.

## 0.15.2 - Weaviate Client Updates

* Chatbot and DocMan: Provide control for WEAVIATE_HOST and WEAVIATE_GRPC_HOST (and PORTs) settings separately via environmental variables.

## 0.15.1 - Document Manager Updates

* DocMan: Bug fixes and add features to process more document types (file or URL).

## 0.15.0 - Document Manager

* Chatbot: Using Document class for RAG functions.
* DocMan: New web based UI for managing documents in the Weaviate vector database. Allows user to upload and embed content from URLs and uploaded files. Provides optional chunking and management of embedded documents.

## 0.14.13 - TPS Calculation

* Chatbot: Fix a bug that was counting null tokens.

## 0.14.12 - Toxic Filter

* Chatbot: Add toxic filter option (uses environmental variable `TOXIC_THRESHOLD`) to analyze and filter out bad prompts. Uses LLM to evaluate and score prompt. Set variable between 0 and 1 or 99 to disable (default).
* Chatbot: Add `EXTRA_BODY` variable (JSON string) to customize chat completion calls.

## 0.14.11 - OpenAI Support

* Chatbot: Add logic to detect OpenAI URL and disable non-OpenAI stop_token_ids.

## 0.14.10 - Fix Popup

* Chatbot: Fix issue where DOM was being corrupted by popup. New logic creates separate div for conversation debug.

## 0.14.9 - Conversation Thread

* Chatbot: Add `Debug Session` link to footer to display conversation thread.

## 0.14.8 - RAG Updates

* Chatbot: Update some RAG to remove duplicate documents.

## 0.14.7 - TemplateResponse

* Update TemplateResponse arguments to current format.

## 0.14.6 - News Links

* Chatbot: Expand `/news/` RAG command to include reference URL links in news article headlines.
* Add response statistics (number of tokens and tokens per second) to footer.
* Serve up local copy of socket.io.js library to help with air-gap installations.

## 0.14.5 - Ollama Support

* Add logic to chatbot to support OpenAI API servers that do not support the `/v1/models` API. This allows the Chatbot to work with Ollama provided the user specifies the LLM_MODEL.

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
* Added `ONESHOT` environmental variable that if `True` will remove conversation threading allowing each query to be answered as a standalone session.
* Added `RAG_ONLY` environmental variable that if `True` will assume all queries should be directed to the default RAG database as set by `WEAVIATE_LIBRARY`.

## 0.12.6 - CUDA Support

* Add CUDA support for sentence transformers.
* Improve web page import function `extract_text_from_html()` for better RAG formatting.

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

* Ported Chatbot to the async FastAPI and Uvicorn ASGI high speed web server implementation.
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

Commands: /reset /version /sessions /rag /news /weather /stock

## 0.10.5 - vLLM Support

* vLLM provides a faster inference engine capable of handling multiple sessions simultaneously. It also runs well in Nvidia Docker containers.
* Chatbot: System prompts are not needed by vLLM as it does the translation based on the model being used. Using system prompts is now a configuration toggle in chatbot.

## 0.10.1 - Misc Updates

* Updated default prompts.
* Minor formatting updates

## 0.10.0 - Chat Prompt Settings

* Settings button allows user to update base and query prompts for the chatbot.

## 0.9.3 - Chat Format and News

* Chatbot: Added `/news` RAG command to chatbot which will cause it to attempt to fetch the latest news and have the LLM summarize it for you.

## 0.9.0 - Classifier

* Chatbot: Added `:` commands that will run a classifier on the prompt to determine RAG method to inform the LLM with current data to provide the response.

## 0.7.1 - Markdown Formatting

* Chatbot: Added "Copy code" button to code excerpts in LLM response.

## 0.7.0 - RAG Features

* Chatbot: Added `@` and `!` commands to pull prompt data documents from vector database for RAG responses.

## 0.1.0 - Initial Release
