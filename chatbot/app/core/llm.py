"""
ChatBot - LLM Functions

This module provides functions to interact with the OpenAI API and manage LLM models.
It includes functions to test the API connection, fetch available models, and send user prompts
to the LLM for streaming responses. It also handles context management and error recovery.

Author: Jason A. Cox
20 Apr 2025
github.com/jasonacox/TinyLLM
"""
# pylint: disable=unnecessary-pass
# pylint: disable=global-statement

# Imports
import asyncio
from typing import Optional, Any
import openai
from openai import OpenAIError
from app.core.config import (client, stats, API_KEY, API_BASE, MYMODEL, TEMPERATURE,
                             EXTRA_BODY, ONESHOT, MAX_IMAGES, VERSION,
                             log, debug, llm_stream)
from app.core.prompts import (base_prompt, expand_prompt, prompts)

class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass

class LLMConnectionError(LLMError):
    """Exception for connection-related errors"""
    pass

class LLMTimeoutError(LLMError):
    """Exception for timeout-related errors"""
    pass

# Test OpenAI API
async def test_model() -> bool:
    """Test the OpenAI API connection with retry logic"""
    global MYMODEL
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            log(f"Testing OpenAI API (attempt {attempt + 1}/{max_retries})...")
            log(f"Using openai library version {openai.__version__}")
            log(f"Connecting to OpenAI API at {API_BASE} using model {MYMODEL}")

            llm = openai.OpenAI(api_key=API_KEY, base_url=API_BASE, timeout=30)

            try:
                models = llm.models.list()
                if len(models.data) == 0:
                    log("LLM: No models available - proceeding.")
            except Exception as erro:
                log(f"LLM: Unable to get models, using default: {str(erro)}")
                models = MYMODEL
            else:
                model_list = [model.id for model in models.data]
                log(f"LLM: Models available: {model_list}")
                if not MYMODEL in model_list:
                    log(f"LLM: Model {MYMODEL} not found in models list.")
                    log("LLM: Switching to an available model: %s" % model_list[0])
                    MYMODEL = model_list[0]

            log(f"LLM: Using model: {MYMODEL}")
            llm.chat.completions.create(
                model=MYMODEL,
                stream=False,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": "Hello"}],
                extra_body=EXTRA_BODY,
            )
            log("OpenAI API connection successful.")
            llm.close()
            return True

        except OpenAIError as e:
            if "timeout" in str(e).lower():
                log(f"Timeout error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    log(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    raise LLMTimeoutError(f"Failed to connect after {max_retries} attempts: {str(e)}") from e
            else:
                raise LLMConnectionError(f"OpenAI API Error: {str(e)}") from e
        except Exception as e:
            log(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                log(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                raise LLMError(f"Failed after {max_retries} attempts: {str(e)}") from e

    return False

# Fetch list of LLM models
def get_models():
    try:
        llm = openai.OpenAI(api_key=API_KEY, base_url=API_BASE)
        models = llm.models.list()
        model_list = [model.id for model in models.data]
        llm.close()
        return model_list
    except Exception as erro:
        log(f"Unable to get models: {str(erro)}")
        return []

# Initialize LLM connection
async def init_llm():
    while True:
        if await test_model():
            break
        else:
            log("Sleeping 5 seconds...")
            await asyncio.sleep(5)

# Function - Send user prompt to LLM for streaming response
async def ask(prompt: str, sid: Optional[str] = None, sio: Optional[Any] = None) -> Any:
    """Send user prompt to LLM for streaming response with improved error handling"""
    global client, stats, llm_stream # pylint: disable=global-variable-not-assigned
    stats["ask"] += 1
    response = False
    max_retries = 3
    retry_delay = 5

    debug(f"Context size = {len(client[sid]['context'])}")

    for attempt in range(max_retries):
        try:
            if ONESHOT:
                client[sid]["context"] = base_prompt()

            if client[sid]["image_data"]:
                image_count = 0
                for turn in reversed(client[sid]["context"]):
                    if "content" in turn and isinstance(turn["content"], list):
                        for item in turn["content"]:
                            if "image_url" in item:
                                image_count += 1
                                if image_count >= MAX_IMAGES:
                                    debug("Too many images - Found image in context, removing 1...")
                                    turn["content"] = ' '.join([x.get("text", "") for x in turn["content"]])

                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{client[sid]['image_data']}"}}
                    ]
                }
                client[sid]["context"].append(message)
            else:
                client[sid]["context"].append({"role": "user", "content": prompt})

            debug(f"context -> LLM [{sid}] = {client[sid]['context']} - model = {MYMODEL}")

            if not llm_stream:
                llm_stream = openai.OpenAI(api_key=API_KEY, base_url=API_BASE, timeout=30)

            response = llm_stream.chat.completions.create(
                model=client[sid]["model"],
                stream=True,
                temperature=TEMPERATURE,
                messages=client[sid]["context"],
                extra_body=EXTRA_BODY,
            )
            client[sid]["image_data"] = ""
            break

        except OpenAIError as e:
            error_msg = str(e)
            if "does not exist" in error_msg:
                await sio.emit('update', {'update': '[Model Unavailable... Retrying]', 'voice': 'user'}, room=sid)
                log("Model does not exist - retrying")
                await test_model()
                client[sid]["model"] = MYMODEL
                await sio.emit('update', {'update': f"TinyLLM Chatbot {VERSION} - {client[sid]['model']} ",
                                      'voice': 'footer', 'model': client[sid]['model']}, room=sid)
            elif "maximum context length" in error_msg:
                if len(prompt) > 1000:
                    prompt = prompt[:len(prompt)//4] + " ... " + prompt[-len(prompt)//4:]
                    log(f"Session {sid} - Reduce prompt size - Now: ~{len(prompt)/4} tokens")
                elif len(client[sid]["context"]) > 4:
                    client[sid]["context"] = client[sid]["context"][:1] + client[sid]["context"][3:]
                    log(f"Session {sid} - Truncate context array: Now: {len(client[sid]['context'])} blocks")
                else:
                    client[sid]["context"] = base_prompt()
                    log(f"Session {sid} - Reset context to base prompt - Now: ~{len(client[sid]['context'])/4} tokens")
            elif "At most" in error_msg and "image" in error_msg:
                for turn in reversed(client[sid]["context"]):
                    if "content" in turn and isinstance(turn["content"], list):
                        debug("Too many images - Found last image in context, removing...")
                        turn["content"] = ' '.join([x.get("text", "") for x in turn["content"]])
                        break
                continue
            elif "Internal Server Error" in error_msg:
                debug("Internal Server Error - Checking for images in context...")
                removed_image_data = False
                for turn in client[sid]["context"]:
                    if "content" in turn and isinstance(turn["content"], list):
                        log("Found image in context, removing...")
                        removed_image_data = True
                        turn["content"] = ' '.join([x.get("text", "") for x in turn["content"]])
                if removed_image_data:
                    await sio.emit('update', {'update': '[Images do not seem to be supported by model... Removing]', 'voice': 'user'}, room=sid)
                    client[sid]["context"].pop()
                    continue
                log(f"ERROR: {error_msg}")
                stats["errors"] += 1
                await sio.emit('update', {'update': error_msg, 'voice': 'user'}, room=sid)
                break
            else:
                log(f"ERROR: {error_msg}")
                stats["errors"] += 1
                await sio.emit('update', {'update': error_msg, 'voice': 'user'}, room=sid)
                break

            if attempt < max_retries - 1:
                log(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                raise LLMError(f"Failed after {max_retries} attempts: {error_msg}") from e

    if not client[sid]["remember"]:
        client[sid]["remember"] = True
        client[sid]["context"].pop()
        client[sid]["context"].append({"role": "user", "content": "Help me remember."})

    return response

async def ask_llm(query,  model=MYMODEL):
    # Ask LLM a question
    global stats # pylint: disable=global-variable-not-assigned
    stats["ask_llm"] += 1
    content = base_prompt(expand_prompt(prompts["clarify"], values={})) + [{"role": "user",
                "content": query}]
    debug(f"ask_llm: {content}")
    llm = openai.AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
    response = await llm.chat.completions.create(
        model=model,
        stream=False,
        temperature=TEMPERATURE,
        messages=content,
        extra_body=EXTRA_BODY,
    )
    # close the openai client
    await llm.close()
    debug(f"ask_llm -> {response.choices[0].message.content.strip()}")
    return response.choices[0].message.content.strip()

async def ask_context(messages, model=MYMODEL):
    # Ask LLM a simple question
    global stats # pylint: disable=global-variable-not-assigned
    stats["ask_context"] += 1
    debug(f"ask_context: {messages}")
    llm = openai.AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
    response = await llm.chat.completions.create(
        model=model,
        stream=False,
        temperature=TEMPERATURE,
        messages=messages,
        extra_body=EXTRA_BODY,
    )
    # close the openai client
    await llm.close()
    debug(f"ask_context -> {response.choices[0].message.content.strip()}")
    return response.choices[0].message.content.strip()
