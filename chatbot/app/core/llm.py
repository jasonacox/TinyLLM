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
from app.core.config import (client, stats, API_KEY, API_BASE, MYMODEL, TEMPERATURE, KEEP_ALIVE, OLLAMA_API,
                             EXTRA_BODY, ONESHOT, MAX_IMAGES, VERSION, INTENT_ROUTER_LLM,
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

# ── Resident models (Ollama) ──────────────────────────────────────────────────────────
# /v1/models says what EXISTS; Ollama's native /api/ps says what is LOADED right now. A model
# that is not resident costs a load on the next request (seconds to a minute for large models,
# and by default Ollama unloads after 5 idle minutes). Surfacing residency lets the picker
# prefer warm models, and warm_model() pins the chosen one with keep_alive.
_RESIDENT_CACHE = {"ts": 0.0, "models": None}

def ollama_api_base():
    """Native Ollama API base derived from OPENAI_API_BASE (…:11434/v1 -> …:11434), or OLLAMA_API."""
    if OLLAMA_API:
        return OLLAMA_API.rstrip("/")
    base = (API_BASE or "").rstrip("/")
    return base[:-3] if base.endswith("/v1") else base

def resident_models(ttl: float = 5.0):
    """Set of model names currently loaded on the Ollama server, or None if the backend is not Ollama
    (or unreachable). Cached for `ttl` seconds so the model dialog never hammers the server."""
    import time as _time
    now = _time.time()
    if _RESIDENT_CACHE["models"] is not None and now - _RESIDENT_CACHE["ts"] < ttl:
        return _RESIDENT_CACHE["models"]
    try:
        import requests as _rq
        r = _rq.get(f"{ollama_api_base()}/api/ps", timeout=2)
        if r.status_code != 200:
            _RESIDENT_CACHE.update(ts=now, models=None); return None
        loaded = {m.get("name") or m.get("model") for m in r.json().get("models", [])}
        loaded |= {n.split(":")[0] for n in list(loaded) if n and n.endswith(":latest")}   # "llama3:latest" answers to "llama3"
        _RESIDENT_CACHE.update(ts=now, models=loaded)
        return loaded
    except Exception as erro:  # not Ollama, or down — degrade to "unknown"
        debug(f"resident_models: {erro}")
        _RESIDENT_CACHE.update(ts=now, models=None)
        return None

def get_models_detail():
    """[{id, resident}] — resident is True/False when the backend is Ollama, None when unknown.
    Resident models sort first so the dialog and /model list lead with what answers instantly."""
    loaded = resident_models()
    def _is_resident(name):
        if loaded is None:
            return None
        return name in loaded or name.split(":")[0] in loaded or f"{name}:latest" in loaded
    detail = [{"id": m, "resident": _is_resident(m)} for m in get_models()]
    detail.sort(key=lambda d: (d["resident"] is not True, d["id"]))
    return detail

def warm_model(model: str, keep_alive=None):
    """Load `model` on an Ollama server and pin it with keep_alive (env KEEP_ALIVE unless given).
    Returns True if the server accepted the request. No-op for non-Ollama backends."""
    ka = KEEP_ALIVE if keep_alive is None else keep_alive
    if ka is None or not model:
        return False
    try:
        import requests as _rq
        ka_val = int(ka) if str(ka).lstrip("-").isdigit() else ka            # -1 / 0 as ints, "30m" as a string
        r = _rq.post(f"{ollama_api_base()}/api/generate", json={"model": model, "prompt": "", "keep_alive": ka_val}, timeout=600)
        ok = r.status_code == 200
        if ok:
            _RESIDENT_CACHE["ts"] = 0.0                                        # invalidate: it is resident now
            log(f"Pinned model {model} resident (keep_alive={ka})")
        else:
            log(f"warm_model({model}) -> HTTP {r.status_code}")
        return ok
    except Exception as erro:
        log(f"warm_model({model}) failed: {erro}")
        return False

_WARM_TASKS = set()                                                            # strong refs so fire-and-forget tasks aren't GC'd

def warm_model_background(model: str, keep_alive=None):
    """Fire-and-forget warm_model() on a worker thread. Call from async handlers: a model switch
    returns immediately and never waits on (or blocks the event loop for) a cold load.
    Returns the task, or None when there is nothing to do (KEEP_ALIVE unset)."""
    if (KEEP_ALIVE if keep_alive is None else keep_alive) is None or not model:
        return None
    task = asyncio.create_task(asyncio.to_thread(warm_model, model, keep_alive))
    _WARM_TASKS.add(task)
    task.add_done_callback(_WARM_TASKS.discard)
    return task

# Initialize LLM connection
async def init_llm():
    while True:
        if await test_model():
            break
        else:
            log("Sleeping 5 seconds...")
            await asyncio.sleep(5)
    if KEEP_ALIVE:
        await asyncio.to_thread(warm_model, MYMODEL)                           # off the event loop

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

async def ask_llm(query, model=MYMODEL, use_intent_model=False):
    # Ask LLM a question
    # If use_intent_model is True and INTENT_ROUTER_LLM is set, use that model instead
    global stats # pylint: disable=global-variable-not-assigned
    stats["ask_llm"] += 1
    
    # Determine which model to use
    selected_model = model
    if use_intent_model and INTENT_ROUTER_LLM:
        selected_model = INTENT_ROUTER_LLM
    
    content = base_prompt(expand_prompt(prompts["clarify"], values={})) + [{"role": "user",
                "content": query}]
    debug(f"ask_llm{'_intent' if use_intent_model else ''} (model: {selected_model}): {content}")
    llm = openai.AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
    response = await llm.chat.completions.create(
        model=selected_model,
        stream=False,
        temperature=TEMPERATURE,
        messages=content,
        extra_body=EXTRA_BODY,
    )
    # close the openai client
    await llm.close()
    debug(f"ask_llm{'_intent' if use_intent_model else ''} -> {response.choices[0].message.content.strip()}")
    return response.choices[0].message.content.strip()

async def ask_llm_intent(query, model=MYMODEL):
    # Ask LLM a question for intent routing - uses INTENT_ROUTER_LLM if set, otherwise falls back to provided model
    # This is a convenience wrapper around ask_llm with use_intent_model=True
    return await ask_llm(query, model=model, use_intent_model=True)

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
