"""
ChatBot - LLM Functions

This module provides functions to interact with the OpenAI API and manage LLM models.
It includes functions to test the API connection, fetch available models, and send user prompts
to the LLM for streaming responses. It also handles context management and error recovery.
"""

# Imports
import time
import openai
from src.config import *
from src.prompts import *

# Test OpenAI API
def test_model():
    global API_KEY, API_BASE, MYMODEL, MAXTOKENS
    log("Testing OpenAI API...")
    try:
        log(f"Using openai library version {openai.__version__}")
        log(f"Connecting to OpenAI API at {API_BASE} using model {MYMODEL}")
        llm = openai.OpenAI(api_key=API_KEY, base_url=API_BASE)
        # Get models
        try:
            models = llm.models.list()
            if len(models.data) == 0:
                log("LLM: No models available - proceeding.")
        except Exception as erro:
            log(f"LLM: Unable to get models, using default: {str(erro)}")
            models = MYMODEL
        else:
            # build list of models
            model_list = [model.id for model in models.data]
            log(f"LLM: Models available: {model_list}")
            if not MYMODEL in model_list:
                log(f"LLM: Model {MYMODEL} not found in models list.")
                log("LLM: Switching to an available model: %s" % model_list[0])
                MYMODEL = model_list[0]
        # Test LLM
        log(f"LLM: Using model: {MYMODEL}")
        llm.chat.completions.create(
            model=MYMODEL,
            stream=False,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": "Hello"}],
            extra_body=EXTRA_BODY,
        )
        log("OpenAI API connection successful.")
        # Close the openai client
        llm.close()
        return True
    except Exception as erro:
        log("OpenAI API Error: %s" % erro)
        log(f"Unable to connect to OpenAI API at {API_BASE} using model {MYMODEL}.")
        return False

# Fetch list of LLM models
def get_models():
    global API_KEY, API_BASE
    try:
        llm = openai.OpenAI(api_key=API_KEY, base_url=API_BASE)
        models = llm.models.list()
        model_list = [model.id for model in models.data]
        llm.close()
        return model_list
    except Exception as erro:
        log(f"Unable to get models: {str(erro)}")
        return []
    
while True:
    if test_model():
        break
    else:
        log("Sleeping 5 seconds...")
        time.sleep(5)


# Function - Send user prompt to LLM for streaming response
async def ask(prompt, sid=None, sio=None):
    global client, stats, llm_stream
    stats["ask"] += 1
    response = False
    debug(f"Context size = {len(client[sid]['context'])}")
    while not response:
        try:
            # Remember context
            if ONESHOT:
                client[sid]["context"] = base_prompt()
            # Process image upload if present
            if client[sid]["image_data"]:
                # go through context and count images, remove if too many
                image_count = 0
                for turn in reversed(client[sid]["context"]):
                    if "content" in turn and isinstance(turn["content"], list):
                        for item in turn["content"]:
                            if "image_url" in item:
                                image_count += 1
                                if image_count >= MAX_IMAGES:
                                    # remove image from context
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
                llm_stream = openai.OpenAI(api_key=API_KEY, base_url=API_BASE)
            response = llm_stream.chat.completions.create(
                model=client[sid]["model"],
                stream=True, # Send response chunks as LLM computes next tokens
                temperature=TEMPERATURE,
                messages=client[sid]["context"],
                extra_body=EXTRA_BODY,
            )
            client[sid]["image_data"] = ""
        except openai.OpenAIError as erro:
            # If we get an error, try to recover
            client[sid]["context"].pop()
            if "does not exist" in str(erro):
                await sio.emit('update', {'update': '[Model Unavailable... Retrying]', 'voice': 'user'},room=sid)
                log("Model does not exist - retrying")
                test_model()
                # set client model to default
                client[sid]["model"] = MYMODEL
                # update footer
                await sio.emit('update', {'update': f"TinyLLM Chatbot {VERSION} - {client[sid]['model']} ",
                                          'voice': 'footer', 'model': client[sid]['model']},room=sid)
            elif "maximum context length" in str(erro):
                if len(prompt) > 1000:
                    # assume we have very large prompt - cut out the middle
                    prompt = prompt[:len(prompt)//4] + " ... " + prompt[-len(prompt)//4:]
                    log(f"Session {sid} - Reduce prompt size - Now: ~{len(prompt)/4} tokens") # tokens are ~4 bytes
                elif len(client[sid]["context"]) > 4:
                    # our context has grown too large, truncate the top
                    client[sid]["context"] = client[sid]["context"][:1] + client[sid]["context"][3:]
                    log(f"Session {sid} - Truncate context array: Now: {len(client[sid]['context'])} blocks")
                else:
                    # our context has grown too large, reset
                    client[sid]["context"] = base_prompt()
                    log(f"Session {sid} - Reset context to base prompt - Now: ~{len(client[sid]['context'])/4} tokens")
            elif "At most" in str(erro) and "image" in str(erro):
                # Remove oldest image from context
                for turn in reversed(client[sid]["context"]):
                    # if turn["content"] is a list, remove image_url
                    if "content" in turn and isinstance(turn["content"], list):
                        debug("Too many images - Found last image in context, removing...")
                        turn["content"] = ' '.join([x.get("text", "") for x in turn["content"]])
                        break
                continue
            elif "Internal Server Error" in str(erro):
                # Check to see if our context has images - if so, remove them
                debug("Internal Server Error - Checking for images in context...")
                removed_image_data = False
                for turn in client[sid]["context"]:
                    # if turn["content"] is a list, remove image_url
                    if "content" in turn and isinstance(turn["content"], list):
                        log("Found image in context, removing...")
                        removed_image_data = True
                        turn["content"] = ' '.join([x.get("text", "") for x in turn["content"]])
                if removed_image_data:
                    # remove last turn in context and retry
                    await sio.emit('update', {'update': '[Images do not seem to be supported by model... Removing]', 'voice': 'user'},room=sid)
                    client[sid]["context"].pop()
                    continue
                log(f"ERROR: {str(erro)}")
                stats["errors"] += 1
                await sio.emit('update', {'update': str(erro), 'voice': 'user'},room=sid)
                break
            else:
                # If all else fails, log the error and break
                log(f"ERROR: {str(erro)}")
                stats["errors"] += 1
                await sio.emit('update', {'update': str(erro), 'voice': 'user'},room=sid)
                break
    if not client[sid]["remember"]:
        client[sid]["remember"] =True
        client[sid]["context"].pop()
        client[sid]["context"].append({"role": "user", "content": "Help me remember."})
    return response

async def ask_llm(query,  model=MYMODEL):
    # Ask LLM a question
    global stats
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
    global stats
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
