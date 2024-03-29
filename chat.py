#!/usr/bin/python3
"""
Llama_cpp CLI ChatBot Example

Python chat client for OpenAI and the llama-cpp-python[server] OpenAI API Compatible 
Web Server. Provides a simple command line interface (CLI) chat session.

Features:
  * Uses OpenAI API
  * Works with local hosted OpenAI compatible llama-cpp-python[server]
  * Retains conversational context for LLM
  * Uses response stream to render LLM chunks instead of waiting for full response

Requirements:
  * pip install openai

Running a llama-cpp-python server:
  * CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
  * pip install llama-cpp-python[server]
  * python3 -m llama_cpp.server --model models/7B/ggml-model.bin

Author: Jason A. Cox
23 Sept 2023
https://github.com/jasonacox/TinyLLM

"""
import openai
import datetime

# Configuration Settings - Showing local LLM
api_key = "OPENAI_API_KEY"                       # Required, use bogus string for Llama.cpp
api_base = "http://localhost:8000/v1"            # Use API endpoint or comment out for OpenAI
agentname = "Jarvis"                             # Set the name of your bot
mymodel  ="tinyllm"                              # Pick model to use e.g. gpt-3.5-turbo for OpenAI
TESTMODE = False                                 # Uses test prompts
USE_SYSTEM = False                               # Use system prompt for first message

# Set base prompt and initialize the context array for conversation dialogue
current_date = datetime.datetime.now()
formatted_date = current_date.strftime("%m/%d/%Y")
baseprompt = "You are %s, a highly intelligent assistant. Keep your answers brief and accurate. Current date is %s." % (agentname, formatted_date)
if USE_SYSTEM:
    context = [{"role": "system", "content": baseprompt}] 
else:
    context = [{"role": "user", "content": baseprompt}, {"role": "assistant", "content": "Okay, let's get started."}] 

# Function - Send prompt to LLM for response
def ask(prompt):
    global context
    # remember context
    context.append({"role": "user", "content": prompt})
    llm = openai.OpenAI(api_key=api_key, base_url=api_base)
    response = llm.chat.completions.create(
        model=mymodel,
        max_tokens=1024,
        stream=True, # Send response chunks as LLM computes next tokens
        temperature=0.7,
        messages=context,
    )
    return response

# Function - Render LLM response output in chunks
def printresponse(response):
    completion_text = ''
    # iterate through the stream of events and print it
    for event in response:
        event_text = event.choices[0].delta.content
        if event_text:
            chunk = event_text
            completion_text += chunk
            print(f"{chunk}",end="",flush=True) 
    print("",flush=True)
    # remember context
    context.append({"role": "assistant", "content" : completion_text})
    return completion_text

# Chatbot Header
print(f"ChatBot - Greetings! My name is {agentname}. Enter an empty line to quit chat.")
print()

prompts = []
if TESTMODE:
    # define the series of questions here
    prompts.append("What is your name?")
    prompts.append("What is today's date?")
    prompts.append("What day of the week is it?")
    prompts.append("Answer this riddle: Ram's mom has three children, Reshma, Raja and a third one. What is the name of the third child?")
    prompts.append("Pick a color.")
    prompts.append("Now write a poem about that color.")
    prompts.append("Thank you very much! Goodbye!")

# Loop to prompt user for input
while True:
    if len(prompts) > 0:
        p = prompts.pop(0)
        print(f"> {p}")
    else:
        p = input("> ")
    if not p or p == "":
        break
    print()
    response=ask(p)
    print(f"{agentname}> ",end="", flush=True)
    ans = printresponse(response)
    print()

print("Done")