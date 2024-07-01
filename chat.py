#!/usr/bin/python3
"""
CLI ChatBot Example

Simple command line chat client for OpenAI compatible API supported LLMs
Including vLLM and llama-cpp-python[server].

Features:
    * Uses OpenAI Compatible API
    * Works with local hosted LLMs that support OpenAI API
    * Retains conversational context for LLM
    * Uses response stream to render LLM chunks instead of waiting for full response
    * Supports test mode with predefined prompts
    * Uses colorama for colored text
    * Auto selects first model if requested model is not available
    * Auto detects system prompt support and stop token requirements

Requirements:
    * pip install openai

Command Line Options:
    --api_base : API base URL (default: http://localhost:8000/v1)
    --api_key : API key 
    --model : Model to use
    --test : Run test prompts

Author: Jason A. Cox
23 Sept 2023
https://github.com/jasonacox/TinyLLM

"""
import openai
import datetime
import argparse
import colorama

version = "1.0.0"

# Set up colorama for colored text
colorama.init()

# Configuration Settings - Showing local LLM
api_key = "OPENAI_API_KEY"                       # Required, use bogus string for Llama.cpp
api_base = "http://localhost:8000/v1"            # Use API endpoint e.g. https://api.openai.com/v1
mymodel = "gpt-3.5-turbo"                        # Pick model to use e.g. gpt-3.5-turbo for OpenAI
agent_name = "TinyLLM"                           # Set the name of your bot
TEST_MODE = False                                # Use test prompts
COLOR_USER = colorama.Fore.WHITE                 # Color for user input
COLOR_BOT = colorama.Fore.LIGHTBLACK_EX          # Color for bot output
EXTRA = {"stop_token_ids":[128001, 128009]}      # Some models may require stop tokens (e.g. phi3)

# Connection to LLM, set the model and initialize the context
def connect():
    global mymodel, agent_name, context, EXTRA
    # Ask LLM for available models, try to match requested model otherwise use first model
    llm = openai.OpenAI(api_key=api_key, base_url=api_base)
    try:
        models = llm.models.list()
        if len(models.data) > 0:
            model_list = [model.id for model in models.data]
            if mymodel not in model_list:
                mymodel = model_list[0]
    except openai.BadRequestError:
        pass
    except Exception as e:
        if "400" in str(e):
            EXTRA = {} # Clear extra body for models that don't support it
        elif not ("404" in str(e)):
            print(f"Error: {e} - Unable to connect to {mymodel} on {api_base}.\n"
                "  Use: --api_base, --api_key and --model to set the connection.")
            exit(1)
    # Set the base prompt for the conversation, check if it supports a system prompt
    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime("%m/%d/%Y")
    base_prompt = f"You are {agent_name}, a highly intelligent assistant. Keep your answers brief and accurate. Current date is {formatted_date}."
    try:
        context = [{"role": "system", "content": base_prompt}] 
        _ = ask("Hello")
    except Exception as e:
        if "stop_token_ids" in str(e):
            EXTRA = {}
        if "user/assistant" in str(e):
            context = [{"role": "user", "content": base_prompt}, {"role": "assistant", "content": "Okay."}]

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
        extra_body=EXTRA,
    )
    return response

# Function - Render LLM response output in chunks
def print_response(response):
    completion_text = ''
    # iterate through the stream of events and print it
    for event in response:
        event_text = event.choices[0].delta.content
        if event_text:
            chunk = event_text
            completion_text += chunk
            # Add color to the bot response
            print(f"{COLOR_BOT}{chunk}",end="",flush=True)
    print("",flush=True)
    # remember context
    context.append({"role": "assistant", "content" : completion_text})
    return completion_text

# Create the argument parser
parser = argparse.ArgumentParser(description='CLI ChatBot')
parser.add_argument('--api_base', type=str, help='API base URL')
parser.add_argument('--api_key', type=str, help='API key')
parser.add_argument('--model', type=str, help='Model to use')
parser.add_argument('--agent_name', type=str, help='Name of the chatbot')
parser.add_argument('--test', action='store_true', help='Run test prompts')
args = parser.parse_args()

# Update configuration settings based on command line arguments
if args.api_base:
    api_base = args.api_base
if args.api_key:
    api_key = args.api_key
if args.model:
    mymodel = args.model
if args.agent_name:
    agent_name = args.agent_name
if args.test:
    TEST_MODE = True

# Initialize the base prompt and set the model
connect()

# Chatbot Header
print(f"ChatBot v{version} - Using {mymodel} on {api_base}\n")
print(f"{COLOR_BOT}{agent_name}> Greetings! My name is {agent_name}. Enter an empty line to quit chat.\n")

prompts = []
if TEST_MODE:
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
        print(f"{COLOR_USER}> {p}")
    else:
        try:
            p = input(f"{COLOR_USER}> ")
        except EOFError:
            break
    if not p or p == "":
        break
    print()
    response=ask(p)
    # Color the user input
    print(f"{COLOR_BOT}{agent_name}> ",end="", flush=True)
    ans = print_response(response)
    print()

print("Done")