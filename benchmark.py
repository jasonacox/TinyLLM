#!/usr/bin/python3
"""
Benchmark Script for LLM on OpenAI API Server

This script is a simple benchmarking tool to test the performance of the LLM model on an
OpenAI API server. It sends a prompt to the server and measures the time it takes to
generate a response. It also measures the number of tokens generated and the tokens per second.

Requirements:
    * pip install openai 

31 Jan 2024
https://github.com/jasonacox/TinyLLM

"""
# Import Libraries
import time
from openai import OpenAI

# Update for your local settings
KEY = "RandomKey"           # OpenAI API Key if using OpenAI
MODEL = "tinyllm"           # Model name - tinyllm, gpt-3.5-turbo, etc.
API = "http://mc1:8000/v1"  # OpenAI https://api.openai.com/v1

# Connect to LLM
print(f"Connecting to LLM on {HOST}...")
print(f"Type 'exit' to quit")
client = OpenAI(api_key=KEY, base_url=API)
system = """
You are a highly intelligent assistant. Keep your answers brief and accurate.
assistant: I am here to help. I'm capable of many human-like tasks.
""".strip()
user = "Write an introduction a user will see when they first start your chatbot program"

# Generate completions and measure time
while not user in ["exit", ""]:
    stime = time.time()
    api_out = client.completions.create(prompt = system + "\n\nuser: " + user + "\nassistant:", 
        model=MODEL, stream=False, max_tokens=1024)
    ctime = round(time.time() - stime, ndigits=3)
    print(api_out.choices[0].text.strip())
    ctokens = int(api_out.usage.completion_tokens)
    tps = round(ctokens / ctime, ndigits=1)
    print(f"-- completion: time {ctime}s, {ctokens} tokens, {tps} tokens/s --")
    # Get next user input
    user = input("==>")
