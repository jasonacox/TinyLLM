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
KEY = "RandomKey"                 # OpenAI API Key if using OpenAI
MODEL = "tinyllm"                 # Model name - tinyllm, gpt-3.5-turbo, etc.
API = "http://localhost:8000/v1"  # OpenAI https://api.openai.com/v1

# Prompts used to test LLM performance
PROMPTS = [
    "What is the meaning of life?",
    "What is the best programming language for beginners?",
    "List the top 10 programming languages.",
    "If I want to build a web application, how do I get started?",
    "What is 1 + 1?",
    "What is the best way to learn programming?",
    "Describe the solar system.",
    "How can I learn to play the guitar?",
    "A girl is walking in the park and sees a dog. What happens next?",
    "An alien lands on Earth and meets a human. There is a conversation. What do they talk about?",
    "Ramses II was a pharaoh of Egypt. What did he do?",
    "Solve this riddle: Ram's mom has three kids. Reshma, Raja, and what's the third kid's name? Let's think step by step and show your logic.",
    "What is the capital of France?",   
    "Tell me a joke?",
    "What is the best way to learn a new language?",
    "How many people live in New York City?",
    "List the top 10 cities in the world and their populations.",
    "When I was 6 my sister was half my age. Now I’m 70 how old is my sister? Work this out step by step and show your logic.",
    "The elephant is the largest land animal. What is the largest animal in the world?",
    "What is the most precious metal?",
    "What is the most important invention in human history?",
]

# Connect to LLM
print(f"Connecting to LLM on {API} and testing {len(PROMPTS)} prompts...")
print()
client = OpenAI(api_key=KEY, base_url=API)
system = """
You are a highly intelligent assistant. Keep your answers brief and accurate.
assistant: I am here to help. I'm capable of many human-like tasks.
""".strip()

# Remember min and max
min_tps = 1000000
max_tps = 0

# Generate completions and measure time
for user in PROMPTS:
    stime = time.time()
    api_out = client.completions.create(prompt = system + "\n\nuser: " + user + "\nassistant:", 
        model=MODEL, stream=False, max_tokens=1024, temperature=0.0)
    ctime = round(time.time() - stime, ndigits=3)
    print(f"Prompt: {user}")
    print(f"Response: {api_out.choices[0].text.strip()}")
    ctokens = int(api_out.usage.completion_tokens)
    tps = round(ctokens / ctime, ndigits=1)
    print(f"-- completion: time {ctime}s, {ctokens} tokens, {tps} tokens/s --")
    print()
    if tps < min_tps:
        min_tps = tps
    if tps > max_tps:
        max_tps = tps

# Report performance
print()
print(f"Completed {len(PROMPTS)} prompts.")
print(f"Min TPS: {min_tps}, Max TPS: {max_tps}, Avg TPS: {round((min_tps + max_tps) / 2, ndigits=1)}")
