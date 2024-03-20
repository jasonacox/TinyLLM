#!/usr/bin/python3
"""
News Bot - Send news alerts to clients

Author: Jason A. Cox
17 Mar 2024
https://github.com/jasonacox/TinyLLM

"""
# Import Libraries
import datetime
import os
import time

import openai
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# Version
VERSION = "v0.0.1"
DEBUG = False

def log(text):
    # Print to console
    if DEBUG:
        print(f"INFO: {text}")

def error(text):
    # Print to console
    print(f"ERROR: {text}")

# Configuration Settings
api_key = os.environ.get("OPENAI_API_KEY", "open_api_key")                  # Required, use bogus string for Llama.cpp
api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")    # Required, use https://api.openai.com for OpenAI
agentname = os.environ.get("AGENT_NAME", "Newsbot")                         # Set the name of your bot
mymodel = os.environ.get("LLM_MODEL", "models/7B/gguf-model.bin")           # Pick model to use e.g. gpt-3.5-turbo for OpenAI
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"                  # Set to True to enable debug mode
MAXTOKENS = int(os.environ.get("MAXTOKENS", 16*1024))                       # Maximum number of tokens to send to LLM
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))                     # LLM temperature
USE_SYSTEM = os.environ.get("USE_SYSTEM", "false").lower == "true"          # Use system in chat prompt if True
RESULTS = int(os.environ.get("RESULTS", 10))                                # Number of results to return from Weaviate
ALPHA_KEY = os.environ.get("ALPHA_KEY", "alpha-key")                        # Alpha Vantage API Key

# Prompt Defaults
prompts = {
    "greeting": "Hi",
    "agentname": "Jarvis",
    "baseprompt": "You are {agentname}, a highly intelligent assistant. The current date is {date}.\n\nYou should give concise responses to very simple questions, but provide thorough responses to more complex and open-ended questions.",
    "weather": "You are a weather forecaster. Keep your answers brief and accurate. Current date is {date} and weather conditions:\n[DATA]{context_str}[/DATA]\nProvide a weather update, current weather alerts, conditions, precipitation and forecast for {location} and answer this: {prompt}.",
    "stock": "You are a stock analyst. Keep your answers brief and accurate. Current date is {date}.",
    "news": "You are a newscaster who specializes in providing headline news. Use only the following context provided by Google News to summarize the top 10 headlines for today. Do not display the pub date or timestamp. Rank headlines by most important to least important. Always include the news organization. Do not add any commentary.\nAlways use this format:\n#. [News Item] - [News Source]\nHere are some examples: \n1. The World is Round - Science\n2. The Election is over and Children have won - US News\n3. Storms Hit the Southern Coast - ABC\nContext: {context_str}\nAnswer:",
    "clarify": "You are a highly intelligent assistant. Keep your answers brief and accurate. {format}.",
    "location": "What location is specified in this prompt, state None if there isn't one. Use a single word answer. [BEGIN] {prompt} [END]",
    "company": "What company is related to the stock price in this prompt? Please state none if there isn't one. Use a single word answer: [BEGIN] {prompt} [END]",
    "rag": "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Back up your answer using facts from the following context.\\nContext: {context_str}\\nQuestion: {prompt}\\nAnswer:",
    "website": "Summarize the following text from URL {url}:\n[BEGIN] {website_text} [END]\nThe above article is about:",
}

# Test OpenAI API
while True:
    log("Testing OpenAI API...")
    try:
        log(f"Using openai library version {openai.__version__}")
        log(f"Connecting to OpenAI API at {api_base} using model {mymodel}")
        llm = openai.OpenAI(api_key=api_key, base_url=api_base)
        # Get models
        models = llm.models.list()
        # build list of models
        model_list = [model.id for model in models.data]
        log(f"LLM: Models available: {model_list}")
        if len(models.data) == 0:
            log("LLM: No models available - check your API key and endpoint.")
            raise Exception("No models available")
        if not mymodel in model_list:
            log(f"LLM: Model {mymodel} not found in models list.")
            if len(model_list) == 1:
                log("LLM: Switching to default model")
                mymodel = model_list[0]
            else:
                log(f"LLM: Unable to find model {mymodel} in models list.")
                raise Exception(f"Model {mymodel} not found")
        log(f"LLM: Using model {mymodel}")
        # Test LLM
        llm.chat.completions.create(
            model=mymodel,
            max_tokens=MAXTOKENS,
            stream=False,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": "Hello"}],
        )
        break
    except Exception as e:
        log("OpenAI API Error: %s" % e)
        log(f"Unable to connect to OpenAI API at {api_base} using model {mymodel}.")
        log("Sleeping 10 seconds...")
        time.sleep(10)

# Expand variables in prompt to values
def expand_prompt(prompt, values):
    # Always use current {date} and {time}
    current_date = datetime.datetime.now()
    values["date"] = current_date.strftime("%B %-d, %Y")
    values["time"] = current_date.strftime("%-I:%M %p")
    for k in values:
        prompt = prompt.replace(f"{{{k}}}", values[k])
    return prompt

# Function to return base conversation prompt
def base_prompt(content=None):
    global baseprompt, agentname, USE_SYSTEM, prompts
    if agentname == "":
        agentname = prompts["agentname"]
    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime("%B %-d, %Y")
    values = {"agentname": agentname, "date": formatted_date}
    baseprompt = expand_prompt(prompts["baseprompt"], values)
    if not content:
        content = baseprompt
    if USE_SYSTEM:
        return [{"role": "system", "content": content}] 
    else:
        return [{"role": "user", "content": content}, {"role": "assistant", "content": "Okay, let's get started."}] 

# Initialize context 
context = base_prompt()

# Function - Send single prompt to LLM for response
def ask(prompt):
    context = base_prompt()
    try:
        context.append({"role": "user", "content": prompt})
        log(f"messages = {context} - model = {mymodel}")
        llm = openai.OpenAI(api_key=api_key, base_url=api_base)
        response = llm.chat.completions.create(
            model=mymodel,
            max_tokens=MAXTOKENS,
            stream=False, # Wait for completion
            temperature=TEMPERATURE,
            messages=context,
        )
    except openai.OpenAIError as e:
        # If we get an error, try to recover
        context.pop()
        if "maximum context length" in str(e):
            # our context has grown too large, reset
            context = base_prompt()   
            error(f"Reset context due to error: {e}")
        else:
            error(f"ERROR: {e}")
    log(f"ask -> {response.choices[0].message.content.strip()}")
    return response.choices[0].message.content.strip()

def ask_llm(query, format=""):
    # Ask LLM a question
    if format == "":
        format = f"Respond in {format}."
    content = base_prompt(expand_prompt(prompts["clarify"], {"format": format})) + [{"role": "user",
                "content": query}]
    log(f"ask_llm: {content}")
    llm = openai.OpenAI(api_key=api_key, base_url=api_base)
    response = llm.chat.completions.create(
        model=mymodel,
        max_tokens=MAXTOKENS,
        stream=False,
        temperature=TEMPERATURE,
        messages=content,
    )
    log(f"ask_llm -> {response.choices[0].message.content.strip()}")
    return response.choices[0].message.content.strip()

# Function - Get weather for location
def get_weather(location):
    # Look up weather for location
    if location == "":
        location = "Los Angeles"
    location = location.replace(" ", "+")
    url = "https://wttr.in/%s?format=j2" % location
    log(f"Fetching weather for {location} from {url}")
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return "Unable to fetch weather for %s" % location
    
# Function - Get stock price for company
def get_stock(company):
    if ALPHA_KEY == "alpha_key":
        return "Unable to fetch stock price for %s - No Alpha Vantage API Key" % company
    # First try to get the ticker symbol
    symbol = ask_llm(f"What is the stock symbol for {company}? Respond with symbol.")
    if "none" in symbol.lower():
        return "Unable to fetch stock price for %s - No matching symbol" % company
    # Check to see if response has multiple words and if so, pick the last one
    if len(symbol.split()) > 1:
        symbol = symbol.split()[-1]
    # Strip off any spaces or non-alpha characters
    symbol = ''.join(e for e in symbol if e.isalnum())
    # Now get the stock price
    url = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=%s&apikey=%s" % (symbol.upper(), ALPHA_KEY)
    log(f"Fetching stock price for {company} from {url}")
    response = requests.get(url)
    if response.status_code == 200:
        try:
            price = response.json()["Global Quote"]["05. price"]
            return f"The price of {company} (symbol {symbol}) is ${price}."
        except:
            return "Unable to fetch stock price for %s - No data available." % company
    
# Function - Get news from Google
def get_top_articles(url, max=10):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'xml')
    items = soup.findAll('item')
    articles = ""
    count = 0
    for item in items:
        title = item.find('title').string.strip()
        pubdate = item.find('pubDate').string.strip()
        articles += f"Headline: {title} - Pub Date: {pubdate}\n"
        count += 1
        if count >= max:
            break
    return articles

# Function - Fetch news for topic
def get_news(topic, max=10):
    if not topic:
        url = "https://news.google.com/rss/"
    else:
        topic = topic.replace(" ", "+")
        url = "https://news.google.com/rss/search?q=%s" % topic
    log(f"Fetching news for {topic} from {url}")
    response = get_top_articles(url, max)
    return response
    

def fetch_news(topic):
    log("Get News")
    context_str = get_news(topic, 25)
    log(f"News Raw Context = {context_str}")
    prompt = expand_prompt(prompts["news"], {"context_str": context_str})
    return ask(prompt)

def handle_weather_command(p):
    log("Get Weather")
    location = ask_llm(expand_prompt(prompts["location"], {"prompt": p}))
    location = ''.join(e for e in location if e.isalnum() or e.isspace())
    if "none" in location.lower():
        context_str = get_weather("")
    else:
        context_str = get_weather(location)
    prompt = expand_prompt(prompts["weather"], {"prompt": p, "context_str": context_str, "location": location})
    return ask(prompt)

def handle_stock_command(prompt):
    log("Stock prompt")
    if not prompt:
        return
    log(f"Stock Prompt: {prompt}")
    company = ask_llm(expand_prompt(prompts["company"], {"prompt": prompt}))
    company = ''.join(e for e in company if e.isalnum() or e.isspace())
    if "none" in company.lower():
        context_str = "Unable to fetch stock price - Unknown company specified." 
    else:
        context_str = get_stock(company)
    log(f"Company = {company} - Context = {context_str}")
    return context_str

# main
if __name__ == "__main__":
    print("TinyLLM News Bot")
    print("")

    # Date
    current_date = datetime.datetime.now()
    print(f"Current date: {current_date.strftime('%B %-d, %Y')}")
    print("")
    # Initialize context
    context = base_prompt()

    # Query LLM for weather
    weather = handle_weather_command("Los Angeles")
    print(f"Weather:\n{weather}")
    print("")

    # Query LLM for stock
    stock = handle_stock_command("Disney")
    print(f"Disney Stock: {stock}")
    print("")

    # Query LLM for news
    news = fetch_news("")
    print(f"News:\n{news}")
    print("")

