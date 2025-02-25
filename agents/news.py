#!/usr/bin/python3
"""
News Bot - Send news alerts to clients

Author: Jason A. Cox
17 Mar 2024
https://github.com/jasonacox/TinyLLM

Requires:
    pip install openai bs4 requests

"""
# Import Libraries
import datetime
import os
import time
import threading
import re

import openai
import requests
from bs4 import BeautifulSoup

# Version
VERSION = "v0.0.3"
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
TIMEOUT = int(os.environ.get("TIMEOUT", 10))                                # Timeout for HTTP requests
MAXTOKENS = int(os.environ.get("MAXTOKENS", 4*1024))                        # Maximum number of tokens to send to LLM
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))                     # LLM temperature
USE_SYSTEM = os.environ.get("USE_SYSTEM", "false").lower() == "true"        # Use system in chat prompt if True
RESULTS = int(os.environ.get("RESULTS", 10))                                # Number of results to return from Weaviate
ALPHA_KEY = os.environ.get("ALPHA_KEY", "alpha-key")                        # Alpha Vantage API Key - https://www.alphavantage.co
COMPANY = os.environ.get("COMPANY", "Google")                               # Company to use for stock news
CITY = os.environ.get("CITY", "Los Angeles")                                # City to use for weather news
CITY_WEEKEND = os.environ.get("CITY_WEEKEND", "Ventura")                    # City to use for weather news on weekends
EMAIL_FORMAT = os.environ.get("EMAIL_FORMAT", "false").lower() == "true"    # Format output for sending email
ABOUT_ME = os.environ.get("ABOUT_ME", 
         "I'm a 31 year old woman who lives in Los Angeles. I have two kids and work as a software engineer.")   
BUDDY_FILE = os.environ.get("BUDDY_FILE", None)             # File to use personalization update 

# Prompt Defaults
prompts = {
    "greeting": "Hi",
    "agentname": "Jarvis",
    "baseprompt": "You are {agentname}, a highly intelligent assistant. The current date is {date}.\n\nYou should give concise responses to very simple questions, but provide thorough responses to more complex and open-ended questions.",
    "weather": "You are a weather forecaster. Keep your answers brief and accurate. Current date is {date} and weather conditions:\n[DATA]{context_str}[/DATA]\nProvide a weather update, current weather alerts, conditions, precipitation and forecast for {location} and answer this: {prompt}.",
    "stock": "You are a stock analyst. Keep your answers brief and accurate. Current date is {date}.",
    "news": "You are a newscaster who specializes in providing headline news. Use only the following context provided by Google News to summarize the top 10 headlines for today. Rank headlines by most important to least important but do not explain why and reduce duplicates. Always include the news organization and ID. List no more than 10 and do not add a preamble or any commentary.\nAlways use this format:\n#. [News Item] - [News Source] - ID: [ID]\nHere are some examples but never use these: \n1. The World is Round - Science - ID: 11\n2. The Election is over and Children have won - US News - ID: 22\n3. Storms Hit the Southern Coast - ABC - ID: 55\n. Context: {context_str}\nTop 10 Headlines with Source and ID:",
    "news_custom": "You are a newscaster who specializes in providing headline news. Use only the following context provided by Google News to summarize the top 10 headlines for today. Rank headlines by most important to least important but do not explain why and reduce duplicates. Always include the news organization and ID. List no more than 10 and do not add a preamble or any commentary.\nAlways use this format:\n#. [News Item] - [News Source] - ID: [ID]\nHere are some examples but never use these: \n1. {topic} is Round - Science - ID: 11\n2. The Election is over and {topic} won - US News - ID: 22\n3. Storms Hit the Southern Coast - ABC - ID: 55\n. Context: {context_str}\nTop 10 Headlines for {topic} with Source and ID:",
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
def ask(prompt, temperature=TEMPERATURE):
    context = base_prompt()
    try:
        context.append({"role": "user", "content": prompt})
        log(f"messages = {context} - model = {mymodel}")
        llm = openai.OpenAI(api_key=api_key, base_url=api_base)
        response = llm.chat.completions.create(
            model=mymodel,
            max_tokens=MAXTOKENS,
            stream=False, # Wait for completion
            temperature=temperature,
            messages=context,
        )
    except openai.OpenAIError as err:
        # If we get an error, try to recover
        context.pop()
        if "maximum context length" in str(err):
            # our context has grown too large, reset
            context = base_prompt()
            error(f"Reset context due to error: {err}")
        else:
            error(f"ERROR: {err}")
    log(f"ask -> {response.choices[0].message.content.strip()}")
    return response.choices[0].message.content.strip()

def ask_llm(query, format="", temperature=TEMPERATURE):
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
        temperature=temperature,
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
    response = requests.get(url, timeout=TIMEOUT)
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
    response = requests.get(url, timeout=TIMEOUT)
    if response.status_code == 200:
        try:
            price = "{:.2f}".format(float(response.json()["Global Quote"]["05. price"]))
            return f"The price of {company} (symbol {symbol}) is ${price}."
        except:
            return "Unable to fetch stock price for %s - No data available." % company

# Function - Get news for topic
def get_top_articles(url, max=10):
    response = requests.get(url, timeout=TIMEOUT)
    soup = BeautifulSoup(response.text, 'xml')
    items = soup.find_all('item')
    articles = ""
    count = 0
    for item in items:
        title = item.find('title').string.strip()
        articles += f"Headline: {title}\n"
        count += 1
        if count >= max:
            break
    return articles

# Cache class to store news items
class Cache:
    def __init__(self, ttl=60):
        self.cache = {}
        self.ttl = ttl
        self.uuid = 0
        self.lock = threading.Lock()

    def set(self, value):
        with self.lock:
            self.cache[self.uuid] = {"value": value, "time": time.time()}
            self.uuid += 1
            return self.uuid - 1

    def get(self, key):
        with self.lock:
            # Clear out old cache items
            for k in list(self.cache.keys()):
                if time.time() - self.cache[k]["time"] > self.ttl:
                    del self.cache[k]
            if key in self.cache:
                return self.cache[key]["value"]
            return None

# Global cache for news items
news_cache = Cache(60)

# Function - Get news from Google
def get_news(topic, max=10):
    if not topic:
        url = "https://news.google.com/rss/"
    else:
        topic = topic.replace(" ", "+")
        url = "https://news.google.com/rss/search?q=%s" % topic
    log(f"Fetching news for {topic} from {url}")
    response = requests.get(url, timeout=TIMEOUT)
    soup = BeautifulSoup(response.text, 'xml')
    items = soup.find_all('item')
    articles = ""
    count = 0
    for item in items:
        title = item.find('title').string.strip()
        #pubdate = item.find('pubDate').string.strip()
        #description = item.find('description').string.strip()
        link = item.find('link').string.strip()
        uuid = news_cache.set(link)
        articles += f"Headline: {title} - ID: {uuid}\n"
        count += 1
        if count >= max:
            break
    return articles


def fetch_news(topic, retries=3, check=False):
    if retries == 0:
        return "Unable to fetch news", "Unable to fetch news"
    log("Get News")
    raw_news = get_news(topic, 25)
    log(f"News Raw Context for topic {topic} = {raw_news}\n\n")
    if topic:
        prompt = expand_prompt(prompts["news_custom"], {"context_str": raw_news, "topic": topic})
    else:
        prompt = expand_prompt(prompts["news"], {"context_str": raw_news})
    answer = ask(prompt)
    # Replace IDs in answer with URLs
    result = ""
    text_only = ""
    all_lines = []
    for line in answer.split("\n"):
        if "ID:" in line:
            elements = line.split("ID: ")
            title = elements[0].strip()
            all_lines.append(title)
            text_only += title + "\n"
            if len(elements) > 1:
                uuid = elements[1].strip()
                # Ensure we have a valid UUId that is a integer
                if not uuid.isdigit():
                    result += line
                    continue    
                url = news_cache.get(int(uuid))
                result += f"{title} <a href=\"{url}\">[Link]</a>"
            else:
                result += line
        else:
            result += line
        result += "\n"
    if check:
        # Query the LLM to see if all_lines are duplicated
        prompt = expand_prompt(prompts["rag"], {"context_str": "\n".join(all_lines), "prompt": "Do these look like the same headline?"})
        answer = ask(prompt)
        if "yes" in answer.lower():
            log("News items are not about {topic}")
            log(f"\n\nresponse={answer}\n\n{all_lines}")
            return fetch_news(topic, retries-1)
    return result, text_only

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

# Use buffer to capture output
output = ""
def buffer(s):
    global output
    output = output + s + "\n"

# main
if __name__ == "__main__":
    buffer("TinyLLM News Bot")
    buffer("")

    # Date
    current_date = datetime.datetime.now()
    buffer(f"Current date: {current_date.strftime('%B %-d, %Y')}")
    buffer("")
    # Initialize context
    context = base_prompt()

    # Get Weather
    try:
        # Query LLM for weather
        if current_date.weekday() >= 5:
            location = CITY_WEEKEND
        else:
            location = CITY
        weather = handle_weather_command(location)
    except Exception as e:
        weather = f"ERROR: {e}"

    # Company Stock Price
    stock = handle_stock_command(COMPANY)
    buffer(f"{COMPANY} Stock: {stock}")
    buffer("")

    # Fetch News Payloads
    news, news_text = fetch_news("")
    company_news, company_text = fetch_news(COMPANY)
    science_news, science_text = fetch_news("Science",check=True)

    # Personalized News Summary
    buddy_request = f"{ABOUT_ME} Provide a paragraph summary of the news that should be most interesting to me. Say it as a concerned friend and are giving me a short update for my day."
    if current_date.weekday() >= 5:
        buddy_request += " It is the weekend so suggestion things to do with the family as well."
    else:
        buddy_request += " It is a work day so add some encouragement."
    buddy = ask_llm(f"Here are the top news items:\n{news_text}\n\n{COMPANY} news:\n{company_text}\n\nScience news:\n{science_text}\n\nWeather:\n{weather}\n\n{buddy_request}")

    # Personalized Summary
    if BUDDY_FILE:
        with open(BUDDY_FILE, "w") as f:
            f.write(buddy)
    buffer(f"{buddy}\n")
    buffer("")

    # Global News
    buffer(f"News:\n{news}")
    buffer("")

    # Company News
    buffer(f"News for {COMPANY}:\n{company_news}")
    buffer("")

    # Science News
    buffer(f"Science News:\n{science_news}")
    buffer("")

    # Weather
    buffer(f"Weather:\n\n{weather}")
    buffer("")

    buffer("\n---")

    # Print version of news
    buffer(f"Newsbot: {VERSION} - {current_date.strftime('%B %-d, %Y')}")
    buffer("\n")

    # Output
    if EMAIL_FORMAT:
        # Clean up output for email
        output = output.replace("\n", "<br>\n")
        # replace 0xb0 with &deg;
        output = output.replace("\xb0", "&deg;")
        # remove unprintable characters
        output = re.sub(r"[^\x00-\x7F]", "", output)
    print(output)
