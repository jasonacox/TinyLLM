"""
ChatBot - Retrieval Augmented Generation (RAG) Functions

This module provides functions to fetch and process information from various sources
such as weather, stock prices, news articles, and documents. It also includes functions for
document management and text extraction from different file formats.

Author: Jason A. Cox
20 Apr 2025
github.com/jasonacox/TinyLLM
"""
# pylint: disable=redefined-outer-name

# Imports
import io

import aiohttp
from bs4 import BeautifulSoup
import requests
from pypdf import PdfReader

# Local imports
from app.core.config import (log, debug, MYMODEL, MAXTOKENS, SEARXNG, RESULTS,
                        WEAVIATE_HOST, WEAVIATE_GRPC_HOST, WEAVIATE_PORT,
                        WEAVIATE_GRPC_PORT, WEAVIATE_AUTH_KEY, UPLOAD_FOLDER,
                        ALPHA_KEY)
from app.rag.documents import Documents
from app.core.llm import ask_llm

# Function - Get weather for location
async def get_weather(location):
    # Look up weather for location
    if location == "":
        location = "Los Angeles"
    location = location.replace(" ", "+")
    url = "https://wttr.in/%s?format=j2" % location
    debug(f"Fetching weather for {location} from {url}")
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        return response.text
    else:
        return "Unable to fetch weather for %s" % location

# Function - Get stock price for company
async def get_stock(company, model=MYMODEL):
    if ALPHA_KEY == "alpha_key":
        return "Unable to fetch stock price for %s - No Alpha Vantage API Key" % company
    # First try to get the ticker symbol
    symbol = await ask_llm(f"What is the stock symbol for {company}? Respond with symbol.", model=model)
    if "none" in symbol.lower():
        return "Unable to fetch stock price for %s - No matching symbol" % company
    # Check to see if response has multiple lines and if so, pick the first one
    symbol = symbol.split("\n")[0].strip()
    # Check to see if there are multiple words and if so, pick the last one
    if len(symbol.split()) > 1:
        symbol = symbol.split()[-1]
    # Strip off any spaces or non-alpha characters
    symbol = ''.join(e for e in symbol if e.isalnum())
    # Now get the stock price
    url = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=%s&apikey=%s" % (symbol.upper(), ALPHA_KEY)
    debug(f"Fetching stock price for {company} from {url}")
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        try:
            price = r.json()["Global Quote"]["05. price"]
            return f"The price of {company} (symbol {symbol}) is ${price}."
        except:
            return "Unable to fetch stock price for %s - No data available." % company

# Function - Get news for topic
async def get_top_articles(url, max=10):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            soup = BeautifulSoup(await response.text(), 'xml')
            items = soup.findAll('item')
            articles = ""
            links = {}
            count = 1
            for item in items:
                title = item.find('title').string.strip()
                #pubdate = item.find('pubDate').string.strip()
                #description = item.find('description').string.strip()
                link = item.find('link').string.strip()
                links[f"LnkID:{count+100}"] = link
                articles += f"Headline: {title} - LnkID:{count+100}\n"
                count += 1
                if count > max:
                    break
            return articles, links

# Function - Fetch news for topic
async def get_news(topic, max=10):
    if "none" in topic.lower() or "current" in topic.lower():
        url = "https://news.google.com/rss/"
    else:
        topic = topic.replace(" ", "+")
        url = "https://news.google.com/rss/search?q=%s" % topic
    debug(f"Fetching news for {topic} from {url}")
    async with aiohttp.ClientSession() as session:
        r, links = await get_top_articles(url, max)
        return r, links

# Function - Extract text from URL
async def extract_text_from_url(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, allow_redirects=True) as response:
                if response.status == 200:
                    # Route extraction based on content type
                    if ";" in response.headers["Content-Type"]:
                        content_type = response.headers["Content-Type"].split(";")[0]
                    else:
                        content_type = response.headers["Content-Type"]
                    content_handlers = {
                        "application/pdf": extract_text_from_pdf,
                        "text/plain": extract_text_from_text,
                        "text/csv": extract_text_from_text,
                        "text/xml": extract_text_from_text,
                        "application/json": extract_text_from_text,
                        "text/html": extract_text_from_html,
                        "application/xml": extract_text_from_text,
                    }
                    if content_type in content_handlers:
                        return await content_handlers[content_type](response)
                    else:
                        return "Unsupported content type"
                else:
                    m = f"Failed to fetch the webpage. Status code: {response.status}"
                    debug(m)
                    return m
    except Exception as erro:
        log(f"An error occurred: {str(erro)}")

# Function - Extract text from PDF
async def extract_text_from_pdf(response):
    # Convert PDF to text
    pdf_content = await response.read()
    pdf2text = ""
    f = io.BytesIO(pdf_content)
    reader = PdfReader(f)
    for page in reader.pages:
        pdf2text = pdf2text + page.extract_text() + "\n"
    return pdf2text

# Function - Extract text from text
async def extract_text_from_text(response):
    return await response.text()

# Function - Extract text from HTML
async def extract_text_from_html(response):
    html_content = await response.text()
    # get title of page from html
    source = "Document Source: " + str(response.url)
    soup = BeautifulSoup(html_content, 'html.parser')
    title = ("Document Title: " + soup.title.string + "\n") if soup.title else ""
    paragraphs = soup.find_all(['p', 'code', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre', 'ol'])
    website_text = f"{title}{source}\nDocument Content:\n" + '\n\n'.join([p.get_text() for p in paragraphs])
    return website_text

# Document Management Settings
rag_documents = Documents(host=WEAVIATE_HOST, grpc_host=WEAVIATE_GRPC_HOST, port=WEAVIATE_PORT,
                          grpc_port=WEAVIATE_GRPC_PORT, retry=3, filepath=UPLOAD_FOLDER,
                          auth_key=WEAVIATE_AUTH_KEY)

# Test Weaviate Connection
if WEAVIATE_HOST != "":
    try:
        rag_documents.connect()
        log(f"RAG: Connected to Weaviate at {WEAVIATE_HOST}")
    except Exception as err:
        log(f"RAG: Unable to connect to Weaviate at {WEAVIATE_HOST} - {str(err)}")
        WEAVIATE_HOST = ""
        log("RAG support disabled.")

# Test SearXNG Connection
if SEARXNG != "":
    try:
        response = requests.get(f"{SEARXNG}/search", timeout=10)
        if response.status_code == 200:
            log(f"SEARXNG: Connected to {SEARXNG}")
        else:
            log(f"SEARXNG Disabled: No search server available at {SEARXNG}")
            SEARXNG = ""
    except Exception as err:
        log(f"SEARXNG: Unable to connect to {SEARXNG} - {str(err)}")
        SEARXNG = ""
        log("SEARXNG support disabled.")

# Find document closely related to query
def query_index(query, library, num_results=RESULTS):
    references = "References:"
    content = ""
    try:
        results = rag_documents.get_documents(library, query=query, num_results=num_results)
    except Exception as erro:
        log(f"Error querying Weaviate: {str(erro)}")
        return None, None
    previous_title = ""
    previous_file = ""
    previous_content = ""
    for ans in results:
        # Skip duplicate titles and files
        if ans['title'] == previous_title and ans['file'] == previous_file:
            continue
        references = references + f"\n - {ans['title']} - {ans['file']}"
        # Skip duplicates of content
        if ans['content'] == previous_content:
            continue
        new_content = ans['content']
        if MAXTOKENS and len(new_content) > MAXTOKENS:
            debug("RAG: Content size exceeded maximum size using chunk.")
            # Cut the middle and insert the chunk in the middle
            new_content = ans['content'][:MAXTOKENS//4] + "..." + (ans.get('chunk') or " ") + "..." + ans['content'][-MAXTOKENS//4:]
        content = content + f"Document: {ans['title']}\nDocument Source: {ans['file']}\nContent: {new_content}\n---\n"
        if (len(content)/4) > MAXTOKENS/2:
            debug("RAG: Content size reached maximum.")
            break
        previous_title = ans['title']
        previous_file = ans['file']
        previous_content = ans['content']
    debug(f"RAG: Retrieved ({len(content)} bytes)")
    return content, references
