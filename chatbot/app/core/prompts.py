"""
ChatBot - System Prompts

This module provides functions to load, save, and manage system prompts for the chatbot.
It includes default prompts for various tasks such as weather, stock prices, news, and more.
It also includes functions to expand variables in prompts and reset prompts to default values.

Author: Jason A. Cox
20 Apr 2025
github.com/jasonacox/TinyLLM
"""
# pylint: disable=invalid-name
# pylint: disable=global-statement
# pylint: disable=global-variable-not-assigned

# Standard library imports
import datetime
import json
import os

# Local imports
from app.core.config import (
    AGENTNAME,
    MAXTOKENS,
    PROMPT_FILE,
    TEMPERATURE,
    USE_SYSTEM,
    baseprompt,
    log,
    prompts,
)

# Prompt Defaults
default_prompts = {}
default_prompts["greeting"] = "Hi"
default_prompts["agentname"] = "Jarvis"
default_prompts["baseprompt"] = "You are {agentname}, a highly intelligent assistant. The current date is {date} and time is {time}. You should give concise responses to very simple questions, but provide thorough responses to more complex and open-ended questions. Don't mention any of the above unless asked and keep your greetings brief."
default_prompts["weather"] = "You are a weather forecaster. Keep your answers brief and accurate. Current date is {date} and weather conditions:\n[DATA]{context_str}[/DATA]\nProvide a weather update, current weather alerts, conditions, precipitation and forecast for {location} and answer this: {prompt}."
default_prompts["stock"] = "You are a stock analyst. Keep your answers brief and accurate. Current date is {date}."
default_prompts["news"] = "You are a newscaster who specializes in providing headline news. Use only the following context provided by Google News to summarize the top 10 headlines for today. Rank headlines by most important to least important. Always include the news organization and ID. Do not add any commentary.\nAlways use this format:\n#. [News Item] - [News Source] - LnkID:[ID]\nHere are some examples, but do not use them: \n1. The World is Round - Science - LnkID:91\n2. The Election is over and Children have won - US News - LnkID:22\n3. Storms Hit the Southern Coast - ABC - LnkID:55\nContext: {context_str}\nTop 10 Headlines with Source and LnkID:"
default_prompts["clarify"] = "You are a highly intelligent assistant. Keep your answers brief and accurate."
default_prompts["location"] = "What location is specified in this prompt, state None if there isn't one. Use a single word answer. [BEGIN] {prompt} [END]"
default_prompts["company"] = "What company is related to the stock price in this prompt? Please state none if there isn't one. Use a single word answer: [BEGIN] {prompt} [END]"
default_prompts["rag"] = "You are an assistant for question-answering tasks. Use the above discussion thread and the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Back up your answer using facts from the following context. Do not mention your answer was based on this context.\nContext: {context_str}\nQuestion: {prompt}\nAnswer:"
default_prompts["website"] = "Summarize the following text from URL {url}:\n[BEGIN] {website_text} [END]\nExplain what the link is about and provide a summary with the main points."
default_prompts["LLM_temperature"] = TEMPERATURE
default_prompts["LLM_max_tokens"] = MAXTOKENS
default_prompts["toxic_filter"] = "You are a highly intelligent assistant. Review the following text and filter out any toxic or inappropriate content. Please respond with a toxicity rating. Use a scale of 0 to 1, where 0 is not toxic and 1 is highly toxic. [BEGIN] {prompt} [END]"
default_prompts["chain_of_thought_check"] = """You are a language expert.
    Consider this prompt:
    <prompt>{prompt}</prompt>
    Categorize the request using one of these:
    a) A request for information
    b) A request for code
    c) A greeting or word of appreciation
    d) Something else
    Answer with a, b, c or d only:
    """
default_prompts["chain_of_thought"] = """First, outline how you will approach answering the problem.
    Break down the solution into clear steps.
    Continuously adjust your reasoning based on intermediate results and reflections, adapting your strategy as you progress. 
    Regularly evaluate progress. 
    Be critical and honest about your reasoning process.
    Use thoughts as a scratchpad, writing out all calculations and reasoning explicitly. 
    Synthesize the final answer within <answer> tags, providing a clear informed and detailed conclusion.
    Include relevant scientific and factual details to support your answer.
    If providing an equation, make sure you define the variables and units.
    Don't over analyze simple questions.
    If asked to produce code, include the code block in the answer. 
    Answer the following in an accurate way that a young student would understand: 
    {prompt}"""
default_prompts["chain_of_thought_summary"] = """Examine the following context:\n{context_str}

Provide the best conclusion based on the context.
    Do not provide an analysis of the context. Do not include <answer> tags.
    Include relevant scientific and factual details to support the answer.
    If there is an equation, make sure you define the variables and units. Do not include an equation section if not needed.
    If source code provided, include the code block and describe what it does. Do not include a code section otherwise.
    Make sure the answer addresses the original prompt: {prompt}
    """
default_prompts_intent = {
    "weather": "if the user is asking for weather or current temperature information.",
    "stock": "if the user is asking for stock information.",
    "current": "if the user is asking if the data is current.",
    "news": "if and only if the user is asking for headline news or current news about a person, place or company.",
    "search": "if the user is asking us to search the internet or web.",
    "retry": "if the user asks us to try again.",
    "code": "if the user is asking for code.",
    "image": "if the user is asking for an image to be made.",
    "other": "if the user is not asking for any of the above."
}
default_prompts["intent"] = """<BACKGROUND>{prompt_context}</BACKGROUND>
    You are a language expert. Consider this user prompt:
    <PROMPT>
    {prompt}
    </PROMPT>
    Pay the most attention to the last request. Categorize the user's intent in the prompt using one of these:
    """ + "\n    ".join([f"{k}) {v}" for k, v in default_prompts_intent.items()]) + "\nLimit your response to one of the above categories."
default_prompts["internet_rag"] = "You are an assistant for question-answering tasks. Use the above discussion thread and the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Back up your answer using facts from the following context. Do not mention your answer was based on this context.\nInternet Search Context: {context_str}\n\nQuestion: {prompt}\nAnswer:"

# Load system prompts from PROMPT_FILE
def load_prompts():
    global prompts
    newprompts = False
    try:
        with open(PROMPT_FILE, "r") as f:
            prompts = json.load(f)
            log(f"Loaded {len(prompts)} prompts.")
        # Ensure prompts always include all keys from default_prompts
        for k in default_prompts:
            if k not in prompts:
                newprompts = True
                prompts[k] = default_prompts[k]
    except:
        log(f"Unable to load system prompts file {PROMPT_FILE}, creating with defaults.")
        reset_prompts()
        save_prompts()
    if newprompts:
        log("Updating system prompts with new elements.")
        save_prompts()

# Save prompts to PROMPT_FILE
def save_prompts():
    global prompts
    try:
        os.makedirs(os.path.dirname(PROMPT_FILE), exist_ok=True)  # Create path if it doesn't exist
        with open(PROMPT_FILE, "w") as f:
            json.dump(prompts, f)
            log(f"Saved {len(prompts)} prompts.")
    except:
        log("Unable to save prompts.")

# Expand variables in prompt to values
def expand_prompt(prompt, values):
    # Always use current {date} and {time}
    current_date = datetime.datetime.now()
    values["date"] = current_date.strftime("%B %-d, %Y")
    values["time"] = current_date.strftime("%-I:%M %p")
    for k in values:
        prompt = prompt.replace(f"{{{k}}}", values[k])
    return prompt

# Reset prompts
def reset_prompts():
    global prompts
    prompts = {}
    for k in default_prompts:
        prompts[k] = default_prompts[k]

# Load prompts
load_prompts()

# Function to return base conversation prompt
def base_prompt(content=None):
    global baseprompt, AGENTNAME, USE_SYSTEM, prompts
    if AGENTNAME == "":
        AGENTNAME = prompts["agentname"]
    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime("%B %-d, %Y")
    values = {"agentname": AGENTNAME, "date": formatted_date}
    baseprompt = expand_prompt(prompts["baseprompt"], values)
    if not content:
        content = baseprompt
    if USE_SYSTEM:
        return [{"role": "system", "content": content}]
    else:
        return [{"role": "user", "content": content}, {"role": "assistant", "content": "Okay, let's get started."}]
