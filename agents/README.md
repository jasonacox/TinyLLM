# LLM Agents

Agents use LLMs to take action or provide curated reports.

## News Bot

The [news.py](news.py) script will use an OpenAI API based LLM to process raw weather payloads, stock prices and top news items into a curated news report. This can be set up as a scheduled cron job to send you the news.

```bash
# Update environmental variables for your use case
ALPHA_KEY=ABC123 OPENAI_API_BASE=http://localhost:8000/v1 COMPANY=MyCompany python3 news.py  
```

## Movie Bot

The [movie.py](movie.py) script uses an LLM to make a movie suggestion. It records previous recommendations to prevent repeating suggestions. Outputs a single line recommendation and writes a full detailed recommendation to MESSAGE_FILE. This can be set up as a scheduled cron job to send you a text or email you a recommendation.

```bash
export OPENAI_API_BASE="http://localhost:4000/v1"
export OPENAI_API_KEY="sk-3-laws-safe"
export LLM_MODEL="llama"
export DATABASE="./movie.db"
export MESSAGE_FILE="./message.txt"

python3 ./movie.py
```

## Conversational Agent

The [conversational.py](conversational.py) script uses an LLM (or two LLMs) to create a back and forth conversation.
This script sets up a teacher and student identities. The teacher and student LLMs take turns responding to each other.
The conversation continues until a stop prompt is given or a maximum number of rounds is reached. The teacher LLM then provides a summary of the conversation and an evaluation of the student.

```bash
python3 conversational.py
```
