# LLM Agents

Agents use LLMs to take action or provide curated reports.

## News Bot

The [news.py](news.py) script will use an OpenAI API based LLM to process raw weather payloads, stock prices and top news items into a curated news report.

```bash
# Update environmental variables for your use case
ALPHA_KEY=ABC123 OPENAI_API_BASE=http://localhost:8000/v1 COMPANY=MyCompany python3 news.py  
```

