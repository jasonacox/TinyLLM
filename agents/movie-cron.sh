#!/bin/bash

# Example cron script to generate and process a daily movie recommendation
# This script can be run on a schedule (e.g., via cron) to automate movie suggestions.
#
# Setup:
# 1. Set up your LLM API endpoint
# 2. Set up any required environment variables for your environment
# 3. Adjust file/database paths as needed for your deployment

export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="your-api-key-here"
export LLM_MODEL="llama-3-vision"
export DATABASE="/path/to/movie.db"
export MESSAGE_FILE="/path/to/message.txt"
export SEARXNG_URL="http://localhost:8080"
export DEBUG="false"

echo ""

MOVIE=$(python3 ./movie.py)
echo "MOVIE: $MOVIE"

# Send the movie recommendation to user (uncomment and configure as needed)
#python3 ./send_page.py "$MOVIE"

# Convert all \n to <br> for HTML output
cat "$MESSAGE_FILE" | sed 's/$/<br>/' | tr -d '\n' > /tmp/movies-out.txt
cat "$MESSAGE_FILE"

# Save MESSAGE to MESSAGE_FILE (already done by movie.py if needed)

# Send email
SUBJECT="Subject: Movie Suggestion 
Content-Type: text/html

"
echo "$SUBJECT" > /tmp/moviebot

# Send message
cat $MESSAGE_FILE | sed 's/$/<br>/' | tr -d '\n' >> /tmp/moviebot
echo "----" >> /tmp/moviebot
echo "$MOVIE" >> /tmp/moviebot

# Email
#cat /tmp/moviebot | msmtp -a gmail name@email.com
