#!/bin/bash

# Example cron script to curate relevant news for a user
# This script is run every day at 8am and sends the user an email with the latest news
#
# Setup:
# 1. Install msmtp
# 2. Set up msmtp with your email provider
# 3. Set up a cron job to run this script every day at 8am
# 4. Set up the environment variables below

# Send news
SUBJECT="Subject: Newsbot News
Content-Type: text/html

"
echo "$SUBJECT" > /tmp/newsbot

export ABOUT_ME="I'm a 35 year old woman who lives in Los Angeles. I work in technology and AI. I have 2 boys."
export COMPANY="Google"
export ALPHA_KEY="alpha-key"
export OPENAI_API_BASE="http://localhost:8000/v1"
export CITY="Los Angeles"
export CITY_WEEKEND="Ventura"
export EMAIL_FORMAT=true

python3 news.py >> /tmp/newsbot

# Email
#cat /tmp/newsbot | msmtp -a emailhost example@example.com

# Print
cat /tmp/newsbot