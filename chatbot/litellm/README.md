# TinyLLM Chatbot with LiteLLM + PostgreSQL

This folder contains a docker-compose file that will start a TinyLLM Chatbot with a LiteLLM proxy and a PostgreSQL database.  The Chatbot will connect to the LiteLLM proxy to access the models.  The LiteLLM proxy will connect to the PostgreSQL database to store usage data.

## Instructions

   1. Edit the config.yaml file to add your models and settings.
   2. Edit the compose.yaml file to adjust the environment variables in the services as needed.
   3. Run `docker compose up -d` to start the services.

The containers will download and launch. The database will be set up in the `./db` folder.

- The Chatbot will be available at http://localhost:5000
- The LiteLLM proxy will be available at http://localhost:4000/ui
- The PostgreSQL pgAdmin interface will be available at http://localhost:5050
