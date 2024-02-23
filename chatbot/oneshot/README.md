# Chatbot for One-Shot Answers

This version of the chatbot allows you to set it up to not remember conversation thread and consider every prompt a new engagement. If the Weaviate vector database is configured, it will perform retrieval augmented generation for every prompt.

Important new Environmental variables:

* WEAVIATE_HOST - Weaviate Host for RAG (Optional)
* WEAVIATE_LIBRARY - Weaviate Library for RAG (Optional)
* RESULTS - Number of results to return from RAG query (default=1)
* ONESHOT - Set to True to enable one-shot mode

