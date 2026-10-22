# Weaviate

Weaviate is an open source, AI-native vector database that includes the option of having an embedded tex2vec sentence transformer.

## docker-compose.yml

```bash
docker compose up -d
```

> **Upgrading from Weaviate 1.23:** the compose files now use Weaviate 1.39. Weaviate only supports upgrading one minor version at a time, so pointing 1.39 directly at a `weaviate_data` folder written by 1.23 is not supported. Either step through each minor version (1.24, 1.25, ... 1.39), or stop the stack, move `weaviate_data` aside and re-ingest your documents.

## Test Weaviate

```python
import requests 
import json
import weaviate
import weaviate.classes as wvc

print("Connecting to Weaviate Vector Database...")
client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051,
    additional_config=weaviate.config.AdditionalConfig(timeout=(15, 150))
)
try:
    print("Pulling blog posts from jasonacox.com...")
    resp = requests.get('https://jasonacox.com/wordpress/json')
    data = json.loads(resp.text)  # Load data

    # Load data
    question_objs = list()
    for i, d in enumerate(data["items"]):
        print(d["title"])
        question_objs.append({
            "title": d["title"],
            "url": d["url"],
            "content_text": d["content_text"],
        })

    # Delete collection
    try:
        client.collections.delete("jasonacox")
    except:
        pass

    # Create collection
    print("Creating collection...")
    questions = client.collections.create(
        name="jasonacox",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers()
    )

    # Add data
    print("Vectorizing and storing blog posts...")
    questions.data.insert_many(question_objs)  # This uses batching under the hood

    # Query the Vector DB
    print("Searching database...")
    questions = client.collections.get("jasonacox")
    
    prompts = ["List some facts about solar.", "What are some tips on leadership?"]

    for query in prompts:
        response = questions.query.near_text(
            query=query,
            limit=5
        )
        print()
        print(query)
        print("------------------")
        for i in response.objects:
            print(f"{i.properties['title']} - {i.properties['url']}")

finally:
    client.close()  # Close client gracefully
    print()
    print("Done.")

```