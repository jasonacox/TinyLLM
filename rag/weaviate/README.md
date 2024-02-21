# Weaviate

Weaviate is an open source, AI-native vector database that includes the option of having an embedded tex2vec sentence transformer.

## docker-compose.yml

```bash
docker compose up -d
```

## Test Weaviate

```python
import requests 
import json
import weaviate

client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051,
    additional_config=weaviate.config.AdditionalConfig(timeout=(5, 15))
)
try:
    resp = requests.get('https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')
    data = json.loads(resp.text)  # Load data

    # Load data
    question_objs = list()
    for i, d in enumerate(data):
        print(d)
        question_objs.append({
            "answer": d["Answer"],
            "question": d["Question"],
            "category": d["Category"],
        })

    # Delete collection
    try:
        client.collections.delete("Question")
    except:
        pass

    # Create collection
    questions = client.collections.create(
        name="Question",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers()
    )

    # Add data
    questions.data.insert_many(question_objs)  # This uses batching under the hood

    # Wrap in try/finally to ensure client is closed gracefully
    questions = client.collections.get("Question")

    response = questions.query.near_text(
        query="biology",
        limit=2
    )
    print("------------------")
    print('Query')
    print(response.objects[0].properties)  # Inspect the first object
    print(response.objects[1].properties)  # Inspect the first object

finally:
    client.close()  # Close client gracefully

```