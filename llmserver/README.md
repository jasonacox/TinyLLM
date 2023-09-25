# llmserver

The `llmserver` is a docker hosted version of the python llama_cpp.server which serves up a LLM with OpenAI API interface.

## Setup

```bash
# Create Container
docker build -t llmserver .

# Run Container
docker run \
    -d \
    -p 8000:8000 \
    -v ./models:/app/models \
    -e MODEL=models/llama-2-7b-chat.Q5_K_M.gguf \
    -e N_GPU_LAYERS=32 \
    -e HOST=0.0.0.0 \
    -e PORT=8000 \
    --name llmserver \
    --restart unless-stopped \
    llmserver
```
