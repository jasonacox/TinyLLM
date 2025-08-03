# TinyLLM Chatbot Image Generation Configuration Examples

## SwarmUI Configuration (Default)

```bash
# Use SwarmUI for image generation (default)
export IMAGE_PROVIDER="swarmui"
export SWARMUI="http://localhost:7801"
export IMAGE_MODEL="OfficialStableDiffusion/sd_xl_base_1.0"
export IMAGE_WIDTH=1024
export IMAGE_HEIGHT=1024
export IMAGE_CFGSCALE=7.5
export IMAGE_STEPS=20
export IMAGE_SEED=-1
export IMAGE_TIMEOUT=300
```

## OpenAI Configuration

```bash
# Use OpenAI for image generation
export IMAGE_PROVIDER="openai"
export OPENAI_API_KEY="your-openai-api-key-here"
export OPENAI_API_BASE="https://api.openai.com/v1"  # Optional, defaults to OpenAI
export OPENAI_IMAGE_MODEL="dall-e-3"    # or "dall-e-2"
export OPENAI_IMAGE_SIZE="1024x1024"    # Valid sizes depend on model
export OPENAI_IMAGE_QUALITY="standard"  # or "hd" (DALL-E 3 only)
export OPENAI_IMAGE_STYLE="vivid"       # or "natural" (DALL-E 3 only)
export IMAGE_TIMEOUT=300
```

## Valid Image Sizes by Model

### DALL-E 2
- 256x256
- 512x512 
- 1024x1024

### DALL-E 3
- 1024x1024 (square)
- 1792x1024 (landscape)
- 1024x1792 (portrait)

## Docker Compose Example

```yaml
version: '3.8'
services:
  chatbot-swarmui:
    image: jasonacox/chatbot:latest
    ports:
      - "5000:5000"
    environment:
      - IMAGE_PROVIDER=swarmui
      - SWARMUI=http://swarmui:7801
      - IMAGE_MODEL=OfficialStableDiffusion/sd_xl_base_1.0
    depends_on:
      - swarmui

  chatbot-openai:
    image: jasonacox/chatbot:latest
    ports:
      - "5001:5000"
    environment:
      - IMAGE_PROVIDER=openai
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_IMAGE_MODEL=dall-e-3
      - OPENAI_IMAGE_SIZE=1024x1024
      - OPENAI_IMAGE_QUALITY=hd
```

## Switching Between Providers

You can switch between providers by changing the `IMAGE_PROVIDER` environment variable:

- Set to `"swarmui"` for local SwarmUI image generation
- Set to `"openai"` for OpenAI DALL-E image generation

The chatbot will automatically use the appropriate configuration and test the connection on startup.

## Notes

- OpenAI image generation requires a valid API key and may incur costs
- SwarmUI requires a local installation and GPU resources
- The chatbot will gracefully fall back to text-only mode if image generation is unavailable
- You can check the current image provider status at `/stats` endpoint
