# TinyLLM Chatbot Image Generation

This is to document the steps to add image generation to the TinyLLM Chatbot.

# SwarmUI

The [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) package has a very robust API set and using ComfyUI on the backend, supports multiple image generation models including Stable Diffusion, Flux, and others. I have created a simple container that runs SwarmUI and makes it easy to set up as a service that the Chatbot can call.

## Setup

The following will install a instance of SwarmUI container I have built and uploaded. This assumes an Nvidia GPU rig and will run as a daemon.

```bash
# Install and run SwarmUI container
docker run -d --gpus all \
    -p 7801:7801 \
    --restart unless-stopped \
    --name swarmui \
    --mount source=swarmdata,target=/SwarmUI/Data \
    --mount source=swarmbackend,target=/SwarmUI/dlbackend \
    --mount source=swarmdlnodes,target=/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/DLNodes \
    -v "$PWD/.swarmui/Models:/SwarmUI/Models" \
    -v "$PWD/.swarmui/Output:/SwarmUI/Output" \
    -v "$PWD/.swarmui/CustomWorkflows:/SwarmUI/src/BuiltinExtensions/ComfyUIBackend/CustomWorkflows" \
    jasonacox/swarmui:0.9.6 \
    --forward_restart $@

# Watch the install logs (optional) - ^C to exit log viewing
docker logs swarmui -f
```

Go to http://localhost:7801 and continue the service setup which will auto-download the Stable Diffusion XL model:

## SwarmUI Installer

<img width="800" alt="image" src="https://github.com/user-attachments/assets/3dadb806-84d9-4517-a241-ab9b19e35c7d" />

You can use "Just Install":

<img width="800" alt="image" src="https://github.com/user-attachments/assets/74f1fadf-56f4-4944-8d9b-141bbb4dd2c2" />

Install Now:

<img width="800" alt="image" src="https://github.com/user-attachments/assets/d3f1f4b9-7c09-40b6-a33e-8da43bda8a19" />

This will take a while for it to download the model and additional tools (e.g. ComfyUI). Once the UI loads, you can select the model (SDXL) and type a prompt for it to generate:

<img width="800" alt="image" src="https://github.com/user-attachments/assets/ba1f67ac-87d9-4bb4-a23d-41e8b9217a7a" />

## API Test

We need to be able to make API calls to get an image. We don't need the server to save the file. We just need it to send us the base64 encoded version that we can have python manipulate and display in the browser. Here is a simple [test.py](./test.py):

```python
import io
import base64
import requests
from PIL import Image

HOST = "http://localhost:7801"  # SwarmUI server URL

# Request parameters for image
data = {
    "session_id": None,
    "images": "1",
    "prompt": "a kitten in a park",
    "model": "OfficialStableDiffusion/sd_xl_base_1.0",
    "width": "1024",
    "height": "1024",
    "cfgscale": "7.0",      # CFG Scale use 7.0 for SDXL and 1.0 for Flux
    "steps": "20",
    "seed": "-1",           # -1 for random seed
    "donotsave": True,      # True = send image as base64
}

# Get a new session ID from the SwarmUI server
headers = {"Content-Type": "application/json"}
response = requests.post(f"{HOST}/API/GetNewSession", headers=headers, data="{}", 
                            timeout=10)
response.raise_for_status()
data["session_id"] = response.json()["session_id"]

# Get the image in base64 format
response = requests.post(f"{HOST}/API/GenerateText2Image", headers=headers, 
                            json=data, timeout=300)
response.raise_for_status()
image_encoded = response.json()["images"][0]

# Decode the base64 image and display it
image = Image.open(io.BytesIO(base64.b64decode(image_encoded.split(",")[1])))
image.show()
```

<img width="200" alt="image" src="https://github.com/user-attachments/assets/d7a2bf0a-a78a-4689-abf7-b2d36640ab2f" />


## Command Line Tool

There is a [image_generator.py](./image_generator.py) script that allows the user to test out different prompts, settings and models from the command line:

```bash
    # Specify parameters
    python3 image_generator.py  "a kitten in a basket" --host "http://localhost:7801" \
        --model "OfficialStableDiffusion/sd_xl_base_1.0" \
        --width 1024 --height 1024 --cfgscale 7.5 --steps 20 --seed -1 \
        --save "output.png"

    # Use defaults
    python3 image_generator.py "anime boy"

    # Interactive session
    python3 image_generator.py
```

# TODO

Incorporate this tool into the chatbot to use with slash (/) commands and intent router.
