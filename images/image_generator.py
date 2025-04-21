#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Generation Commandline Tool
========================================
This script generates an image using a specified model and prompt via an API.
It retrieves a session ID, sends a request to generate an image, and downloads the image.

Requires: SwarmUI API 

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
    jasonacox/swarmui \
    --forward_restart $@

Usage arguments: 
    python3 image_generator.py <prompt> [args]

    Args:
    --model: The model to use for generation.
    --width: The width of the generated image.
    --height: The height of the generated image.
    --cfgscale: CFG scale value.
    --steps: Number of steps for generation.
    --seed: Seed value for generation.
    --host: API host URL.
    --save: Path to save the generated image.
    --help: Show this help message and exit.

Example usage:
    python3 image_generator.py --prompt "a kitten in a basket" \
        --model "OfficialStableDiffusion/sd_xl_base_1.0" \
        --width 1024 --height 1024 --cfgscale 7.5 --steps 20 --seed -1 \
        --host "http://localhost:7801" --save "output.png"
    python3 image_generator.py --prompt "anime boy"
    python3 image_generator.py

Author: Jason A. Cox
20 Apr 2025
github.com/jasonacox/TinyLLM
"""

import os
import io
import sys
import base64
import requests
from PIL import Image
import argparse

# Check for arguments
parser = argparse.ArgumentParser(description="Generate an image using a specified model and prompt.")
# Add arguments
parser.add_argument("prompt", type=str, nargs='?', default="a kitten in a basket", help="Prompt for image generation")
parser.add_argument("--model", type=str, help="Model to use for generation")
parser.add_argument("--width", type=int, default=1024, help="Width of the generated image")
parser.add_argument("--height", type=int, default=1024, help="Height of the generated image")
parser.add_argument("--cfgscale", type=float, default=7.5, help="CFG scale value")
parser.add_argument("--steps", type=int, default=20, help="Number of steps for generation")
parser.add_argument("--seed", type=int, default=-1, help="Seed value for generation")
parser.add_argument("--host", type=str, default="http://localhost:7801", help="API host URL")
parser.add_argument("--save", type=str, help="Path to save the generated image")
# Parse the arguments
if len(sys.argv) == 1:
    # prompt user for input
    prompt = input("Enter the prompt for image generation: ")
    if not prompt:
        sys.exit("Prompt is required.")
    model = input("Enter the model to use for generation [OfficialStableDiffusion/sd_xl_base_1.0]: ")
    if not model:
        model = "OfficialStableDiffusion/sd_xl_base_1.0"
    width = input("Enter the width of the generated image [1024]: ")
    if not width:
        width = 1024
    else:
        width = int(width)
    height = input("Enter the height of the generated image [1024]: ")
    if not height:
        height = 1024
    else:
        height = int(height)
    cfgscale = input("Enter the CFG scale value [7.5]: ")
    if not cfgscale:
        cfgscale = 7.5
    else:
        cfgscale = float(cfgscale)
    steps = input("Enter the number of steps for generation [20]: ")
    if not steps:
        steps = 20
    else:
        steps = int(steps)
    seed = input("Enter the seed value for generation [-1]: ")
    if not seed:
        seed = -1
    else:
        seed = int(seed)
    host = input("Enter the API host URL [http://localhost:7801]: ")
    if not host:
        host = "http://localhost:7801"
    save_path = input("Enter the path to save the generated image [output.png]: ")
    if not save_path:
        save_path = "output.png"
    else:
        save_path = os.path.abspath(save_path)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
    # Check if the host URL ends with a slash and remove it
else:
    # parse command line arguments

    args = parser.parse_args()
    prompt = args.prompt
    if args.model:
        model = args.model
    else:
        model = "OfficialStableDiffusion/sd_xl_base_1.0"
    width = args.width
    height = args.height
    cfgscale = args.cfgscale
    steps = args.steps
    seed = args.seed
    host = args.host
    save_path = args.save
    if save_path:
        save_path = os.path.abspath(save_path)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
    else:
        save_path = os.path.join(os.getcwd(), "output.png")
# Set the base URL for the API
# Check if the host URL ends with a slash and remove it
if host.endswith("/"):
    host = host[:-1]

def get_session_id():
    """Get a new session ID from the API."""
    headers = {"Content-Type": "application/json"}
    try:
        # Send the request to the API
        response = requests.post(f"{host}/API/GetNewSession", headers=headers, data="{}", 
                                 timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Server ({host}) error: {e}")
        return None
    # Check if the response is valid
    if response.status_code != 200:
        print(f"Server ({host}) error: {response.status_code} - {response.text}")
        return None
    return response.json()["session_id"]

def generate_image(session_id, prompt, model, width, height, cfgscale=cfgscale,
                   steps=steps, seed=seed, host=host):
    """Generate an image using the provided session ID, prompt, model, width, and height.
    Args:
        session_id (str): The session ID for the API.
        prompt (str): The prompt for image generation.
        model (str): The model to use for generation.
        width (int): The width of the generated image.
        height (int): The height of the generated image.
        cfgscale (float): CFG scale value.
        steps (int): Number of steps for generation.
        seed (int): Seed value for generation.
    Returns:
        str: The base64 encoded image data.    
    """
    text2image_post_data = {
        # Basic generation parameters
        "session_id": "xxxx",                      # Session ID for the API
        "prompt": "a kitten in a basket",          # Prompt for image generation
        "negativeprompt": "",                      # Negative prompt for generation
        "model": "HiDream/hidream_i1_dev_fp8",     # Model to use for generation

        # Image dimensions and batch settings
        "width": "1024",                           # Width of the generated image
        "height": "1024",                          # Height of the generated image
        "aspectratio": "1:1",                      # Aspect ratio of the image
        "batchsize": "1",                          # Batch size for generation
        "colordepth": "8bit",                      # Color depth of the image

        # Generation control parameters
        "steps": "20",                             # Number of steps for generation
        "seed": "-1",                              # Seed for generation 
        "cfgscale": "1",                           # CFG scale value
        "renormcfg": "0",                          # Renormalization CFG

        # Model-specific settings
        "automaticvae": True,                      # Use automatic VAE
        "modelspecificenhancements": True,         # Model specific enhancements
        "torchcompile": "Disabled",                # Torch compile mode
        "internalbackendtype": "Any",              # Internal backend type

        # LORA settings
        "loras": "",                               # LORA models to use
        "loraweights": "",                         # LORA weights
        "lorasectionconfinement": "",              # LORA section confinement
        "loratencweights": "",                     # LORA tensor weights

        # Advanced processing options
        "gligenmodel": "None",                     # GLIGEN model to use
        "removebackground": False,                 # Remove background
        "seamlesstileable": "false",               # Seamless tileable
        "shiftedlatentaverageinit": False,         # Shifted latent average init
        "usecfgzerostar": False,                   # Use CFG zero star
        "zeronegative": False,                     # Zero negative

        # Output and preview settings
        "donotsave": True,                         # Do not save the image = base64 encoded
        "donotsaveintermediates": False,           # Do not save intermediate images
        "nopreviews": False,                       # No previews
        "outputintermediateimages": False,         # Output intermediate images

        # Miscellaneous settings
        "webhooks": "Normal",                      # Webhooks for generation
        "personalnote": "",                        # Personal note for the image
        "extra_metadata": {},                      # Extra metadata for the image
        "presets": [],                             # Presets for generation

        # Regional and mask settings
        "controlnetpreviewonly": False,            # ControlNet preview only
        "debugregionalprompting": False,           # Debugging for regional prompting
        "maskcompositeunthresholded": False,       # Mask composite unthresholded
        "regionalobjectcleanupfactor": "0",        # Regional object cleanup factor
        "savesegmentmask": False,                  # Save segment mask
        "segmentsortorder": "left-right",          # Segment sort order

        # Video-specific settings
        "trimvideoendframes": "0",                 # Trim video end frames
        "trimvideostartframes": "0",               # Trim video start frames
        "videopreviewtype": "animate",             # Video preview type

        # Generation behavior
        "colorcorrectionbehavior": "None",         # Color correction behavior
        "noseedincrement": False,                  # No seed increment
        "wildcardseedbehavior": "Random",          # Wildcard seed behavior
    }
    headers = {"Content-Type": "application/json"}
    data = {
        "session_id": session_id,
        "images": "1",
        "prompt": prompt,
        "model": model,
        "width": str(width),
        "height": str(height),
        "cfgscale": str(cfgscale),
        "steps": str(steps),
        "seed": str(seed),
        "donotsave": True,
    }
    try:
        # Send the request to the API
        response = requests.post(f"{host}/API/GenerateText2Image", headers=headers, 
                                 json=data, timeout=300)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Server ({host}) error: {e}")
        return None
    # Check if the response is valid
    if response.status_code != 200:
        print(f"Server ({host}) error: {response.status_code} - {response.text}")
        return None
    #print(f"Response: {response.json()}")
    image_encoded = response.json()["images"][0]
    return image_encoded

def download_image(image_url):
    """Download the generated image from the provided URL."""
    response = requests.get(image_url)
    response.raise_for_status()
    image_name = os.path.basename(image_url)
    with open(image_name, "wb") as f:
        f.write(response.content)
    return image_name

def main():
    session_id = get_session_id()
    # Print Banner and Info
    print("Image Generation Script")
    print("=======================================")
    print(f" - Prompt: {prompt}")
    print(f" - Model: {model}")
    print(f" - Width: {width}")
    print(f" - Height: {height}")
    print(f" - CFG Scale: {cfgscale}")
    print(f" - Steps: {steps}")
    print(f" - Seed: {seed}")
    print(f" - Host: {host}")
    print(f" - Save Path: {save_path}")
    print("=======================================")
    # Print the session ID  
    print(f"Session ID: {session_id}")
    print("=======================================")
    
    # Generate the image
    image_encoded = generate_image(session_id, prompt, model, width, height, cfgscale, steps, seed, host)
    if not image_encoded:
        print("Failed to generate image.")
        return
    image = Image.open(io.BytesIO(base64.b64decode(image_encoded.split(",")[1])))
    image.show()
    # Save the image if a save path is provided
    if save_path:
        image.save(save_path)
        print(f"Image saved to {save_path}")
    else:
        print("No save path provided, image not saved.")


if __name__ == "__main__":
    main()
