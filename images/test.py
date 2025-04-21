#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python script to generate an image using the SwarmUI API.
This script demonstrates how to send a request to the SwarmUI server
to generate an image based on a text prompt and display the result.

Author: Jason A. Cox
20 Apr 2025
github.com/jasonacox/TinyLLM
"""

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
