#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SwarmUIImageGenerator class for generating images using the SwarmUI API.

This class provides asynchronous methods to initialize the API connection, generate images based on prompts,
and handle session management. It uses aiohttp for async API calls and the PIL library
to handle image processing.

Test: python3 app/image/swarmui_generator.py   

Author: Jason A. Cox
2 Aug 2025
github.com/jasonacox/TinyLLM
"""
import io
import base64
import aiohttp
import requests
from typing import Optional

from PIL import Image
from .base import BaseImageGenerator

if __name__ == "__main__":
    def log(text):
        print(text)
    def debug(text):
        print(text)
else:
    from app.core.config import (log, debug)

class SwarmUIImageGenerator(BaseImageGenerator):
    def __init__(self, host="http://localhost:7801", model="OfficialStableDiffusion/sd_xl_base_1.0",
                 width=1024, height=1024, cfgscale=7.5, steps=20, seed=-1, gen_timeout=300):
        """Initialize the SwarmUIImageGenerator with the API host."""
        super().__init__(
            host=host, model=model, width=width, height=height,
            cfgscale=cfgscale, steps=steps, seed=seed, gen_timeout=gen_timeout
        )
        if not host.startswith(("http://", "https://")):
            host = "http://" + host
        self.host = host.rstrip('/')
        self.session_id = None
        self.model = model
        self.width = width
        self.height = height
        self.cfgscale = cfgscale
        self.steps = steps
        self.seed = seed
        self.gen_timeout = gen_timeout
        self._session = None

    def get_provider_name(self) -> str:
        """Get the name of the image generation provider."""
        return "SwarmUI"

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    def test_connection(self):
        """Test the connection to the API server."""
        debug(f"ImageGenerator params: {self.__dict__}")
        try:
            response = requests.get(f"{self.host}/API/GetNewSession", timeout=10)
            debug(f"ImageGenerator response: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            log(f"SwarmUI server ({self.host}) not reachable: {e}")
            return False

    async def async_test_connection(self):
        """Test the connection to the API server."""
        debug(f"Async ImageGenerator params: {self.__dict__}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.host}/API/GetNewSession", timeout=10) as response:
                    return response.status == 200
        except Exception as e:
            log(f"SwarmUI server ({self.host}) not reachable: {e}")
            return False

    async def _get_session_id(self):
        """Get a new session ID from the API."""
        headers = {"Content-Type": "application/json"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.host}/API/GetNewSession",
                                     headers=headers,
                                     json={},
                                     timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.session_id = data["session_id"]
                        return self.session_id
        except Exception as e:
            log(f"Server ({self.host}) error: {e}")
            self.session_id = None
            return None

    async def generate(self, prompt: str = "a kitten in a basket", **kwargs) -> Optional[str]:
        """
        Generate an image using the specified parameters asynchronously.
        """
        if not self.session_id:
            self.session_id = await self._get_session_id()
        if not self.session_id:
            debug("AsyncImageGenerator: Unable to get session ID")
            return None

        # Extract parameters from kwargs with defaults
        model = kwargs.get('model', self.model)
        width = kwargs.get('width', self.width)
        height = kwargs.get('height', self.height)
        cfgscale = kwargs.get('cfgscale', self.cfgscale)
        steps = kwargs.get('steps', self.steps)
        seed = kwargs.get('seed', self.seed)

        params = {
            "model": model,
            "width": width,
            "height": height,
            "cfgscale": cfgscale,
            "steps": steps,
            "seed": seed,
        }

        data = {
            "session_id": self.session_id,
            "images": "1",
            "prompt": str(prompt),
            **{k: str(v) for k, v in params.items()},
            "donotsave": True,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.host}/API/GenerateText2Image",
                    headers={"Content-Type": "application/json"},
                    json=data,
                    timeout=self.gen_timeout
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        image_encoded = response_data["images"][0]
                        return image_encoded

        except Exception as e:
            log(f"SwarmUI server ({self.host}) error: {e}")
            return None

# Test
if __name__ == "__main__":
    import asyncio
    import os

    async def main():
        # Use environment variable or default to localhost for portability
        test_host = os.environ.get("SWARMUI_TEST_HOST", "http://localhost:7801")
        image_gen = SwarmUIImageGenerator(host=test_host)
        
        print(f"Testing SwarmUI connection to: {test_host}")
        
        async def test_image_gen():
            image_encoded = await image_gen.generate(prompt="A beautiful sunset over the mountains")
            if image_encoded:
                image = Image.open(io.BytesIO(base64.b64decode(image_encoded.split(",")[1])))
                print(f"Height = {image.height}, Width = {image.width}")
                image.show()
            else:
                print("Image generation failed")
        # Test the connection and generate an image
        if image_gen.test_connection():
            print("Connection successful")
            await test_image_gen()
        else:
            print("Connection failed")

    asyncio.run(main())
