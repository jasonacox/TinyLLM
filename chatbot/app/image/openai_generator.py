#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI Image Generator for generating images using OpenAI's DALL-E API.

This class provides asynchronous methods to generate images using OpenAI's 
image generation API. It supports both DALL-E 2 and DALL-E 3 models.

Author: Jason A. Cox
2 Aug 2025
github.com/jasonacox/TinyLLM
"""

import io
import base64
import aiohttp
import requests
import os
from typing import Optional

from PIL import Image

if __name__ == "__main__":
    def log(text):
        print(text)
    def debug(text):
        print(text)
    VERSION = "Test"
    # For standalone testing, create a minimal base class
    class BaseImageGenerator:
        def __init__(self, **kwargs):
            self.config = kwargs
        def get_provider_name(self):
            raise NotImplementedError
else:
    from app.core.config import log, debug
    from .base import BaseImageGenerator

class OpenAIImageGenerator(BaseImageGenerator):
    def __init__(self, api_key: str = None, api_base: str = None,
                 model: str = None, size: str = None, 
                 quality: str = None, style: str = None, gen_timeout: int = None):
        """
        Initialize the OpenAI Image Generator.
        
        All parameters are optional and will be read from environment variables if not provided.
        This allows for dedicated OpenAI image settings separate from main LLM settings.
        
        Args:
            api_key (str): OpenAI API key (defaults to OPENAI_IMAGE_API_KEY env var)
            api_base (str): API base URL (defaults to OPENAI_IMAGE_API_BASE env var)
            model (str): Model to use (defaults to OPENAI_IMAGE_MODEL env var)
            size (str): Image size (defaults to OPENAI_IMAGE_SIZE env var)
            quality (str): Image quality (defaults to OPENAI_IMAGE_QUALITY env var)
            style (str): Image style (defaults to OPENAI_IMAGE_STYLE env var)
            gen_timeout (int): Request timeout (defaults to IMAGE_TIMEOUT env var)
        """
        # Read from environment variables with fallbacks
        self.api_key = api_key or os.environ.get("OPENAI_IMAGE_API_KEY", "")
        self.api_base = api_base or os.environ.get("OPENAI_IMAGE_API_BASE", "https://api.openai.com/v1")
        self.model = model or os.environ.get("OPENAI_IMAGE_MODEL", "dall-e-3")
        self.size = size or os.environ.get("OPENAI_IMAGE_SIZE", "1024x1024")
        self.quality = quality or os.environ.get("OPENAI_IMAGE_QUALITY", "standard")
        self.style = style or os.environ.get("OPENAI_IMAGE_STYLE", "vivid")
        self.gen_timeout = gen_timeout or int(os.environ.get("IMAGE_TIMEOUT", "300"))
        
        # Always ensure we're using the actual OpenAI API
        if self.api_base != "https://api.openai.com/v1":
            log(f"OpenAI Image Generator: Overriding API base from {self.api_base} to https://api.openai.com/v1")
            self.api_base = "https://api.openai.com/v1"
        
        # Log the configuration being used
        if self.api_key:
            masked_key = "*" * 16
            log(f"OpenAI Image Generator initialized with API key: {masked_key}")
        else:
            log("OpenAI Image Generator initialized with API key: NOT SET")
        
        super().__init__(
            api_key=self.api_key, api_base=self.api_base, model=self.model, size=self.size,
            quality=self.quality, style=self.style, gen_timeout=self.gen_timeout
        )
        
        # Validate model and size combination
        if self.model == "dall-e-2":
            valid_sizes = ["256x256", "512x512", "1024x1024"]
            if self.size not in valid_sizes:
                log(f"Warning: Size {self.size} not valid for DALL-E 2. Using 1024x1024")
                self.size = "1024x1024"
        elif self.model == "dall-e-3":
            valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
            if self.size not in valid_sizes:
                log(f"Warning: Size {self.size} not valid for DALL-E 3. Using 1024x1024")
                self.size = "1024x1024"

    def get_provider_name(self) -> str:
        """Get the name of the image generation provider."""
        return "OpenAI"

    def test_connection(self) -> bool:
        """Test the connection to the OpenAI API."""
        debug(f"OpenAI ImageGenerator params: {self.__dict__}")
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            # Test with a simple models request
            response = requests.get(f"{self.api_base}/models", headers=headers, timeout=10)
            debug(f"OpenAI ImageGenerator response: {response.status_code}")
            # Accept both 200 (success) and 401 (unauthorized) as valid responses
            # 401 means the endpoint is reachable but the API key might be invalid
            # We'll let the actual image generation fail gracefully if the key is bad
            return response.status_code in [200, 401]
        except Exception as e:
            log(f"OpenAI API server ({self.api_base}) not reachable: {e}")
            return False

    async def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generate an image using OpenAI's image generation API.
        
        Args:
            prompt (str): Text description of the image to generate
            **kwargs: Optional parameters including model, size, quality, style, n
            
        Returns:
            Optional[str]: Base64 encoded image data if successful, None otherwise
        """
        # Extract parameters from kwargs with defaults
        model = kwargs.get('model', self.model)
        size = kwargs.get('size', self.size)
        quality = kwargs.get('quality', self.quality)
        style = kwargs.get('style', self.style)
        n = kwargs.get('n', 1)
        
        # Validate n parameter based on model
        if model == "dall-e-3" and n > 1:
            log("DALL-E 3 only supports generating 1 image at a time. Setting n=1")
            n = 1
        elif model == "dall-e-2" and n > 10:
            log("DALL-E 2 supports maximum 10 images. Setting n=10")
            n = 10

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": "b64_json"  # Get base64 encoded images
        }

        # Add DALL-E 3 specific parameters
        if model == "dall-e-3":
            data["quality"] = quality
            data["style"] = style

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/images/generations",
                    headers=headers,
                    json=data,
                    timeout=self.gen_timeout
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        # Return the first image's base64 data
                        if response_data.get("data") and len(response_data["data"]) > 0:
                            return response_data["data"][0]["b64_json"]
                    else:
                        error_text = await response.text()
                        log(f"OpenAI API error {response.status}: {error_text}")
                        return None

        except Exception as e:
            log(f"OpenAI API server ({self.api_base}) error: {e}")
            return None

# Test
if __name__ == "__main__":
    import asyncio

    async def main():
        # Test with environment variables
        image_gen = OpenAIImageGenerator()
        
        async def test_image_gen():
            image_encoded = await image_gen.generate(prompt="A beautiful sunset over the mountains")
            if image_encoded:
                image = Image.open(io.BytesIO(base64.b64decode(image_encoded)))
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
