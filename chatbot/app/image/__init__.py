#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image generation module for TinyLLM Chatbot.

This module provides factory functions and imports for image generators,
supporting multiple image generation backends (SwarmUI, OpenAI, etc.).

Test: python3 app/image/__init__.py   

Author: Jason A. Cox
2 Aug 2025
github.com/jasonacox/TinyLLM
"""
import io
import base64
from typing import Optional

from PIL import Image
from .base import BaseImageGenerator
from .swarmui_generator import SwarmUIImageGenerator
from .openai_generator import OpenAIImageGenerator

if __name__ == "__main__":
    def log(text):
        print(text)
    def debug(text):
        print(text)
else:
    from app.core.config import (log, debug)

def create_image_generator(provider: str = "swarmui", **kwargs) -> BaseImageGenerator:
    """
    Factory function to create the appropriate image generator based on provider.
    
    Args:
        provider (str): The image generation provider ("swarmui" or "openai")
        **kwargs: Configuration parameters for the specific provider
        
    Returns:
        BaseImageGenerator: An instance of the appropriate image generator
        
    Raises:
        ValueError: If the provider is not supported
    """
    provider = provider.lower()
    
    if provider == "swarmui":
        return SwarmUIImageGenerator(**kwargs)
    elif provider == "openai":
        return OpenAIImageGenerator(**kwargs)
    else:
        raise ValueError(f"Unsupported image generation provider: {provider}")

# Legacy alias for backward compatibility
ImageGenerator = SwarmUIImageGenerator

# Test
if __name__ == "__main__":
    import asyncio

    async def main():
        image_gen = SwarmUIImageGenerator(host="http://10.0.1.25:7801")
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
