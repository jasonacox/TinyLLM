#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base ImageGenerator abstract class for generating images.

This module defines the interface that all image generators must implement,
allowing for multiple image generation backends (SwarmUI, OpenAI, etc.).

Author: Jason A. Cox
2 Aug 2025
github.com/jasonacox/TinyLLM
"""

from abc import ABC, abstractmethod
from typing import Optional

class BaseImageGenerator(ABC):
    """Abstract base class for image generators."""
    
    def __init__(self, **kwargs):
        """Initialize the image generator with configuration parameters."""
        self.config = kwargs
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test the connection to the image generation service.
        
        Returns:
            bool: True if connection is successful, False otherwise.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate an image based on the given prompt.
        
        Args:
            prompt (str): The text prompt describing the image to generate.
            **kwargs: Additional parameters specific to the implementation.
            
        Returns:
            Optional[str]: Base64 encoded image data if successful, None otherwise.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of the image generation provider.
        
        Returns:
            str: Name of the provider (e.g., "SwarmUI", "OpenAI").
        """
        raise NotImplementedError
