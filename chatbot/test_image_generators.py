#!/usr/bin/env python3
"""
Test script for the image generation system.

This script tests both SwarmUI and OpenAI image generators to ensure
they are properly configured and can be instantiated.
"""

import sys
import os

# Add the chatbot app to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_swarmui_generator():
    """Test SwarmUI image generator."""
    print("Testing SwarmUI Image Generator...")
    try:
        from app.image import create_image_generator
        
        generator = create_image_generator(
            provider="swarmui",
            host="http://localhost:7801",
            model="OfficialStableDiffusion/sd_xl_base_1.0",
            width=1024,
            height=1024
        )
        
        print(f"✓ SwarmUI generator created: {generator.get_provider_name()}")
        print(f"  Host: {generator.host}")
        print(f"  Model: {generator.model}")
        
        # Test connection (will likely fail unless SwarmUI is running)
        if generator.test_connection():
            print("✓ SwarmUI connection successful")
        else:
            print("⚠ SwarmUI connection failed (server may not be running)")
            
    except Exception as e:
        print(f"✗ SwarmUI generator test failed: {e}")

def test_openai_generator():
    """Test OpenAI image generator."""
    print("\nTesting OpenAI Image Generator...")
    try:
        from app.image import create_image_generator
        
        generator = create_image_generator(
            provider="openai",
            api_key=os.environ.get("OPENAI_API_KEY", "fake-key-for-testing"),  # Use env var or clearly fake key
            api_base="https://api.openai.com/v1",  # Ensure we test against real OpenAI
            model="dall-e-3",
            size="1024x1024"
        )
        
        print(f"✓ OpenAI generator created: {generator.get_provider_name()}")
        print(f"  Model: {generator.model}")
        print(f"  Size: {generator.size}")
        
        # Test connection (will fail with test key)
        if generator.test_connection():
            print("✓ OpenAI connection successful")
        else:
            print("⚠ OpenAI connection failed (expected with test key or local server)")
            
    except Exception as e:
        print(f"✗ OpenAI generator test failed: {e}")

def test_local_llm_compatibility():
    """Test that OpenAI generator fails gracefully with local LLM servers."""
    print("\nTesting OpenAI generator with local LLM server...")
    try:
        from app.image import create_image_generator
        
        # Test with typical local LLM server endpoint
        generator = create_image_generator(
            provider="openai",
            api_key=os.environ.get("OPENAI_API_KEY", "sk-test-fake-key"),  # Use env var or obviously fake key
            api_base="http://localhost:8000/v1",  # Typical local LLM server
            model="dall-e-3",
            size="1024x1024"
        )
        
        print("✓ OpenAI generator created for local server test")
        
        # This should fail gracefully
        if generator.test_connection():
            print("✓ Local server supports OpenAI image models")
        else:
            print("⚠ Local server doesn't support OpenAI image models (expected)")
            
    except Exception as e:
        print(f"⚠ Local LLM compatibility test: {e}")

def test_factory_function():
    """Test the factory function error handling."""
    print("\nTesting factory function error handling...")
    try:
        from app.image import create_image_generator
        
        # Test invalid provider
        try:
            generator = create_image_generator(provider="invalid")
            print("✗ Factory function should have raised ValueError")
        except ValueError as e:
            print(f"✓ Factory function correctly raised ValueError: {e}")
            
    except Exception as e:
        print(f"✗ Factory function test failed: {e}")

if __name__ == "__main__":
    print("Image Generation System Test")
    print("=" * 40)
    
    test_swarmui_generator()
    test_openai_generator()
    test_local_llm_compatibility()
    test_factory_function()
    
    print("\n" + "=" * 40)
    print("Test completed. Check results above.")
    print("\nTo use in production:")
    print("- For SwarmUI: Set IMAGE_PROVIDER=swarmui and configure SWARMUI URL")
    print("- For OpenAI: Set IMAGE_PROVIDER=openai and provide OPENAI_API_KEY")
    print("\nNote: OpenAI image generation requires a real OpenAI API endpoint,")
    print("      not a local LLM server with OpenAI-compatible API.")
