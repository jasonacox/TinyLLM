"""
ChatBot - Custom Exceptions

This module defines custom exceptions for the TinyLLM Chatbot.
"""
# pylint: disable=unnecessary-pass

class ChatbotError(Exception):
    """Base exception for all chatbot errors"""
    pass

class ServerError(ChatbotError):
    """Exception for server-related errors"""
    pass

class ConfigurationError(ChatbotError):
    """Exception for configuration-related errors"""
    pass

class LLMError(ChatbotError):
    """Exception for LLM-related errors"""
    pass

class RAGError(ChatbotError):
    """Exception for RAG-related errors"""
    pass

class AuthenticationError(ChatbotError):
    """Exception for authentication-related errors"""
    pass

class FileError(ChatbotError):
    """Exception for file-related errors"""
    pass
