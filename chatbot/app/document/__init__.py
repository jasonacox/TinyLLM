"""
Document Generation Package for TinyLLM Chatbot

This package provides document generation capabilities for the chatbot.
"""

from .document_generator import DocumentGenerator, generate_document_from_response, get_document_generator

__all__ = ['DocumentGenerator', 'generate_document_from_response', 'get_document_generator']
