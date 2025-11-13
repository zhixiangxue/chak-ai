"""
Google AI Provider - OpenAI API Compatible

Google provides Gemini models through an OpenAI-compatible API.
Official documentation: https://ai.google.dev/gemini-api/docs

Supported models:
- Gemini 2.0: gemini-2.0-flash-exp
- Gemini 1.5: gemini-1.5-pro, gemini-1.5-flash
- Gemini Pro: gemini-pro
"""
from typing import Optional, Dict, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class GoogleConfig(BaseProviderConfig):
    """Configuration for Google AI provider."""
    base_url: Optional[str] = "https://generativelanguage.googleapis.com/v1beta/openai/"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for Google AI."""
        return v or "https://generativelanguage.googleapis.com/v1beta/openai/"


class GoogleMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for Google AI message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'google' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "google"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'google' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "google"
        return metadata


class GoogleProvider(OpenAICompatibleProvider):
    """Google AI provider implementation."""
    pass  # Uses base implementation
