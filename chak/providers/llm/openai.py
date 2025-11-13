"""
OpenAI Provider

Official OpenAI API provider.
Official documentation: https://platform.openai.com/docs/api-reference

Supported models:
- GPT-4 series: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini
- GPT-3.5 series: gpt-3.5-turbo
- O1 series: o1, o1-mini, o1-preview
"""
from typing import Optional

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class OpenAIConfig(BaseProviderConfig):
    """Configuration for OpenAI provider."""
    base_url: Optional[str] = "https://api.openai.com/v1"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for OpenAI."""
        return v or "https://api.openai.com/v1"


class OpenAIMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for OpenAI message formats."""
    pass  # Uses base implementation with 'openai' as provider name


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI provider implementation."""
    pass  # Uses base implementation
