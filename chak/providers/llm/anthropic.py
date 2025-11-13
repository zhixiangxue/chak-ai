"""
Anthropic Claude Provider - OpenAI API Compatible

Anthropic provides Claude models through an OpenAI-compatible API.
Official documentation: https://docs.anthropic.com/

Supported models:
- Claude 3.5 series: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
- Claude 3 series: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
- Latest: claude-sonnet-4-5
"""
from typing import Optional, Dict, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class AnthropicConfig(BaseProviderConfig):
    """Configuration for Anthropic provider."""
    base_url: Optional[str] = "https://api.anthropic.com/v1/"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for Anthropic."""
        return v or "https://api.anthropic.com/v1/"


class AnthropicMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for Anthropic message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'anthropic' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "anthropic"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'anthropic' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "anthropic"
        return metadata


class AnthropicProvider(OpenAICompatibleProvider):
    """Anthropic Claude provider implementation."""
    pass  # Uses base implementation
