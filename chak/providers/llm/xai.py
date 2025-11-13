"""
xAI (Grok) Provider - OpenAI API Compatible

xAI provides Grok models through an OpenAI-compatible API.
Official documentation: https://docs.x.ai/

Supported models:
- grok-4: Latest Grok model
- grok-vision-beta: Grok model with vision capabilities
"""
from typing import Optional, Dict, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class XAIConfig(BaseProviderConfig):
    """Configuration for xAI provider."""
    base_url: Optional[str] = "https://api.x.ai/v1"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for xAI."""
        return v or "https://api.x.ai/v1"
    
    @field_validator('timeout', mode='before')
    @classmethod
    def set_default_timeout(cls, v):
        """Set default timeout for reasoning models (3600s)."""
        return v if v is not None else 3600


class XAIMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for xAI message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'xai' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "xai"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'xai' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "xai"
        return metadata


class XAIProvider(OpenAICompatibleProvider):
    """xAI provider implementation."""
    pass  # Uses base implementation
