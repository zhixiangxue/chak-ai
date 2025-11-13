"""
Mistral AI Provider - OpenAI API Compatible

Mistral AI provides powerful open-source models through an OpenAI-compatible API.
Official documentation: https://docs.mistral.ai/

Supported models:
- Mistral Large: mistral-large-latest, mistral-large-2411
- Mistral Medium: mistral-medium-latest
- Mistral Small: mistral-small-latest
- Codestral: codestral-latest (code generation)
- Pixtral: pixtral-large-latest (vision)
"""
from typing import Optional, Dict, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class MistralConfig(BaseProviderConfig):
    """Configuration for Mistral AI provider."""
    base_url: Optional[str] = "https://api.mistral.ai/v1"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for Mistral AI."""
        return v or "https://api.mistral.ai/v1"


class MistralMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for Mistral AI message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'mistral' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "mistral"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'mistral' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "mistral"
        return metadata


class MistralProvider(OpenAICompatibleProvider):
    """Mistral AI provider implementation."""
    pass  # Uses base implementation
