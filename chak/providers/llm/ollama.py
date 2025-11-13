"""
Ollama Provider - OpenAI API Compatible

Ollama provides local LLM inference with an OpenAI-compatible API.
Official documentation: https://ollama.com/blog/openai-compatibility

Supported models:
- All models available in Ollama library (llama2, mistral, codellama, etc.)
"""
from typing import Optional, Dict, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class OllamaConfig(BaseProviderConfig):
    """Configuration for Ollama provider."""
    base_url: Optional[str] = "http://localhost:11434/v1"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for Ollama (local server)."""
        return v or "http://localhost:11434/v1"
    
    @field_validator('api_key', mode='before')
    @classmethod
    def set_default_api_key(cls, v):
        """Set default API key for Ollama (required but unused)."""
        return v or "ollama"


class OllamaMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for Ollama message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'ollama' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "ollama"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'ollama' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "ollama"
        return metadata


class OllamaProvider(OpenAICompatibleProvider):
    """Ollama provider implementation."""
    pass  # Uses base implementation
