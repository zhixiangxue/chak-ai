"""
vLLM Provider - OpenAI API Compatible

vLLM is a high-performance LLM inference engine with OpenAI-compatible API.
Official documentation: https://docs.vllm.ai/en/latest/

Supported models:
- All models supported by vLLM (depends on your deployment)
- Common: llama, mistral, qwen, yi, etc.
"""
from typing import Optional, Dict, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class VLLMConfig(BaseProviderConfig):
    """Configuration for vLLM provider."""
    base_url: Optional[str] = "http://localhost:8000/v1"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for vLLM (local server)."""
        return v or "http://localhost:8000/v1"
    
    @field_validator('api_key', mode='before')
    @classmethod
    def set_default_api_key(cls, v):
        """Set default API key for vLLM (required but often unused in local deployment)."""
        return v or "EMPTY"


class VLLMMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for vLLM message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'vllm' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "vllm"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'vllm' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "vllm"
        return metadata


class VLLMProvider(OpenAICompatibleProvider):
    """vLLM provider implementation."""
    pass  # Uses base implementation
