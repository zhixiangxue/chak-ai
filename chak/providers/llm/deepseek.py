from typing import Optional, Dict, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class DeepSeekConfig(BaseProviderConfig):
    """DeepSeek-specific configuration."""
    base_url: Optional[str] = "https://api.deepseek.com"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for DeepSeek."""
        return v or "https://api.deepseek.com"


class DeepSeekMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for DeepSeek message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'deepseek' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "deepseek"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'deepseek' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "deepseek"
        return metadata


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider implementation."""
    pass  # Uses base implementation
