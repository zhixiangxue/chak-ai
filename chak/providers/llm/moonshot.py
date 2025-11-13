from typing import Optional, Dict, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class MoonshotConfig(BaseProviderConfig):
    """Moonshot-specific configuration."""
    base_url: Optional[str] = "https://api.moonshot.cn/v1"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for Moonshot."""
        return v or "https://api.moonshot.cn/v1"


class MoonshotMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for Moonshot message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'moonshot' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "moonshot"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'moonshot' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "moonshot"
        return metadata


class MoonshotProvider(OpenAICompatibleProvider):
    """Moonshot provider implementation."""
    pass  # Uses base implementation
