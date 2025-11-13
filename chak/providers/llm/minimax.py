from typing import Optional, Dict, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class MiniMaxConfig(BaseProviderConfig):
    """MiniMax-specific configuration."""
    base_url: Optional[str] = "https://api.minimaxi.com/v1"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for MiniMax."""
        return v or "https://api.minimaxi.com/v1"


class MiniMaxMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for MiniMax message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'minimax' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "minimax"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'minimax' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "minimax"
        return metadata


class MiniMaxProvider(OpenAICompatibleProvider):
    """MiniMax provider implementation."""
    pass  # Uses base implementation
