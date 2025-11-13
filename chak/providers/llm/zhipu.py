from typing import Optional, Dict, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class ZhipuConfig(BaseProviderConfig):
    """Zhipu AI-specific configuration."""
    base_url: Optional[str] = "https://open.bigmodel.cn/api/paas/v4/"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for Zhipu AI."""
        return v or "https://open.bigmodel.cn/api/paas/v4/"


class ZhipuMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for Zhipu AI message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'zhipu' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "zhipu"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'zhipu' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "zhipu"
        return metadata


class ZhipuProvider(OpenAICompatibleProvider):
    """Zhipu AI provider implementation."""
    pass  # Uses base implementation
