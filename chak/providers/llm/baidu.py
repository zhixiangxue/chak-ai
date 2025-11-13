from typing import Optional, Dict, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class BaiduConfig(BaseProviderConfig):
    """Baidu Qianfan-specific configuration."""
    base_url: Optional[str] = "https://qianfan.baidubce.com/v2"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for Baidu Qianfan."""
        return v or "https://qianfan.baidubce.com/v2"


class BaiduMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for Baidu message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'baidu' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "baidu"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'baidu' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "baidu"
        return metadata


class BaiduProvider(OpenAICompatibleProvider):
    """Baidu provider implementation."""
    pass  # Uses base implementation
