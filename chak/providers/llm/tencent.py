from typing import Optional, Dict, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class TencentConfig(BaseProviderConfig):
    """Tencent Cloud-specific configuration."""
    base_url: Optional[str] = "https://api.hunyuan.cloud.tencent.com/v1"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for Tencent Cloud."""
        return v or "https://api.hunyuan.cloud.tencent.com/v1"


class TencentMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for Tencent Cloud message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'tencent' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "tencent"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'tencent' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "tencent"
        return metadata


class TencentProvider(OpenAICompatibleProvider):
    """Tencent Cloud provider implementation."""
    pass  # Uses base implementation
