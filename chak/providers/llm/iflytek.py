from typing import Optional, Dict, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class IFlyTekConfig(BaseProviderConfig):
    """iFlytek (讯飞) specific configuration."""
    base_url: Optional[str] = "http://maas-api.cn-huabei-1.xf-yun.com/v1"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for iFlytek."""
        return v or "http://maas-api.cn-huabei-1.xf-yun.com/v1"


class IFlyTekMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for iFlytek message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'iflytek' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "iflytek"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'iflytek' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "iflytek"
        return metadata


class IFlyTekProvider(OpenAICompatibleProvider):
    """iFlytek provider implementation."""
    pass  # Uses base implementation
