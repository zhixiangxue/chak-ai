from typing import Optional, Dict, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class VolcEngineConfig(BaseProviderConfig):
    """VolcEngine-specific configuration."""
    base_url: Optional[str] = "https://ark.cn-beijing.volces.com/api/v3"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for VolcEngine."""
        return v or "https://ark.cn-beijing.volces.com/api/v3"


class VolcEngineMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for VolcEngine message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'volcengine' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "volcengine"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'volcengine' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "volcengine"
        return metadata


class VolcEngineProvider(OpenAICompatibleProvider):
    """VolcEngine provider implementation."""
    pass  # Uses base implementation
