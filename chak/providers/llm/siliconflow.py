from typing import Optional, Dict, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class SiliconFlowConfig(BaseProviderConfig):
    """SiliconFlow-specific configuration."""
    base_url: Optional[str] = "https://api.siliconflow.cn/v1"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for SiliconFlow."""
        return v or "https://api.siliconflow.cn/v1"


class SiliconFlowMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for SiliconFlow message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'siliconflow' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "siliconflow"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'siliconflow' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "siliconflow"
        return metadata


class SiliconFlowProvider(OpenAICompatibleProvider):
    """SiliconFlow provider implementation."""
    pass  # Uses base implementation
