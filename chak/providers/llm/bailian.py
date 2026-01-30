from typing import Optional, Dict, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


class BailianConfig(BaseProviderConfig):
    """Bailian-specific configuration."""
    base_url: Optional[str] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    organization: Optional[str] = None
    project: Optional[str] = None
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for Bailian (DashScope compatible mode)."""
        return v or "https://dashscope.aliyuncs.com/compatible-mode/v1"


class BailianMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for Bailian message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'bailian' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "bailian"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'bailian' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "bailian"
        return metadata


class BailianProvider(OpenAICompatibleProvider):
    """Bailian provider implementation."""
    
    def _extend_client_kwargs(self, kwargs: dict):
        """Add Bailian-specific organization parameter if exists."""
        if isinstance(self.config, BailianConfig) and self.config.organization:
            kwargs["organization"] = self.config.organization
    
    def _apply_reasoning_params(self, kwargs: dict) -> None:
        """Transform chak's unified reasoning params to Bailian's format.
        
        Bailian uses extra_body with:
        - enable_thinking: True (to enable reasoning mode)
        - thinking_budget: int (optional, maps to chak's 'budget')
        """
        reasoning = kwargs.pop('reasoning', None)
        if not reasoning:
            return
        
        # Initialize extra_body if not exists
        if 'extra_body' not in kwargs:
            kwargs['extra_body'] = {}
        
        # Enable thinking mode (presence of reasoning dict means enable)
        kwargs['extra_body']['enable_thinking'] = True
        
        # Map chak's 'budget' to Bailian's 'thinking_budget'
        if 'budget' in reasoning:
            kwargs['extra_body']['thinking_budget'] = reasoning['budget']