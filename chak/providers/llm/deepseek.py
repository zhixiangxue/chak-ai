from typing import Optional, Dict, Any, List

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider
from ...message import Message
from ...metadata import Metadata


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

    def to_provider_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Extend base format to include reasoning_content in assistant messages.

        DeepSeek thinking mode requires that reasoning_content produced in a
        previous response is echoed back verbatim in the corresponding
        assistant message of the next request.
        """
        result = super().to_provider_format(messages)
        for msg, formatted in zip(messages, result):
            if formatted.get("role") == "assistant":
                rc = getattr(msg, "reasoning_content", None)
                if rc:
                    formatted["reasoning_content"] = rc
        return result

    def _build_metadata(self, response: Any, choice: Any) -> Metadata:
        """Build metadata with 'deepseek' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata.provider = "deepseek"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'deepseek' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "deepseek"
        return metadata


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider implementation."""
    pass  # Uses base implementation
