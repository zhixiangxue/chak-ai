"""
MiniMax Provider - Anthropic API Compatible (Token Plan)

MiniMax provides MiniMax M series models through an Anthropic-compatible API.
Official documentation: https://platform.minimaxi.com/docs/token-plan/quickstart

Supported models:
- MiniMax-M3: Latest MiniMax model via Token Plan
"""
from typing import Optional, Dict, Any

from pydantic import field_validator

from .anthropic_compat import AnthropicCompatibleProvider, AnthropicCompatibleMessageConverter
from .base import BaseProviderConfig
from ...metadata import Metadata


class MiniMaxConfig(BaseProviderConfig):
    """MiniMax-specific configuration."""
    base_url: Optional[str] = "https://api.minimaxi.com/anthropic"

    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for MiniMax."""
        return v or "https://api.minimaxi.com/anthropic"


class MiniMaxMessageConverter(AnthropicCompatibleMessageConverter):
    """Converter for MiniMax message formats."""

    def _build_metadata(self, response: Any) -> Metadata:
        """Build metadata with 'minimax' as provider name."""
        metadata = super()._build_metadata(response)
        metadata.provider = "minimax"
        return metadata

    def _usage_to_metadata(self, usage: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'minimax' as provider name."""
        metadata = super()._usage_to_metadata(usage)
        metadata["provider"] = "minimax"
        return metadata


class MiniMaxProvider(AnthropicCompatibleProvider):
    """MiniMax provider implementation using Anthropic-compatible API."""

    def __init__(self, config: MiniMaxConfig, converter: MiniMaxMessageConverter = None):
        super().__init__(config, converter or MiniMaxMessageConverter())
