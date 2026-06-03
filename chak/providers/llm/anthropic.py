"""
Anthropic Claude Provider - Native SDK

Uses the official Anthropic SDK for full feature support including:
- Prompt caching (cache_control: {"type": "ephemeral"})
- Extended thinking (thinking parameter)
- Native tool use (input_schema instead of OpenAI parameters)
- Streaming with proper event handling

Official documentation: https://docs.anthropic.com/

Supported models:
- Claude 4: claude-haiku-4-5
- Claude 3.7: claude-3-7-sonnet-20250219
- Claude 3.5: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
- Claude 3: claude-3-opus-20240229, claude-3-haiku-20240307
"""
from typing import Optional

from pydantic import field_validator

from .anthropic_compat import AnthropicCompatibleProvider, AnthropicCompatibleMessageConverter
from .base import BaseProviderConfig
from ...schemas import Cache


class AnthropicConfig(BaseProviderConfig):
    """Configuration for Anthropic native SDK provider."""
    base_url: Optional[str] = None  # SDK handles endpoint internally
    cache: Optional[Cache] = None  # Prompt caching settings

    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        return v or None


class AnthropicMessageConverter(AnthropicCompatibleMessageConverter):
    """Converter for Anthropic native Messages API format.

    Inherits all conversion logic from AnthropicCompatibleMessageConverter.
    _build_metadata already sets provider="anthropic" by default.
    """
    pass


class AnthropicProvider(AnthropicCompatibleProvider):
    """Anthropic Claude provider using the official Anthropic SDK.

    Supports:
    - Non-streaming / streaming chat
    - Tool use (converted from OpenAI format at call time)
    - Extended thinking (via 'reasoning' param)
    - Prompt caching (pass cache_control in message content blocks)

    Cache support is activated automatically when AnthropicConfig.cache is set.
    """

    def __init__(self, config: AnthropicConfig, converter: AnthropicMessageConverter = None):
        super().__init__(config, converter or AnthropicMessageConverter())
