"""Message metadata types

This module defines metadata structures for messages.
Separated from providers to avoid circular imports.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Normalized token usage information for provider responses.

    Field names follow OpenAI's official naming convention.
    cache_creation_input_tokens and cache_read_input_tokens are Anthropic
    prompt-caching fields; other providers may populate them too.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_creation_input_tokens: int = 0  # Tokens written to cache (Anthropic)
    cache_read_input_tokens: int = 0      # Tokens read from cache (Anthropic)


class Metadata(BaseModel):
    """Standardized metadata for provider responses.

    This structure is used by BaseMessage.metadata and provides a stable
    contract across all providers.
    """

    provider: str = ""
    model: Optional[str] = None
    usage: Optional[Usage] = None
    finish_reason: Optional[str] = None
    request_id: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
