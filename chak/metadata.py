"""Message metadata types

This module defines metadata structures for messages.
Separated from providers to avoid circular imports.
"""

from typing import Any, Dict, List, Optional

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


class FailureRecord(BaseModel):
    """A single failed provider attempt within a request."""

    attempt_index: int = 0
    provider: str = ""
    model: str = ""
    base_url: str = ""
    error: str = ""
    status_code: Optional[int] = None
    error_type: Optional[str] = None


class ProviderTrace(BaseModel):
    """Provider routing trace for a single model request.

    Records which providers were attempted, which failed, and which
    ultimately resolved the request. Always populated — even for direct
    (non-resilient) calls — so developers never need to check for None.
    """

    primary_provider: str = ""
    primary_model: str = ""
    fallback_used: bool = False
    failover_attempts: int = 0
    failed_providers: List[FailureRecord] = Field(default_factory=list)
    resolved_provider: str = ""
    resolved_model: str = ""


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
    provider_trace: Optional[ProviderTrace] = None
