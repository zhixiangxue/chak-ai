from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Iterator, Union

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ... import __version__
from ...exceptions import ProviderError, ErrorType
from ...message import Message, UnifiedStreamChunk
from ...metadata import Metadata, Usage, ProviderTrace


class BaseProviderConfig(BaseModel):
    """Base configuration for all providers using Pydantic."""
    api_key: str
    model: str
    base_url: Optional[str] = None
    timeout: int = 120  # Increased from 30s to 120s for structured output with large prompts
    max_retries: int = 3
    provider_name: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")  # Allow extra fields (e.g. temperature)

    @field_validator('api_key')
    @classmethod
    def api_key_non_empty(cls, v):
        if not v:
            raise ValueError("API key cannot be empty")
        return v
    
    @field_validator('model')
    @classmethod
    def model_non_empty(cls, v):
        if not v:
            raise ValueError("Model cannot be empty")
        return v


class BaseMessageConverter(ABC):
    """Base class for message format conversion."""

    @abstractmethod
    def to_provider_format(self, messages: List[Message]) -> Any:
        """Convert standard messages to provider-specific format."""
        pass

    @abstractmethod
    def from_provider_response(self, response: Any) -> Message:
        """Convert provider response to standard Message."""
        pass

    @abstractmethod
    def from_provider_chunk(self, chunk: Any) -> UnifiedStreamChunk:
        """Convert provider streaming chunk to UnifiedStreamChunk."""
        pass


class Provider(ABC):
    """Base provider class with simplified design."""

    def __init__(self, config: BaseProviderConfig, converter: BaseMessageConverter):
        self.config = config
        self.converter = converter
        self._client = None
        self._initialize_client()

    def _create_http_client(self) -> httpx.Client:
        """Create HTTP client with Chak User-Agent header."""
        return httpx.Client(
            headers={"User-Agent": f"Chak/{__version__}"}
        )

    @abstractmethod
    def _initialize_client(self):
        """Initialize the provider-specific client."""
        pass

    @property
    def provider_name(self) -> str:
        """Canonical provider name from config, falling back to class name."""
        return str(getattr(self.config, "provider_name", None) or self.__class__.__name__)

    @property
    def model_name(self) -> str:
        """Model name from config."""
        return str(getattr(self.config, "model", "") or "")

    def send(
            self,
            messages: List[Message],
            stream: bool = False,
            **kwargs
    ):
        """Unified send method for both streaming and non-streaming."""
        try:
            provider_messages = self.converter.to_provider_format(messages)

            if stream:
                return self._wrap_stream_errors(self._send_stream(provider_messages, **kwargs))
            else:
                response = self._send_complete(provider_messages, **kwargs)
                result = self.converter.from_provider_response(response)
                self._ensure_provider_trace(result)
                return result

        except Exception as e:
            raise self._normalize_error(e) from e

    def _ensure_provider_trace(self, message: Any) -> None:
        """Set a default ProviderTrace on the message metadata if not already set.

        Non-resilient providers produce messages without a trace; this ensures
        every message has one so developers never need to check for None.
        ResilientProvider sets its own trace via _annotate_message, so this
        no-ops when a trace is already present.
        """
        metadata = getattr(message, "metadata", None)
        if not isinstance(metadata, Metadata):
            return
        if metadata.provider_trace is not None:
            return  # Already set by ResilientProvider
        metadata.provider_trace = ProviderTrace(
            primary_provider=self.provider_name,
            primary_model=self.model_name,
            fallback_used=False,
            failover_attempts=0,
            failed_providers=[],
            resolved_provider=self.provider_name,
            resolved_model=self.model_name,
        )

    def _normalize_error(self, error: BaseException) -> ProviderError:
        """Convert a raw exception into ProviderError.

        Subclasses SHOULD override this to precisely map their SDK's
        exception types.  The base implementation is a conservative
        fallback that marks everything as 'unknown' (not retryable).
        """
        if isinstance(error, ProviderError):
            error.provider = error.provider or self.provider_name
            error.model = error.model or self.config.model
            error.base_url = error.base_url or getattr(self.config, "base_url", None)
            return error
        return ProviderError(
            f"{self.__class__.__name__} error: {error}",
            provider=self.provider_name,
            model=self.config.model,
            base_url=getattr(self.config, "base_url", None),
            status_code=None,
            error_type=ErrorType.UNKNOWN,
            raw_error=error,
        )

    def _wrap_stream_errors(self, stream: Iterator[Any]) -> Iterator[Any]:
        try:
            for chunk in stream:
                yield chunk
        except Exception as e:
            raise self._normalize_error(e) from e

    @abstractmethod
    def _send_complete(self, messages: Any, **kwargs) -> Any:
        """Send non-streaming request."""
        pass

    @abstractmethod
    def _send_stream(self, messages: Any, **kwargs) -> Iterator[Any]:
        """Send streaming request."""
        pass

    def close(self):
        """Clean up resources."""
        if self._client and hasattr(self._client, 'close'):
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
