# src/chak/providers/llm/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Iterator

import httpx
from pydantic import BaseModel, Field, field_validator

from ... import __version__
from ...exceptions import ProviderError
from ...message import Message, MessageChunk


class BaseProviderConfig(BaseModel):
    """Base configuration for all providers using Pydantic."""
    api_key: str
    model: str
    base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    headers: Dict[str, str] = Field(default_factory=dict)

    class Config:
        extra = "allow"  # 允许额外字段（如temperature等）

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
    def from_provider_chunk(self, chunk: Any) -> MessageChunk:
        """Convert provider streaming chunk to standard MessageChunk."""
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
                return self._send_stream(provider_messages, **kwargs)
            else:
                response = self._send_complete(provider_messages, **kwargs)
                return self.converter.from_provider_response(response)

        except Exception as e:
            raise ProviderError(f"{self.__class__.__name__} error: {e}") from e

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