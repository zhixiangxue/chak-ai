# src/chak/providers/llm/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Iterator

import httpx
import openai
from pydantic import BaseModel, Field, field_validator

from ... import __version__
from ...exceptions import ProviderError
from ...message import Message, MessageChunk, AIMessage, ChatCompletionMessageToolCall, Function


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


# ========== OpenAI Compatible Base Classes ==========

class OpenAICompatibleMessageConverter(BaseMessageConverter):
    """OpenAI SDK compatible message converter base class."""
    
    def to_provider_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert to OpenAI-compatible message format with tool support."""
        result = []
        for msg in messages:
            # Basic message structure
            formatted_msg: Dict[str, Any] = {
                "role": msg.role or "user",
                "content": msg.content or ""
            }
            
            # Add tool_calls if present (for assistant messages)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # Convert tool_calls to dict format
                formatted_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls  # type: ignore
                ]
            
            # Add tool_call_id if present (for tool messages)
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:  # type: ignore
                formatted_msg["tool_call_id"] = msg.tool_call_id  # type: ignore
            
            result.append(formatted_msg)
        
        return result
    
    def from_provider_response(self, response: Any) -> AIMessage:
        """Convert OpenAI-compatible response to standard AIMessage."""
        choice = response.choices[0]
        message = choice.message
        
        # Extract and convert tool_calls if present
        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # Convert OpenAI tool_calls to our format
            tool_calls = [
                ChatCompletionMessageToolCall(
                    id=tc.id,
                    type="function",
                    function=Function(
                        name=tc.function.name,
                        arguments=tc.function.arguments
                    )
                )
                for tc in message.tool_calls
            ]
        
        return AIMessage(
            content=message.content or "",
            tool_calls=tool_calls,
            metadata=self._build_metadata(response, choice)
        )
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata - subclasses can override to change provider name."""
        return {
            "provider": "openai",  # Subclass should override this
            "model": response.model,
            "usage": getattr(response, 'usage', {}),
            "finish_reason": choice.finish_reason,
        }
    
    def from_provider_chunk(self, chunk: Any) -> MessageChunk:
        """Convert OpenAI-compatible streaming chunk to standard MessageChunk."""
        choice = chunk.choices[0] if chunk.choices else None
        delta = choice.delta if choice else None
        
        content = delta.content if delta and delta.content else ""
        is_final = bool(choice and choice.finish_reason is not None)
        
        return MessageChunk(
            content=content,
            is_final=is_final,
            metadata=self._build_chunk_metadata(chunk, choice)
        )
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata - subclasses can override."""
        return {
            "provider": "openai",
            "model": chunk.model,
            "finish_reason": choice.finish_reason if choice else None
        }


class OpenAICompatibleProvider(Provider):
    """OpenAI SDK compatible provider base class."""
    
    def _initialize_client(self):
        """Initialize OpenAI-compatible client."""
        client_kwargs = {
            "api_key": self.config.api_key,
            "base_url": self.config.base_url,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
            "http_client": self._create_http_client(),
        }
        
        # Allow subclass to extend with additional parameters
        self._extend_client_kwargs(client_kwargs)
        
        self._client = openai.OpenAI(**client_kwargs)
    
    def _extend_client_kwargs(self, kwargs: dict):
        """Hook method: subclasses can override to add extra client parameters."""
        pass
    
    def _send_complete(self, messages: List, **kwargs) -> Any:
        """Send non-streaming request to OpenAI-compatible API."""
        return self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            **kwargs
        )
    
    def _send_stream(self, messages: List, **kwargs) -> Iterator[Any]:
        """Send streaming request to OpenAI-compatible API."""
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            yield chunk