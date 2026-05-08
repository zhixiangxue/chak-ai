from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Iterator, Union

import httpx
import openai
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ... import __version__
from ...exceptions import ProviderError
from ...message import Message, MessageChunk, ReasoningChunk, AIMessage, ChatCompletionMessageToolCall, Function, UnifiedStreamChunk
from ...metadata import Metadata, Usage


class BaseProviderConfig(BaseModel):
    """Base configuration for all providers using Pydantic."""
    api_key: str
    model: str
    base_url: Optional[str] = None
    timeout: int = 120  # Increased from 30s to 120s for structured output with large prompts
    max_retries: int = 3
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
        """Convert to OpenAI-compatible message format with multimodal support."""
        result = []
        
        for msg in messages:
            # Build basic message structure
            formatted_msg: Dict[str, Any] = {
                "role": msg.role or "user",
                "content": msg.content or ""
            }
            
            # Add tool_calls if present (for assistant messages)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
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
        """Convert OpenAI-compatible chat completion response to standard AIMessage."""
        choice = response.choices[0]
        message = choice.message

        # Extract and convert tool_calls if present
        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
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

        # Split user-visible content and reasoning content
        content, reasoning_content = self._split_reasoning_and_content(message)

        return AIMessage(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            metadata=self._build_metadata(response, choice),
        )

    
    def _split_reasoning_and_content(self, message: Any):
        """Split message into content and reasoning in a provider-agnostic way.

        Base implementation only:
        - Reads `message.content` as-is
        - Reads optional `message.reasoning_content` if present
        Provider-specific converters (e.g. OpenAI, Bailian) should override
        this method if their SDK embeds reasoning in provider-specific
        structures.
        """
        content = getattr(message, "content", None)
        reasoning = getattr(message, "reasoning_content", None)

        if content is None:
            normalized_content: Any = ""
        else:
            normalized_content = content

        return normalized_content, reasoning

    def _build_metadata(self, response: Any, choice: Any) -> Metadata:
        """Build metadata - subclasses can override to change provider name."""
        raw_usage = getattr(response, "usage", None)
        usage: Optional[Usage] = None

        if raw_usage is not None:
            if isinstance(raw_usage, dict):
                prompt_tokens = int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0)
                completion_tokens = int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0)
                total_tokens = int(raw_usage.get("total_tokens") or (prompt_tokens + completion_tokens))
            else:
                prompt_tokens = int(
                    getattr(raw_usage, "prompt_tokens", None)
                    or getattr(raw_usage, "input_tokens", 0)
                    or 0
                )
                completion_tokens = int(
                    getattr(raw_usage, "completion_tokens", None)
                    or getattr(raw_usage, "output_tokens", 0)
                    or 0
                )
                total_tokens = int(
                    getattr(raw_usage, "total_tokens", None)
                    or (prompt_tokens + completion_tokens)
                )

            usage = Usage(
                prompt_tokens=max(prompt_tokens, 0),
                completion_tokens=max(completion_tokens, 0),
                total_tokens=max(total_tokens, 0),
            )

        return Metadata(
            provider="openai",  # Subclass should override this
            model=getattr(response, "model", None),
            usage=usage,
            finish_reason=choice.finish_reason if choice is not None else None,
        )
    
    def from_provider_chunk(self, chunk: Any) -> UnifiedStreamChunk:
        """Convert OpenAI-compatible streaming chunk to UnifiedStreamChunk.
        
        Base implementation handles:
        - Answer content (delta.content)
        - Tool calls delta (delta.tool_calls)
        - Finish reason and metadata
        
        Subclasses should override if their provider supports reasoning streaming.
        """
        from ...message import ToolCallDelta
        
        choice = chunk.choices[0] if chunk.choices else None
        delta = choice.delta if choice else None
        
        content = delta.content if delta and delta.content else ""
        is_final = bool(choice and choice.finish_reason is not None)
        finish_reason = choice.finish_reason if choice else None
        
        # Extract tool_calls delta
        tool_calls_delta = []
        if delta and hasattr(delta, 'tool_calls') and delta.tool_calls:
            for tc_delta in delta.tool_calls:
                tool_call_delta = ToolCallDelta(
                    index=getattr(tc_delta, 'index', 0) or 0,
                    id=getattr(tc_delta, 'id', None),
                    type=getattr(tc_delta, 'type', None),
                    function_name=getattr(tc_delta.function, 'name', None) if hasattr(tc_delta, 'function') else None,
                    function_arguments=getattr(tc_delta.function, 'arguments', None) if hasattr(tc_delta, 'function') else None,
                )
                tool_calls_delta.append(tool_call_delta)
        
        # Build metadata
        metadata = self._build_chunk_metadata(chunk, choice)
        
        return UnifiedStreamChunk(
            content=content,
            reasoning_content=None,  # Base implementation doesn't support reasoning
            tool_calls_delta=tool_calls_delta,
            is_final=is_final,
            finish_reason=finish_reason,
            metadata=metadata,
        )
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata - subclasses can override."""
        metadata = {
            "provider": "openai",
            "model": getattr(chunk, "model", None),
            "finish_reason": choice.finish_reason if choice else None,
        }
        # Add usage info if available (for stream chunks with stream_options)
        if hasattr(chunk, "usage") and chunk.usage:
            raw_usage = chunk.usage
            if isinstance(raw_usage, dict):
                prompt_tokens = int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0)
                completion_tokens = int(raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0)
                total_tokens = int(raw_usage.get("total_tokens") or (prompt_tokens + completion_tokens))
            else:
                prompt_tokens = int(
                    getattr(raw_usage, "prompt_tokens", None)
                    or getattr(raw_usage, "input_tokens", 0)
                    or 0
                )
                completion_tokens = int(
                    getattr(raw_usage, "completion_tokens", None)
                    or getattr(raw_usage, "output_tokens", 0)
                    or 0
                )
                total_tokens = int(
                    getattr(raw_usage, "total_tokens", None)
                    or (prompt_tokens + completion_tokens)
                )

            metadata["usage"] = {
                "prompt_tokens": max(prompt_tokens, 0),
                "completion_tokens": max(completion_tokens, 0),
                "total_tokens": max(total_tokens, 0),
            }

        return metadata


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
    
    def _apply_reasoning_params(self, kwargs: dict) -> None:
        """Apply reasoning parameters to kwargs based on provider-specific format.
        
        Subclasses should override this to transform the unified 'reasoning' dict
        into provider-specific parameters.
        
        Args:
            kwargs: Request parameters dict that will be passed to SDK.
                   May contain 'reasoning' key with provider-agnostic settings.
        """
        # Base implementation: do nothing (for providers that don't support reasoning)
        pass
    
    def _send_complete(self, messages: List, **kwargs) -> Any:
        """Send non-streaming request to OpenAI-compatible API."""
        # Apply provider-specific reasoning parameter transformations
        self._apply_reasoning_params(kwargs)
        
        raw_response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            **kwargs
        )
        
        return raw_response
    
    def _send_stream(self, messages: List, **kwargs) -> Iterator[Any]:
        """Send streaming request to OpenAI-compatible API."""
        # Apply provider-specific reasoning parameter transformations
        self._apply_reasoning_params(kwargs)
        
        # Add stream_options to include usage in streaming mode (if not already set)
        if 'stream_options' not in kwargs:
            kwargs['stream_options'] = {"include_usage": True}
        
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            yield chunk