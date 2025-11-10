"""
xAI (Grok) Provider - OpenAI API Compatible

xAI provides Grok models through an OpenAI-compatible API.
Official documentation: https://docs.x.ai/

Supported models:
- grok-4: Latest Grok model
- grok-vision-beta: Grok model with vision capabilities
"""
from typing import Optional, List, Dict, Any, Iterator

import openai
from pydantic import field_validator

from .base import BaseProviderConfig, BaseMessageConverter, Provider
from ...message import Message, MessageChunk, AIMessage


class XAIConfig(BaseProviderConfig):
    """Configuration for xAI provider."""
    base_url: Optional[str] = "https://api.x.ai/v1"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for xAI."""
        return v or "https://api.x.ai/v1"
    
    @field_validator('timeout', mode='before')
    @classmethod
    def set_default_timeout(cls, v):
        """Set default timeout for reasoning models (3600s)."""
        return v if v is not None else 3600


class XAIMessageConverter(BaseMessageConverter):
    """Converter for xAI message formats."""
    
    def to_provider_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert to xAI message format (OpenAI compatible)."""
        return [
            {
                "role": msg.role or "user",
                "content": msg.content or ""
            }
            for msg in messages
        ]
    
    def from_provider_response(self, response: Any) -> AIMessage:
        """Convert xAI response to standard AIMessage."""
        choice = response.choices[0]
        message = choice.message
        
        return AIMessage(
            content=message.content or "",
            metadata={
                "provider": "xai",
                "model": response.model,
                "usage": getattr(response, 'usage', {}),
                "finish_reason": choice.finish_reason,
            }
        )
    
    def from_provider_chunk(self, chunk: Any) -> MessageChunk:
        """Convert xAI streaming chunk to standard MessageChunk."""
        choice = chunk.choices[0] if chunk.choices else None
        delta = choice.delta if choice else None
        
        content = delta.content if delta and delta.content else ""
        is_final = bool(choice and choice.finish_reason is not None)
        
        return MessageChunk(
            content=content,
            is_final=is_final,
            metadata={
                "provider": "xai",
                "model": chunk.model,
                "finish_reason": choice.finish_reason if choice else None
            }
        )


class XAIProvider(Provider):
    """xAI (Grok) provider implementation."""
    
    def _initialize_client(self):
        """Initialize xAI client."""
        self._client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            http_client=self._create_http_client(),
        )
    
    def _send_complete(self, messages: List, **kwargs) -> Any:
        """Send non-streaming request to xAI."""
        return self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            **kwargs
        )
    
    def _send_stream(self, messages: List, **kwargs) -> Iterator[Any]:
        """Send streaming request to xAI."""
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            yield chunk
