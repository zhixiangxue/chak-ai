"""
Anthropic Claude Provider - OpenAI API Compatible

Anthropic provides Claude models through an OpenAI-compatible API.
Official documentation: https://docs.anthropic.com/

Supported models:
- Claude 3.5 series: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
- Claude 3 series: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
- Latest: claude-sonnet-4-5
"""
from .base import BaseProviderConfig, BaseMessageConverter, Provider
from pydantic import field_validator
from typing import Optional, List, Dict, Any, Iterator
import openai
from ...message import Message, MessageChunk, AIMessage


class AnthropicConfig(BaseProviderConfig):
    """Configuration for Anthropic provider."""
    base_url: Optional[str] = "https://api.anthropic.com/v1/"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for Anthropic."""
        return v or "https://api.anthropic.com/v1/"


class AnthropicMessageConverter(BaseMessageConverter):
    """Converter for Anthropic message formats."""
    
    def to_provider_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert to Anthropic message format (OpenAI compatible)."""
        return [
            {
                "role": msg.role or "user",
                "content": msg.content or ""
            }
            for msg in messages
        ]
    
    def from_provider_response(self, response: Any) -> AIMessage:
        """Convert Anthropic response to standard AIMessage."""
        choice = response.choices[0]
        message = choice.message
        
        return AIMessage(
            content=message.content or "",
            metadata={
                "provider": "anthropic",
                "model": response.model,
                "usage": getattr(response, 'usage', {}),
                "finish_reason": choice.finish_reason,
            }
        )
    
    def from_provider_chunk(self, chunk: Any) -> MessageChunk:
        """Convert Anthropic streaming chunk to standard MessageChunk."""
        choice = chunk.choices[0] if chunk.choices else None
        delta = choice.delta if choice else None
        
        content = delta.content if delta and delta.content else ""
        is_final = bool(choice and choice.finish_reason is not None)
        
        return MessageChunk(
            content=content,
            is_final=is_final,
            metadata={
                "provider": "anthropic",
                "model": chunk.model,
                "finish_reason": choice.finish_reason if choice else None
            }
        )


class AnthropicProvider(Provider):
    """Anthropic Claude provider implementation."""
    
    def _initialize_client(self):
        """Initialize Anthropic client."""
        self._client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )
    
    def _send_complete(self, messages: List, **kwargs) -> Any:
        """Send non-streaming request to Anthropic."""
        return self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            **kwargs
        )
    
    def _send_stream(self, messages: List, **kwargs) -> Iterator[Any]:
        """Send streaming request to Anthropic."""
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            yield chunk
