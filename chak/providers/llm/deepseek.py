from typing import Optional, List, Dict, Any, Iterator

import openai
from pydantic import field_validator

from .base import BaseProviderConfig, BaseMessageConverter, Provider
from ...message import Message, MessageChunk, AIMessage


class DeepSeekConfig(BaseProviderConfig):
    """DeepSeek-specific configuration."""
    base_url: Optional[str] = "https://api.deepseek.com"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for DeepSeek."""
        return v or "https://api.deepseek.com"


class DeepSeekMessageConverter(BaseMessageConverter):
    """Converter for DeepSeek message formats."""
    
    def to_provider_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert to DeepSeek message format (OpenAI compatible)."""
        return [
            {
                "role": msg.role or "user",
                "content": msg.content or ""
            }
            for msg in messages
        ]
    
    def from_provider_response(self, response: Any) -> AIMessage:
        """Convert DeepSeek response to standard AIMessage."""
        choice = response.choices[0]
        message = choice.message
        
        return AIMessage(
            content=message.content or "",
            metadata={
                "provider": "deepseek",
                "model": response.model,
                "usage": getattr(response, 'usage', {}),
                "finish_reason": choice.finish_reason,
            }
        )
    
    def from_provider_chunk(self, chunk: Any) -> MessageChunk:
        """Convert DeepSeek streaming chunk to standard MessageChunk."""
        choice = chunk.choices[0] if chunk.choices else None
        delta = choice.delta if choice else None
        
        content = delta.content if delta and delta.content else ""
        is_final = bool(choice and choice.finish_reason is not None)
        
        return MessageChunk(
            content=content,
            is_final=is_final,
            metadata={
                "provider": "deepseek", 
                "model": chunk.model,
                "finish_reason": choice.finish_reason if choice else None
            }
        )


class DeepSeekProvider(Provider):
    """DeepSeek provider implementation."""
    
    def _initialize_client(self):
        """Initialize DeepSeek client."""
        self._client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            http_client=self._create_http_client(),
        )
    
    def _send_complete(self, messages: List, **kwargs) -> Any:
        """Send non-streaming request to DeepSeek."""
        return self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            **kwargs
        )
    
    def _send_stream(self, messages: List, **kwargs) -> Iterator[Any]:
        """Send streaming request to DeepSeek."""
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            yield chunk
