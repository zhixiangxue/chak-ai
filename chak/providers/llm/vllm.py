"""
vLLM Provider - OpenAI API Compatible

vLLM is a high-performance LLM inference engine with OpenAI-compatible API.
Official documentation: https://docs.vllm.ai/en/latest/

Supported models:
- All models supported by vLLM (depends on your deployment)
- Common: llama, mistral, qwen, yi, etc.
"""
from typing import Optional, List, Dict, Any, Iterator

import openai
from pydantic import field_validator

from .base import BaseProviderConfig, BaseMessageConverter, Provider
from ...message import Message, MessageChunk, AIMessage


class VLLMConfig(BaseProviderConfig):
    """Configuration for vLLM provider."""
    base_url: Optional[str] = "http://localhost:8000/v1"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for vLLM (local server)."""
        return v or "http://localhost:8000/v1"
    
    @field_validator('api_key', mode='before')
    @classmethod
    def set_default_api_key(cls, v):
        """Set default API key for vLLM (required but often unused in local deployment)."""
        return v or "EMPTY"


class VLLMMessageConverter(BaseMessageConverter):
    """Converter for vLLM message formats."""
    
    def to_provider_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert to vLLM message format (OpenAI compatible)."""
        return [
            {
                "role": msg.role or "user",
                "content": msg.content or ""
            }
            for msg in messages
        ]
    
    def from_provider_response(self, response: Any) -> AIMessage:
        """Convert vLLM response to standard AIMessage."""
        choice = response.choices[0]
        message = choice.message
        
        return AIMessage(
            content=message.content or "",
            metadata={
                "provider": "vllm",
                "model": response.model,
                "usage": getattr(response, 'usage', {}),
                "finish_reason": choice.finish_reason,
            }
        )
    
    def from_provider_chunk(self, chunk: Any) -> MessageChunk:
        """Convert vLLM streaming chunk to standard MessageChunk."""
        choice = chunk.choices[0] if chunk.choices else None
        delta = choice.delta if choice else None
        
        content = delta.content if delta and delta.content else ""
        is_final = bool(choice and choice.finish_reason is not None)
        
        return MessageChunk(
            content=content,
            is_final=is_final,
            metadata={
                "provider": "vllm",
                "model": chunk.model,
                "finish_reason": choice.finish_reason if choice else None
            }
        )


class VLLMProvider(Provider):
    """vLLM provider implementation."""
    
    def _initialize_client(self):
        """Initialize vLLM client."""
        self._client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            http_client=self._create_http_client(),
        )
    
    def _send_complete(self, messages: List, **kwargs) -> Any:
        """Send non-streaming request to vLLM."""
        return self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            **kwargs
        )
    
    def _send_stream(self, messages: List, **kwargs) -> Iterator[Any]:
        """Send streaming request to vLLM."""
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            yield chunk
