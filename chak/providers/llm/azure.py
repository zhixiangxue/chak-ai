"""
Azure OpenAI Provider - OpenAI API Compatible

Microsoft Azure OpenAI Service provides hosted OpenAI models.
Official documentation: https://learn.microsoft.com/en-us/azure/ai-services/openai/

Note: Azure OpenAI uses different authentication and endpoint format:
- Authentication: api-key header (not Bearer token)
- Endpoint: https://{resource}.openai.azure.com/openai/deployments/{deployment}/...
- API Version: Required query parameter (e.g., 2024-02-01)

Supported models (via deployments):
- GPT-4 series: gpt-4, gpt-4-turbo, gpt-4o
- GPT-3.5 series: gpt-3.5-turbo
"""
from .base import BaseProviderConfig, BaseMessageConverter, Provider
from pydantic import field_validator
from typing import Optional, List, Dict, Any, Iterator
import openai
from ...message import Message, MessageChunk, AIMessage


class AzureConfig(BaseProviderConfig):
    """Configuration for Azure OpenAI provider."""
    base_url: Optional[str] = None  # Must be provided by user
    api_version: str = "2024-02-01"  # Default to latest stable version
    
    @field_validator('base_url', mode='before')
    @classmethod
    def validate_base_url(cls, v):
        """Validate that base_url is provided for Azure."""
        if not v:
            raise ValueError(
                "Azure OpenAI requires base_url. "
                "Format: https://{your-resource-name}.openai.azure.com"
            )
        return v


class AzureMessageConverter(BaseMessageConverter):
    """Converter for Azure OpenAI message formats."""
    
    def to_provider_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert to Azure OpenAI message format (same as OpenAI)."""
        return [
            {
                "role": msg.role or "user",
                "content": msg.content or ""
            }
            for msg in messages
        ]
    
    def from_provider_response(self, response: Any) -> AIMessage:
        """Convert Azure OpenAI response to standard AIMessage."""
        choice = response.choices[0]
        message = choice.message
        
        return AIMessage(
            content=message.content or "",
            metadata={
                "provider": "azure",
                "model": response.model,
                "usage": getattr(response, 'usage', {}),
                "finish_reason": choice.finish_reason,
            }
        )
    
    def from_provider_chunk(self, chunk: Any) -> MessageChunk:
        """Convert Azure OpenAI streaming chunk to standard MessageChunk."""
        choice = chunk.choices[0] if chunk.choices else None
        delta = choice.delta if choice else None
        
        content = delta.content if delta and delta.content else ""
        is_final = bool(choice and choice.finish_reason is not None)
        
        return MessageChunk(
            content=content,
            is_final=is_final,
            metadata={
                "provider": "azure",
                "model": chunk.model,
                "finish_reason": choice.finish_reason if choice else None
            }
        )


class AzureProvider(Provider):
    """Azure OpenAI provider implementation."""
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client."""
        # Ensure base_url is not None (already validated in config)
        if not self.config.base_url:
            raise ValueError("Azure OpenAI requires base_url")
        
        # Azure OpenAI uses AzureOpenAI client with different parameters
        self._client = openai.AzureOpenAI(
            api_key=self.config.api_key,
            azure_endpoint=self.config.base_url,
            api_version=getattr(self.config, 'api_version', '2024-02-01'),
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )
    
    def _send_complete(self, messages: List, **kwargs) -> Any:
        """Send non-streaming request to Azure OpenAI."""
        return self._client.chat.completions.create(
            model=self.config.model,  # This is the deployment name in Azure
            messages=messages,
            **kwargs
        )
    
    def _send_stream(self, messages: List, **kwargs) -> Iterator[Any]:
        """Send streaming request to Azure OpenAI."""
        stream = self._client.chat.completions.create(
            model=self.config.model,  # This is the deployment name in Azure
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            yield chunk
