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
from typing import Optional, Dict, Any

import openai
from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider


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


class AzureMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for Azure OpenAI message formats."""
    
    def _build_metadata(self, response: Any, choice: Any) -> Dict[str, Any]:
        """Build metadata with 'azure' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata["provider"] = "azure"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'azure' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "azure"
        return metadata


class AzureProvider(OpenAICompatibleProvider):
    """Azure OpenAI provider implementation."""
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client with AzureOpenAI class."""
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
