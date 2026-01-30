"""
OpenAI Provider

Official OpenAI API provider.
Official documentation: https://platform.openai.com/docs/api-reference

Supported models:
- GPT-4 series: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini
- GPT-3.5 series: gpt-3.5-turbo
- O1 series: o1, o1-mini, o1-preview
"""
from typing import Optional, Any

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider
from ...message import AIMessage


class OpenAIConfig(BaseProviderConfig):
    """Configuration for OpenAI provider."""
    base_url: Optional[str] = "https://api.openai.com/v1"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for OpenAI."""
        return v or "https://api.openai.com/v1"


class OpenAIMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for OpenAI message formats."""

    def from_provider_response(self, response: Any) -> AIMessage:
        """Handle both Chat Completions and Responses API responses for OpenAI."""
        if hasattr(response, "choices") and response.choices:
            # Chat Completions style - use base implementation
            return super().from_provider_response(response)

        # Responses API style response
        content, reasoning_content, metadata = self._from_responses_response(response)
        return AIMessage(
            content=content,
            reasoning_content=reasoning_content,
            metadata=metadata,
        )

    def _from_responses_response(self, response: Any):
        """Handle OpenAI Responses API response.

        基于 2.txt Reasoning models 文档：
        - 最终答案文本来自 `response.output_text` 或 output 中 type="message" 的 output_text
        - 推理摘要来自 output 中 type="reasoning" 的 summary 数组
        """
        # 1) Final answer content
        content: str = ""
        if hasattr(response, "output_text") and response.output_text:
            content = response.output_text
        else:
            output_items = getattr(response, "output", None)
            if output_items:
                for item in output_items:
                    item_type = getattr(item, "type", None)
                    if item_type is None and isinstance(item, dict):
                        item_type = item.get("type")
                    if item_type == "message":
                        contents = getattr(item, "content", None)
                        if contents is None and isinstance(item, dict):
                            contents = item.get("content")
                        if contents:
                            texts: list[str] = []
                            for c in contents:
                                c_type = getattr(c, "type", None)
                                if c_type is None and isinstance(c, dict):
                                    c_type = c.get("type")
                                if c_type == "output_text":
                                    text = getattr(c, "text", None)
                                    if text is None and isinstance(c, dict):
                                        text = c.get("text")
                                    if isinstance(text, str):
                                        texts.append(text)
                            if texts:
                                content = "".join(texts)
                                break

        # 2) Reasoning summary content
        reasoning_content: Optional[str] = None
        output_items = getattr(response, "output", None)
        if output_items:
            for item in output_items:
                item_type = getattr(item, "type", None)
                if item_type is None and isinstance(item, dict):
                    item_type = item.get("type")
                if item_type == "reasoning":
                    summary_list = getattr(item, "summary", None)
                    if summary_list is None and isinstance(item, dict):
                        summary_list = item.get("summary")
                    if summary_list:
                        texts: list[str] = []
                        for s in summary_list:
                            text = getattr(s, "text", None)
                            if text is None and isinstance(s, dict):
                                text = s.get("text")
                            if isinstance(text, str):
                                texts.append(text)
                        if texts:
                            reasoning_content = "".join(texts)
                            break

        metadata = self._build_metadata(response, choice=None)
        return content, reasoning_content, metadata


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI provider implementation."""
    
    def _send_complete(self, messages, **kwargs):
        """Send non-streaming request for OpenAI.

        OpenAI recommends the
        Responses API for reasoning models. Here we optimistically use
        `client.responses.create` first; if that fails (e.g. model not
        supported by Responses), we fall back to the chat completions
        implementation in the base class.
        """
        model = self.config.model

        # Prefer Responses API when available
        try:
            response = self._client.responses.create(
                model=model,
                input=messages,
                **kwargs,
            )

            # DEBUG: print raw Responses API result
            from rich import print as rprint
            rprint("\n=== DEBUG OpenAI _send_complete (responses) ===")
            rprint(response)
            rprint("=== END DEBUG ===\n")

            return response
        except Exception:
            # Fallback to chat.completions path from base class
            response = super()._send_complete(messages, **kwargs)

            from rich import print as rprint
            rprint("\n=== DEBUG OpenAI _send_complete (chat fallback) ===")
            rprint(response)
            rprint("=== END DEBUG ===\n")

            return response
