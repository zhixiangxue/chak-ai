"""
OpenAI Provider

Official OpenAI API provider.
Official documentation: https://platform.openai.com/docs/api-reference

Supported models:
- GPT-4 series: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini
- GPT-3.5 series: gpt-3.5-turbo
- O1 series: o1, o1-mini, o1-preview
"""
from typing import Optional, Any, Union, get_args

from pydantic import field_validator
from openai.types.responses import (
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseOutputItemAddedEvent,
    ResponseCompletedEvent,
    ResponseStreamEvent,
)

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider
from ...message import AIMessage, MessageChunk, ReasoningChunk

# Get all Responses API event types from the Union
_RESPONSE_EVENT_TYPES = get_args(ResponseStreamEvent)


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
    
    def from_provider_chunk(self, chunk: Any) -> Union[MessageChunk, ReasoningChunk]:
        """Convert OpenAI streaming chunk to MessageChunk or ReasoningChunk.
        
        Handles both:
        - Chat Completions API chunks (delta-based)
        - Responses API streaming events (event-based)
        """
        # Check if this is a Responses API event
        if isinstance(chunk, _RESPONSE_EVENT_TYPES):
            return self._from_responses_event(chunk)
        
        # Fall back to Chat Completions chunk handling
        return super().from_provider_chunk(chunk)
    
    def _from_responses_event(self, event: Any) -> Union[MessageChunk, ReasoningChunk]:
        """Handle OpenAI Responses API streaming events.
        
        Event types and flow:
        1. response.created: Response object created
        2. response.in_progress: Response is being generated
        3. response.output_item.added (reasoning): Reasoning started
        4. response.reasoning_summary_part.added: Reasoning summary part added
        5. response.reasoning_summary_text.delta: Reasoning summary text delta (REASONING CONTENT HERE)
        6. response.reasoning_summary_text.done: Reasoning summary text completed
        7. response.reasoning_summary_part.done: Reasoning summary part completed
        8. response.output_item.done (reasoning): Reasoning completed
        9. response.output_item.added (message): Answer message started
        10. response.content_part.added: Content part added
        11. response.output_text.delta: Text content delta (ANSWER CONTENT HERE)
        12. response.output_text.done: Text completed
        13. response.content_part.done: Content part completed
        14. response.output_item.done (message): Message completed
        15. response.completed: Response generation completed
        
        Note: Reasoning raw content is encrypted and not streamed.
        However, reasoning summary (when requested) IS streamed as deltas.
        """
        # Handle reasoning summary text delta events (REASONING CONTENT)
        if isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
            return ReasoningChunk(
                content=event.delta,
                is_final=False,
            )
        
        # Handle answer text delta events (ANSWER CONTENT)
        if isinstance(event, ResponseTextDeltaEvent):
            return MessageChunk(
                content=event.delta,
                is_final=False,
            )
        
        # Handle output item added events
        if isinstance(event, ResponseOutputItemAddedEvent):
            item_type = getattr(event.item, 'type', None)
            
            # Reasoning item added - reasoning started
            if item_type == 'reasoning':
                return ReasoningChunk(content="", is_final=False)
            
            # Message item added - answer message started
            if item_type == 'message':
                return MessageChunk(content="", is_final=False)
        
        # Handle completion events
        if isinstance(event, ResponseCompletedEvent):
            metadata = self._build_metadata(event.response, choice=None)
            return MessageChunk(content="", is_final=True, metadata=metadata)
        
        # For other events (created, in_progress, done, etc.), return empty MessageChunk
        return MessageChunk(content="", is_final=False)

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

        OpenAI recommends the Responses API for reasoning models. Here we
        optimistically use `client.responses.create` first; if that fails
        (e.g. model not supported by Responses), we fall back to the chat
        completions implementation in the base class.
        """
        model = self.config.model

        # Prefer Responses API when available
        try:
            response = self._client.responses.create(
                model=model,
                input=messages,
                **kwargs,
            )

            return response
        except Exception:
            # Fallback to chat.completions path from base class
            response = super()._send_complete(messages, **kwargs)

            return response
    
    def _send_stream(self, messages, **kwargs):
        """Send streaming request for OpenAI.

        OpenAI's Responses API supports streaming with reasoning. Here we
        optimistically use `client.responses.create` with `stream=True` first;
        if that fails, we fall back to chat completions streaming.
        """
        model = self.config.model

        # Prefer Responses API streaming when available
        try:
            return self._client.responses.create(
                model=model,
                input=messages,
                stream=True,
                **kwargs,
            )
        except Exception:
            # Fallback to chat.completions streaming from base class
            return super()._send_stream(messages, **kwargs)
