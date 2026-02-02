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
from ...message import AIMessage, MessageChunk, ReasoningChunk, UnifiedStreamChunk
from ...schemas import Reasoning


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
    
    def from_provider_chunk(self, chunk: Any) -> UnifiedStreamChunk:
        """Convert OpenAI streaming chunk to UnifiedStreamChunk.
        
        Handles both:
        - Chat Completions API chunks (delta-based, has 'choices' attribute)
        - Responses API streaming events (event-based, no 'choices' attribute)
        """
        # Distinguish by checking for 'choices' attribute
        # Chat Completions chunks have 'choices', Responses API events don't
        if hasattr(chunk, 'choices'):
            # Chat Completions chunk handling
            return super().from_provider_chunk(chunk)
        else:
            # Responses API event handling
            return self._from_responses_event(chunk)
    
    def _from_responses_event(self, event: Any) -> UnifiedStreamChunk:
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
            return UnifiedStreamChunk(
                content="",
                reasoning_content=event.delta,
                is_final=False,
            )
        
        # Handle answer text delta events (ANSWER CONTENT)
        if isinstance(event, ResponseTextDeltaEvent):
            return UnifiedStreamChunk(
                content=event.delta,
                reasoning_content=None,
                is_final=False,
            )
        
        # Handle output item added events
        if isinstance(event, ResponseOutputItemAddedEvent):
            item_type = getattr(event.item, 'type', None)
            
            # Reasoning item added - reasoning started
            if item_type == 'reasoning':
                return UnifiedStreamChunk(content="", reasoning_content="", is_final=False)
            
            # Message item added - answer message started
            if item_type == 'message':
                return UnifiedStreamChunk(content="", reasoning_content=None, is_final=False)
        
        # Handle completion events
        if isinstance(event, ResponseCompletedEvent):
            metadata = self._build_metadata(event.response, choice=None)
            return UnifiedStreamChunk(
                content="",
                reasoning_content=None,
                is_final=True,
                finish_reason=getattr(event.response, 'status', None),
                metadata=metadata.model_dump() if metadata else None,
            )
        
        # For other events (created, in_progress, done, etc.), return empty chunk
        return UnifiedStreamChunk(content="", reasoning_content=None, is_final=False)

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
    
    def _apply_reasoning_params(self, kwargs: dict) -> None:
        """Apply reasoning parameters for OpenAI.
        
        OpenAI Responses API natively supports 'reasoning' parameter with format:
            reasoning = {"effort": "low|medium|high", "summary": "auto|detailed|concise"}
        
        This method transforms chak's Reasoning object to OpenAI's format.
        For Chat Completions API fallback, the parameter will be removed in exception handlers.
        """
        reasoning = kwargs.get('reasoning')
        if not reasoning:
            return
        
        # Convert Reasoning object to dict if needed
        if isinstance(reasoning, Reasoning):
            reasoning_dict = reasoning.model_dump(exclude_none=True)
        elif isinstance(reasoning, dict):
            reasoning_dict = reasoning
        else:
            # Unknown type, remove it
            kwargs.pop('reasoning', None)
            return
        
        # Build OpenAI reasoning parameter
        openai_reasoning = {}
        
        # Map effort (direct mapping)
        if 'effort' in reasoning_dict:
            openai_reasoning['effort'] = reasoning_dict['effort']
        
        # Map summary (chak uses "auto"/"enabled"/"disabled", OpenAI uses "auto"/"detailed"/"concise")
        if 'summary' in reasoning_dict:
            summary_value = reasoning_dict['summary']
            if summary_value == 'enabled':
                openai_reasoning['summary'] = 'auto'  # Use auto to get best available
            elif summary_value == 'auto':
                openai_reasoning['summary'] = 'auto'
            # 'disabled' means don't include summary parameter at all
        
        # Replace with OpenAI format
        if openai_reasoning:
            kwargs['reasoning'] = openai_reasoning
        else:
            # No valid reasoning config, remove it
            kwargs.pop('reasoning', None)
    
    def _send_complete(self, messages, **kwargs):
        """Send non-streaming request for OpenAI.

        OpenAI has two APIs:
        1. Responses API: Supports reasoning, but NOT function calling
        2. Chat Completions API: Supports function calling, but NOT reasoning
        
        Strategy:
        - If reasoning is requested → Use Responses API
        - Otherwise → Use Chat Completions API (default, supports function calling)
        
        Returns:
            Raw SDK response object (will be parsed by converter later)
        """
        model = self.config.model
        has_reasoning = 'reasoning' in kwargs
        
        # Route 1: Responses API (for reasoning)
        if has_reasoning:
            try:
                response = self._client.responses.create(
                    model=model,
                    input=messages,
                    **kwargs,
                )
                # Return raw Responses API response
                return response
            except Exception as e:
                # Friendly error for unsupported models
                error_msg = str(e).lower()
                if 'unsupported parameter' in error_msg or 'reasoning' in error_msg:
                    raise ValueError(
                        f"Model '{model}' does not support reasoning parameter. "
                        f"Reasoning is only supported by specific models like gpt-5, gpt-5-mini, etc. "
                        f"Please use a reasoning-capable model or remove the reasoning parameter."
                    ) from e
                # Other errors: fallback to Chat Completions without reasoning
                kwargs.pop('reasoning', None)
        
        # Route 2: Chat Completions API (default, supports function calling)
        # Apply provider-specific reasoning parameter transformations
        self._apply_reasoning_params(kwargs)
        
        # Call Chat Completions API
        raw_response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        # Return raw Chat Completions response
        return raw_response
    
    def _send_stream(self, messages, **kwargs):
        """Send streaming request for OpenAI.

        OpenAI has two APIs:
        1. Responses API: Supports reasoning, but NOT function calling
        2. Chat Completions API: Supports function calling, but NOT reasoning
        
        Strategy:
        - If reasoning is requested → Use Responses API streaming
        - Otherwise → Use Chat Completions API streaming (default, supports function calling)
        """
        model = self.config.model
        has_reasoning = 'reasoning' in kwargs

        # Route 1: Responses API streaming (for reasoning)
        if has_reasoning:
            try:
                return self._client.responses.create(
                    model=model,
                    input=messages,
                    stream=True,
                    **kwargs,
                )
            except Exception as e:
                # Friendly error for unsupported models
                error_msg = str(e).lower()
                if 'unsupported parameter' in error_msg or 'reasoning' in error_msg:
                    raise ValueError(
                        f"Model '{model}' does not support reasoning parameter. "
                        f"Reasoning is only supported by specific models like gpt-5, gpt-5-mini, etc. "
                        f"Please use a reasoning-capable model or remove the reasoning parameter."
                    ) from e
                # Other errors: fallback to Chat Completions without reasoning
                kwargs.pop('reasoning', None)
        
        # Route 2: Chat Completions API streaming (default, supports function calling)
        # Apply provider-specific reasoning parameter transformations
        self._apply_reasoning_params(kwargs)
        
        # Add stream_options to include usage in streaming mode (if not already set)
        if 'stream_options' not in kwargs:
            kwargs['stream_options'] = {"include_usage": True}
        
        stream = self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs
        )
        
        return stream
