from typing import Optional, Dict, Any, List

from pydantic import field_validator

from .base import BaseProviderConfig, OpenAICompatibleMessageConverter, OpenAICompatibleProvider
from ...message import Message
from ...metadata import Metadata


class DeepSeekConfig(BaseProviderConfig):
    """DeepSeek-specific configuration."""
    base_url: Optional[str] = "https://api.deepseek.com"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for DeepSeek."""
        return v or "https://api.deepseek.com"


class DeepSeekMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for DeepSeek message formats."""

    def to_provider_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Extend base format to include reasoning_content in assistant messages.

        DeepSeek thinking mode requires that reasoning_content produced in a
        previous response is echoed back verbatim in the corresponding
        assistant message of the next request.
        """
        result = super().to_provider_format(messages)
        for msg, formatted in zip(messages, result):
            if formatted.get("role") == "assistant":
                rc = getattr(msg, "reasoning_content", None)
                if rc:
                    formatted["reasoning_content"] = rc
        return result

    def _build_metadata(self, response: Any, choice: Any) -> Metadata:
        """Build metadata with 'deepseek' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata.provider = "deepseek"
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'deepseek' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "deepseek"
        return metadata


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider implementation."""

    def _apply_reasoning_params(self, kwargs: dict) -> None:
        """DeepSeek-specific kwargs preprocessing.

        DeepSeek thinking-capable models (e.g. ``deepseek-v4-pro``) default to
        thinking mode enabled, but the API rejects forced ``tool_choice`` in
        that mode with HTTP 400::

            Thinking mode does not support this tool_choice

        Forced tool_choice means either ``"required"`` or a specific function
        dict like ``{"type": "function", "function": {"name": ...}}`` --
        which is exactly what chak's structured output (``returns=``) emits.

        To keep structured output working out of the box, this hook auto-injects
        ``extra_body={"thinking": {"type": "disabled"}}`` whenever a forced
        tool_choice is detected and the caller has not explicitly configured
        the ``thinking`` field. Per DeepSeek docs, ``{"type": "disabled"}`` is
        the official way to turn thinking off via the OpenAI-compatible API.

        Behavior:
        - Trigger only when ``tool_choice`` is a dict or the string ``"required"``.
        - Respect the caller: if ``extra_body['thinking']`` is already present,
          leave it untouched (the caller knows what they are doing).
        - Other ``extra_body`` keys are preserved via shallow merge.
        """
        super()._apply_reasoning_params(kwargs)

        tool_choice = kwargs.get("tool_choice")
        is_forced = isinstance(tool_choice, dict) or tool_choice == "required"
        if not is_forced:
            return

        extra_body = kwargs.get("extra_body") or {}
        if "thinking" in extra_body:
            # Caller explicitly configured thinking -- do not override.
            return

        merged = dict(extra_body)
        merged["thinking"] = {"type": "disabled"}
        kwargs["extra_body"] = merged
