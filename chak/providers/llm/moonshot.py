from typing import Optional, Dict, Any

from pydantic import field_validator

from .openai_compat import OpenAICompatibleMessageConverter, OpenAICompatibleProvider
from .base import BaseProviderConfig
from ...metadata import Metadata


class MoonshotConfig(BaseProviderConfig):
    """Moonshot-specific configuration."""
    base_url: Optional[str] = "https://api.moonshot.cn/v1"

    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for Moonshot."""
        return v or "https://api.moonshot.cn/v1"


class MoonshotMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for Moonshot message formats."""

    def _build_metadata(self, response: Any, choice: Any) -> Metadata:
        """Build metadata with 'moonshot' as provider name."""
        metadata = super()._build_metadata(response, choice)
        metadata.provider = "moonshot"
        return metadata

    def _build_chunk_metadata(self, chunk: Any, choice: Any) -> Dict[str, Any]:
        """Build chunk metadata with 'moonshot' as provider name."""
        metadata = super()._build_chunk_metadata(chunk, choice)
        metadata["provider"] = "moonshot"
        return metadata


# Model families whose thinking cannot be disabled at all — per Moonshot's
# official docs (see /guide/use-kimi-k2-thinking-model). For these models,
# injecting ``thinking={"type": "disabled"}`` either errors or is silently
# ignored, and forced ``tool_choice`` (chak's structured-output shape)
# fundamentally cannot succeed. We skip the injection so the underlying API
# error surfaces cleanly instead of being masked by a second failure.
_THINKING_ALWAYS_ON_PREFIXES = (
    "kimi-k3",           # K3 flagship — thinking always on, uses reasoning_effort="max"
    "kimi-k2.7-code",    # K2.7-code family (incl. -highspeed) — thinking always on
)


# Model families where forced ``tool_choice`` structured output is broken by
# always-on thinking (see ``_THINKING_ALWAYS_ON_PREFIXES``) but Moonshot's
# ``response_format={"type": "json_schema"}`` still works. Chak routes
# ``returns=<PydanticModel>`` through the response_format path for these
# so structured output remains usable end-to-end.
#
# Kept intentionally narrow: legacy ``moonshot-v1-*`` and the disable-capable
# ``kimi-k2.5``/``kimi-k2.6`` families keep the classic tool-call path (which
# Path A's thinking-disable injection already fixed) to avoid changing
# behavior for models that are known to work.
_JSON_SCHEMA_RESPONSE_FORMAT_PREFIXES = _THINKING_ALWAYS_ON_PREFIXES


def _thinking_can_be_disabled(model: str) -> bool:
    """Return True iff the model supports thinking.type='disabled'.

    Whitelist-by-blacklist: we explicitly know K3 and K2.7-code cannot
    disable thinking. Every other Moonshot model (k2.5, k2.6, moonshot-v1-*,
    plus future models until docs say otherwise) is treated as "injection
    is safe" — either because it's a thinking model that supports the flag
    (k2.5/k2.6) or because it's a non-thinking model where the extra field
    is silently ignored (moonshot-v1-*).
    """
    lower = (model or "").lower()
    return not any(lower.startswith(p) for p in _THINKING_ALWAYS_ON_PREFIXES)


class MoonshotProvider(OpenAICompatibleProvider):
    """Moonshot provider implementation."""

    def supports_json_schema_response_format(self, model: str) -> bool:
        """Enable ``response_format=json_schema`` for always-thinking Kimi models.

        Rationale: those families (K3, K2.7-code) reject forced
        ``tool_choice`` because thinking can't be disabled, so chak's default
        structured-output path is fundamentally unusable on them. Moonshot's
        OpenAI-compatible ``response_format`` accepts a JSON schema alongside
        thinking, which lets chak recover a working ``returns=`` flow.

        Kept in sync with ``_JSON_SCHEMA_RESPONSE_FORMAT_PREFIXES`` — see that
        constant for the "why not more families" reasoning.
        """
        lower = (model or "").lower()
        return any(lower.startswith(p) for p in _JSON_SCHEMA_RESPONSE_FORMAT_PREFIXES)

    def _apply_reasoning_params(self, kwargs: dict) -> None:
        """Moonshot-specific kwargs preprocessing.

        Moonshot's thinking-capable Kimi models reject forced ``tool_choice``
        with HTTP 400::

            tool_choice 'specified' is incompatible with thinking enabled

        Forced tool_choice means either ``"required"`` or a specific-function
        dict like ``{"type": "function", "function": {"name": ...}}`` — which
        is exactly what chak's structured output (``returns=``) emits. Same
        shape as DeepSeek's thinking-mode restriction; same fix pattern.

        This hook auto-injects ``extra_body={"thinking": {"type": "disabled"}}``
        whenever a forced tool_choice is detected AND the model supports
        disabling thinking (see ``_thinking_can_be_disabled``). Per Moonshot
        docs, ``{"type": "disabled"}`` is the official OpenAI-compat way to
        turn thinking off.

        Behavior:
        - Trigger only when ``tool_choice`` is a dict or the string ``"required"``.
        - Respect the caller: if ``extra_body['thinking']`` is already present,
          leave it untouched (the caller knows what they are doing).
        - Skip models that cannot disable thinking (K3, K2.7-code). For those,
          structured output currently isn't supportable on this provider — the
          native 400 will bubble up unmasked.
        - Other ``extra_body`` keys are preserved via shallow merge.
        """
        super()._apply_reasoning_params(kwargs)

        tool_choice = kwargs.get("tool_choice")
        is_forced = isinstance(tool_choice, dict) or tool_choice == "required"
        if not is_forced:
            return

        model = kwargs.get("model") or ""
        if not _thinking_can_be_disabled(model):
            # Nothing to do — this model forces thinking on. Let the caller
            # see the underlying "tool_choice 'specified' is incompatible with
            # thinking enabled" error verbatim; masking it would be worse.
            return

        extra_body = kwargs.get("extra_body") or {}
        if "thinking" in extra_body:
            # Caller explicitly configured thinking -- do not override.
            return

        merged = dict(extra_body)
        merged["thinking"] = {"type": "disabled"}
        kwargs["extra_body"] = merged
