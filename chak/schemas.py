"""Common types and configurations for chak.

This module defines configuration types used across the library.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Reasoning(BaseModel):
    """Configuration for reasoning mode.
    
    Used by reasoning-capable models (OpenAI o1/o3, Bailian QwQ).
    
    Example:
        >>> reasoning = Reasoning(effort="high", summary="enabled")
        >>> response = conv.send("Solve this problem", reasoning=reasoning)
    """
    
    effort: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Reasoning effort level"
    )
    summary: Literal["auto", "enabled", "disabled"] = Field(
        default="auto",
        description="Whether to include reasoning summary"
    )
    budget: Optional[int] = Field(
        default=None,
        description="Optional token budget for providers that support it (e.g., DashScope thinking_budget)"
    )


class Cache(BaseModel):
    """Provider-agnostic prompt caching configuration.

    Controls which parts of the request are cached and for how long.
    Each provider translates these settings into its own wire format.

    Supported by: Anthropic, OpenAI.

    Anthropic behavior:
        - ``system_prompt``: attach ``cache_control`` to the system prompt block.
        - ``tools``: attach ``cache_control`` to the last tool definition.
        - ``ttl``: 300 s (default, 5-min ephemeral) or 3600 s (1-hour).
        - ``key``: not used (Anthropic has no equivalent).

    OpenAI behavior:
        Caching is **automatic** for prompts ≥ 1024 tokens — no markers needed.
        - ``key``: passed as ``prompt_cache_key`` to improve cache hit rates
          across requests that share long prefixes.
        - ``system_prompt``: on GPT-5.6+ models, wraps the system prompt with
          an explicit ``prompt_cache_breakpoint``. On older models this is
          ignored (caching is still automatic).
        - ``ttl``: not directly used (OpenAI manages retention internally).

    Attributes:
        system_prompt: Cache the system prompt block.
        tools: Cache the last tool definition block.
        ttl: Cache time-to-live in **seconds**.
            Common values: 300 (5 min, default) or 3600 (1 hour).
            Each provider maps this to its nearest supported duration.
        key: Optional cache routing key (OpenAI ``prompt_cache_key``).
            Reuse the same key across requests that share a common prefix
            to improve cache hit rates.

    Example::

        # Anthropic — default 5-minute cache
        Cache(system_prompt=True, tools=True)

        # Anthropic — 1-hour cache
        Cache(system_prompt=True, tools=True, ttl=3600)

        # OpenAI — improve hit rate with a routing key
        Cache(key="tenant:acme:assistant-v1")

        # OpenAI — GPT-5.6+ explicit breakpoint + routing key
        Cache(system_prompt=True, key="tenant:acme:assistant-v1")
    """
    system_prompt: bool = False
    tools: bool = False
    ttl: int = Field(default=300, description="Cache TTL in seconds (300 or 3600)")
    key: Optional[str] = Field(
        default=None,
        description="Cache routing key for OpenAI prompt_cache_key",
    )
