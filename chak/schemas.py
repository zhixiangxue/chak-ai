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

    Currently supported by: Anthropic.

    Attributes:
        system_prompt: Cache the system prompt block.
        tools: Cache the last tool definition block.
        ttl: Cache time-to-live in **seconds**.
            Common values: 300 (5 min, default) or 3600 (1 hour).
            Each provider maps this to its nearest supported duration.

    Example::

        # Default 5-minute cache
        Cache(system_prompt=True, tools=True)

        # 1-hour cache — better for chat products with sporadic requests
        Cache(system_prompt=True, tools=True, ttl=3600)
    """
    system_prompt: bool = False
    tools: bool = False
    ttl: int = Field(default=300, description="Cache TTL in seconds (300 or 3600)")
