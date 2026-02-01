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
