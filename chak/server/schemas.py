"""
Pydantic schemas for WebSocket messages.
"""

from typing import Optional, List, Dict, Any, Literal

from pydantic import BaseModel, Field


class InitMessage(BaseModel):
    """Initialize conversation message."""
    
    type: Literal["init"] = "init"
    model_uri: str = Field(description="Model URI, e.g., 'openai:gpt-4'")
    system_message: Optional[str] = Field(default=None, description="System message")
    context_strategy: Optional[str] = Field(
        default="noop",
        description="Context strategy: noop, fifo, summarize"
    )


class SendMessage(BaseModel):
    """Send message to conversation."""
    
    type: Literal["send"] = "send"
    message: str = Field(description="Message content")
    role: str = Field(default="user", description="Message role")
    stream: bool = Field(default=False, description="Enable streaming")


class AddMessagesMessage(BaseModel):
    """Add messages to conversation."""
    
    type: Literal["add_messages"] = "add_messages"
    messages: List[Dict[str, Any]] = Field(description="Message list")


class StatsResponse(BaseModel):
    """Statistics response."""
    
    type: Literal["stats"] = "stats"
    data: Dict[str, Any] = Field(description="Statistics data")


class ErrorResponse(BaseModel):
    """Error response."""
    
    type: Literal["error"] = "error"
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Error details")
