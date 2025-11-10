from datetime import datetime
from typing import Literal, Optional, List, Union, Dict, Any

from pydantic import BaseModel, Field


class Function(BaseModel):
    """Represents a function call."""

    arguments: str
    name: str


class ChatCompletionMessageToolCall(BaseModel):
    """Represents a tool call in a chat completion message."""

    id: str
    function: Function
    type: Literal["function"]


# ===== Base Message =====
class BaseMessage(BaseModel):
    """所有消息的基类"""
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    refusal: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 元数据（provider、model、usage等）
    timestamp: datetime = Field(default_factory=datetime.now)  # 消息创建时间


# ===== Real Conversation Messages =====
class HumanMessage(BaseMessage):
    """人类消息"""
    role: Literal["user"] = "user"


class AIMessage(BaseMessage):
    """AI 消息"""
    role: Literal["assistant"] = "assistant"


class SystemMessage(BaseMessage):
    """系统消息（用户设置的系统指令）"""
    role: Literal["system"] = "system"


class ToolMessage(BaseMessage):
    """工具消息"""
    role: Literal["tool"] = "tool"


# ===== Strategy Marker Message =====
class MarkerMessage(BaseMessage):
    """
    标记消息（由 ContextStrategy 插入的特殊标记）
    
    用于标记上下文管理的处理节点，如摘要、截断等。
    这些消息不是真实对话，而是对话流中的“里程碑标记”。
    
    Examples:
        # 摘要标记
        MarkerMessage(
            content="[SUMMARY] 之前讨论了 A、B、C",
            metadata={"type": "summary"}
        )
        
        # 截断标记
        MarkerMessage(
            content="[已截断 50 条消息]",
            metadata={"type": "truncated", "count": 50}
        )
    """
    role: Literal["context"] = "context"


# ===== Unified Message Type =====
Message = Union[
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    MarkerMessage,
]


class MessageChunk(BaseModel):
    """Represents a streaming chunk of a message."""

    content: str = ""
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None
    final_message: Optional['Message'] = None  # 当 is_final=True 时，包含完整的最终消息
