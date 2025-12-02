from datetime import datetime
from typing import Literal, Optional, List, Union, Dict, Any, TYPE_CHECKING

from pydantic import BaseModel, Field

# Always import Attachment for runtime (Pydantic needs it)
from .attachment import Attachment


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
    content: Optional[Union[str, List[Dict[str, Any]]]] = None  # Text or multimodal content array
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    refusal: Optional[str] = None
    attachments: List[Attachment] = Field(default_factory=list)  # Original attachments associated with this message
    metadata: Dict[str, Any] = Field(default_factory=dict)  # 元数据（provider、model、usage等）
    timestamp: datetime = Field(default_factory=datetime.now)  # 消息创建时间
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like Attachment
    
    def is_multimodal(self) -> bool:
        """Check if this message contains multimodal content
        
        Returns:
            True if content is a list (multimodal), False if string (text only)
        """
        return isinstance(self.content, list)


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
    tool_call_id: Optional[str] = None  # OpenAI required field for tool response


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
