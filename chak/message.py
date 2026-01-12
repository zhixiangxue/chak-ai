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


# ===== Stream Events (for event=True mode) =====
class StreamEvent(BaseModel):
    """流式事件基类
    
    用于 event=True 模式，提供工具调用的完整可观测性。
    开发者可以使用 isinstance() 或 match-case (Python 3.10+) 来处理不同类型的事件。
    
    事件类型：
    - MessageChunk: LLM 内容输出（包含文本、元数据、最终消息等）
    - ToolCallStartEvent: 工具调用开始
    - ToolCallSuccessEvent: 工具调用成功
    - ToolCallErrorEvent: 工具调用失败
    """
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())


class ToolCallStartEvent(StreamEvent):
    """工具调用开始事件
    
    当 LLM 决定调用工具时触发，包含工具名称和参数。
    
    Attributes:
        tool_name: 工具名称
        call_id: 工具调用唯一标识
        arguments: 工具调用参数（已解析为 dict）
    
    Example:
        match event:
            case ToolCallStartEvent(tool_name=name, arguments=args):
                print(f"🔧 调用 {name}")
                print(f"📨 参数: {args}")
    """
    tool_name: str = ""
    call_id: str = ""
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolCallSuccessEvent(StreamEvent):
    """工具调用成功事件
    
    工具执行成功后触发，包含执行结果。
    
    Attributes:
        tool_name: 工具名称
        call_id: 工具调用唯一标识（与 ToolCallStartEvent 的 call_id 对应）
        result: 工具执行结果（字符串格式）
    
    Example:
        match event:
            case ToolCallSuccessEvent(tool_name=name, call_id=cid, result=res):
                print(f"✅ {name} 成功: {res}")
    """
    tool_name: str = ""
    call_id: str = ""
    result: str = ""


class ToolCallErrorEvent(StreamEvent):
    """工具调用失败事件
    
    工具执行失败后触发，包含错误信息。
    
    Attributes:
        tool_name: 工具名称
        call_id: 工具调用唯一标识（与 ToolCallStartEvent 的 call_id 对应）
        error: 错误信息
    
    Example:
        match event:
            case ToolCallErrorEvent(tool_name=name, call_id=cid, error=err):
                print(f"❌ {name} 失败: {err}")
    """
    tool_name: str = ""
    call_id: str = ""
    error: str = ""
