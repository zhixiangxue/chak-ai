import uuid
from datetime import datetime
from typing import Literal, Optional, List, Union, Dict, Any, TYPE_CHECKING
from contextvars import ContextVar

from pydantic import BaseModel, Field

# Always import Attachment for runtime (Pydantic needs it)
from .attachment import Attachment
from .metadata import Metadata

# Context variable for tracking turn ID across the call stack
_current_turn_id: ContextVar[Optional[str]] = ContextVar('turn_id', default=None)


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
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Unique message ID
    content: Optional[Union[str, List[Dict[str, Any]]]] = None  # Text or multimodal content array
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    refusal: Optional[str] = None
    attachments: List[Attachment] = Field(default_factory=list)  # Original attachments associated with this message
    metadata: Metadata = Field(default_factory=Metadata)  # Metadata (provider, model, usage, etc.)
    custom: Dict[str, Any] = Field(default_factory=dict)  # Custom data for application-specific use
    timestamp: datetime = Field(default_factory=datetime.now)  # 消息创建时间
    turn_id: Optional[str] = Field(default_factory=lambda: _current_turn_id.get())  # Turn ID from context
    
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


# ===== Unified Message Type =====
Message = Union[
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
]


class MessageChunk(BaseModel):
    """Represents a streaming chunk of answer content.

    This chunk represents user-visible assistant output.
    """

    content: str = ""
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None
    final_message: Optional['Message'] = None  # When is_final=True, contains the complete final message


class ReasoningChunk(BaseModel):
    """Represents a streaming chunk of reasoning content.

    This chunk represents model reasoning or thinking output.
    """

    content: str = ""
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None


class ToolCallDelta(BaseModel):
    """Represents an incremental update to a tool call.
    
    Used in streaming responses to accumulate tool_calls progressively.
    """
    index: int = 0
    id: Optional[str] = None
    type: Optional[str] = None
    function_name: Optional[str] = None
    function_arguments: Optional[str] = None  # Incremental arguments string


class UnifiedStreamChunk(BaseModel):
    """Unified streaming chunk format for all providers.
    
    This is the internal format that all provider converters must produce.
    Manager only processes this unified format, never raw provider chunks.
    
    Attributes:
        content: User-visible text content (if any)
        reasoning_content: Reasoning/thinking content (if any)
        tool_calls_delta: Incremental tool call updates (if any)
        finish_reason: Finish reason ('stop', 'tool_calls', etc.)
        is_final: Whether this is the last chunk
        metadata: Additional metadata
    """
    content: str = ""
    reasoning_content: Optional[str] = None
    tool_calls_delta: List[ToolCallDelta] = Field(default_factory=list)
    finish_reason: Optional[str] = None
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None
# ===== Stream Events (for event=True mode) =====
class StreamEvent(BaseModel):
    """流式事件基类
    
    用于 event=True 模式，提供工具调用的完整可观测性。
    开发者可以使用 isinstance() 或 match-case (Python 3.10+) 来处理不同类型的事件。
    
    事件类型：
    - MessageChunk: LLM 最终答案内容输出（包含文本、元数据、最终消息等）
    - ReasoningChunk: LLM 推理内容输出（思考过程、summary 等）
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
        timestamp: 事件时间戳（秒）
    
    Example:
        match event:
            case ToolCallStartEvent(tool_name=name, arguments=args, timestamp=ts):
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
        timestamp: 事件时间戳（秒）
    
    Example:
        match event:
            case ToolCallSuccessEvent(tool_name=name, call_id=cid, result=res, timestamp=ts):
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
        timestamp: 事件时间戳（秒）
    
    Example:
        match event:
            case ToolCallErrorEvent(tool_name=name, call_id=cid, error=err, timestamp=ts):
                print(f"❌ {name} 失败: {err}")
    """
    tool_name: str = ""
    call_id: str = ""
    error: str = ""
