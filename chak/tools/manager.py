"""
ToolManager 类：管理 LLM + 工具调用循环

核心功能：
- 自动管理多轮工具调用循环
- 并行执行多个工具调用
- 自动重试和错误处理
- 与上下文策略集成
- 支持 MCP 工具和原生函数工具
"""

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Union

# Matches the full HTTPS URL embedded inside an attachment:// reference emitted by tools.
_ATTACHMENT_RE = re.compile(r"attachment://(https?://\S+)")

# Extension → Attachment subclass mapping for resolving tool-produced attachment URLs.
from ..attachment.base import Attachment as _Attachment
from ..attachment import PDF as _PDF, DOC as _DOC, Excel as _Excel, CSV as _CSV, TXT as _TXT, Image as _Image

_EXT_TO_TYPE: Dict[str, type] = {
    ".pdf":  _PDF,
    ".docx": _DOC,
    ".doc":  _DOC,
    ".xlsx": _Excel,
    ".xls":  _Excel,
    ".csv":  _CSV,
    ".md":   _TXT,
    ".txt":  _TXT,
    ".png":  _Image,
    ".jpg":  _Image,
    ".jpeg": _Image,
    ".gif":  _Image,
    ".webp": _Image,
}


def _url_to_attachment(url: str) -> _Attachment:
    """Resolve a tool-output HTTPS URL to a typed Attachment subclass (no reader — output only)."""
    ext = Path(url.split("?")[0]).suffix.lower()
    cls = _EXT_TO_TYPE.get(ext)
    if cls is None:
        return _Attachment(source=url)
    return cls(source=url)


# Phrases that strongly indicate the model genuinely does NOT support function
# calling.  Each phrase must appear as a contiguous lower-case substring of the
# provider error message.  The previous heuristic matched the bare word
# ``invalid`` which also fires on generic Anthropic ``invalid_request_error``
# responses (e.g. malformed content blocks), masking the real failure as a
# spurious "doesn't support function calling" warning.  Keep this list narrow.
_NO_TOOL_SUPPORT_PHRASES = (
    "does not support tool",
    "does not support tools",
    "does not support function",
    "doesn't support tool",
    "doesn't support tools",
    "doesn't support function",
    "function calling is not supported",
    "tool calling is not supported",
    "tools are not supported",
    "tool_choice is not supported",
    "tool_use is not supported",
    "function_call is not supported",
    "no tool support",
    "tools parameter is not supported",
)


def _is_no_tool_support_error(error: Exception) -> bool:
    """Return True only when the provider error clearly indicates the model
    cannot accept tool definitions, so the tool loop can safely retry without
    tools.  Conservative on purpose: any unknown 4xx/5xx is re-raised.
    """
    msg = str(error).lower()
    return any(phrase in msg for phrase in _NO_TOOL_SUPPORT_PHRASES)

from ..metadata import Metadata, Usage
from ..utils.logger import logger

if TYPE_CHECKING:
    from ..message import Message

# Import tool types for type hints
from .mcp.tool import MCPTool
from .native.function import NativeFunctionTool
from .native.object import NativeObjectTool
from .skills.object import SkillObjectTool
from .skills.claude import ClaudeSkill


def _dict_to_metadata(meta: Optional[Dict[str, Any]]) -> Metadata:
    """Convert a streaming chunk metadata dict to a Metadata object.

    Chunk metadata is produced by provider converters as a plain dict;
    this helper converts it to the structured Metadata type expected by
    BaseMessage.metadata so that token usage survives into AIMessage.
    """
    if not meta:
        return Metadata()
    usage_dict = meta.get('usage')
    usage = Usage(**usage_dict) if usage_dict else None
    return Metadata(
        provider=meta.get('provider', ''),
        model=meta.get('model'),
        finish_reason=meta.get('finish_reason'),
        usage=usage,
    )


@dataclass
class ToolCallResult:
    """工具调用结果"""
    call_id: str
    content: str
    is_error: bool
    # Attachment objects extracted from the tool output (output-type, no reader).
    # These travel out-of-band from the LLM context so the frontend can render downloads.
    attachments: List[_Attachment] = field(default_factory=list)


@dataclass
class HITLRequest:
    """Human-in-the-loop handler request.

    Passed to HITLHandler before a tool is executed. The handler may pause
    execution for any duration (e.g. to interact with the user) and must
    return an HITLDecision.
    """
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str


@dataclass
class HITLDecision:
    """Decision returned by a HITLHandler.

    Use the class methods to construct — do not instantiate directly:

        HITLDecision.abort()
        HITLDecision.allow()
        HITLDecision.allow(overrides={"key": value})

    Attributes:
        action: "abort" or "allow"
        overrides: Dict of argument overrides to merge before tool execution.
                   Only meaningful when action == "allow".
    """
    action: str
    overrides: Dict[str, Any]

    @classmethod
    def abort(cls) -> "HITLDecision":
        """Abort the tool call entirely. The tool will not be executed."""
        return cls(action="abort", overrides={})

    @classmethod
    def allow(cls, overrides: Optional[Dict[str, Any]] = None) -> "HITLDecision":
        """Allow the tool call, optionally injecting argument overrides.

        Args:
            overrides: Dict of argument key/value pairs to merge into the
                       tool's arguments before execution. Pass None or omit
                       to allow without modification.
        """
        return cls(action="allow", overrides=overrides or {})

    @property
    def is_aborted(self) -> bool:
        """True if this decision aborts the tool call."""
        return self.action == "abort"

    @property
    def is_allowed(self) -> bool:
        """True if this decision allows the tool call."""
        return self.action == "allow"


# HITLHandler: async callable from HITLRequest -> HITLDecision.
HITLHandler = Callable[[HITLRequest], Awaitable[HITLDecision]]

# Sentinel: HITL not yet resolved, run internal handler.
_HITL_PENDING = object()

class ToolManager:
    """
    管理 LLM + 工具调用循环（通用工具管理器）
    
    支持任意类型的工具，只要实现了以下接口（鸭子类型）：
    - name: str - 工具名称
    - to_openai_tool() -> dict - 转换为 OpenAI 格式
    - call(arguments: dict) -> Any - 调用工具
    
    自动处理：
    1. 调用 LLM（带工具列表）
    2. 如果 LLM 返回 tool_calls → 执行工具 → 添加结果 → 回到步骤 1
    3. 如果 LLM 不调用工具 → 返回最终答案
    
    支持的工具类型：
    - MCPTool: MCP 协议工具
    - NativeFunctionTool: 原生 Python 函数
    - NativeObjectTool: 原生 Python 对象（多方法）
    - 任何实现了上述接口的自定义工具
    
    Example:
        from chak.tools.mcp import Server
        
        # 混合使用 MCP 工具和原生函数
        def my_func(x: int) -> int:
            return x * 2
        
        server = Server(url="...")
        mcp_tools = await server.tools()
        
        manager = ToolManager([my_func, *mcp_tools])
        response = await manager.execute_loop(
            provider=provider,
            messages=messages,
            model_uri="openai/gpt-4o"
        )
    """
    
    def __init__(self, tools: List[Union[MCPTool, NativeFunctionTool, NativeObjectTool, SkillObjectTool, ClaudeSkill]], max_iterations: int = 50, executor=None, hitl_handler: Optional["HITLHandler"] = None):
        """
        Args:
            tools: 工具列表（MCPTool、NativeFunctionTool、NativeObjectTool 或 SkillObjectTool）
            max_iterations: 最大迭代次数（防止无限循环），默认 50（每个 skill 调用消耗 2-3 轮）
            executor: 执行器实例（ThreadPoolExecutor/ProcessPoolExecutor）或 None（使用 asyncio）
            hitl_handler: Human-in-the-loop middleware called before each tool execution
        """
        self.tools = tools
        self._tool_map = self._build_tool_map()
        self._skill_map = self._build_skill_map()
        self._active_skill: Optional[SkillObjectTool] = None  # Track currently active skill instance
        self._selected_methods: List[str] = []  # Track methods selected by LLM in Stage 2
        self.max_iterations = max_iterations if max_iterations is not None else 50
        self.executor = executor
        self.hitl_handler = hitl_handler
    
    def _build_tool_map(self) -> Dict[str, Any]:
        """
        Build tool name -> tool object mapping for regular tools.
        
        For NativeObjectTool, expand all methods.
        Skip SkillObjectTool (handled separately).
        
        Returns:
            tool_map: tool name -> tool object
        """
        tool_map = {}
        
        for tool in self.tools:
            if isinstance(tool, SkillObjectTool):
                # Skip SkillObjectTool (handled in _build_skill_map)
                continue
            elif isinstance(tool, NativeObjectTool):
                # Expand object methods
                for method_name, method_tool in tool._method_tools.items():
                    tool_map[method_name] = method_tool
            elif isinstance(tool, ClaudeSkill):
                # Register both the skill and its companion file reader
                tool_map[tool.name] = tool
                file_reader = tool.get_file_reader()
                tool_map[file_reader.name] = file_reader
            else:
                # MCPTool or NativeFunctionTool
                tool_map[tool.name] = tool
        
        return tool_map
    
    def _build_skill_map(self) -> Dict[str, SkillObjectTool]:
        """
        Build skill name -> SkillObjectTool mapping.
        
        Returns:
            skill_map: skill name -> SkillObjectTool instance
        """
        skill_map = {}
        
        for tool in self.tools:
            if isinstance(tool, SkillObjectTool):
                skill_map[tool.name] = tool
        
        return skill_map
    
    def _get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Get current stage OpenAI tool list.
        
        Stage 1: Skill entry tools (no method param) + regular tools
        Stage 2: Skill entry tools (with method param enum) + regular tools
        Stage 3: Only selected methods as individual tools + regular tools
        
        Returns:
            OpenAI tool definition list
        """
        openai_tools = []
        
        if self._active_skill and self._selected_methods:
            # Stage 3: Expose only selected methods as individual tools
            logger.info(f"3️⃣ [Skill Stage 3] Exposing {len(self._selected_methods)} selected method(s): {', '.join(self._selected_methods)}")
            for method_name in self._selected_methods:
                try:
                    method_tool = self._active_skill.get_method_tool(method_name)
                    openai_tools.append(method_tool.to_openai_tool())
                except ValueError:
                    logger.warning(f"⚠️ [Skill] Selected method '{method_name}' not found, skipping")
            
            # Also include regular tools (non-skill tools)
            for tool in self.tools:
                if isinstance(tool, NativeObjectTool):
                    openai_tools.extend(tool.to_openai_tools())
                elif isinstance(tool, SkillObjectTool):
                    # Skip other skills when one is active
                    continue
                elif isinstance(tool, ClaudeSkill):
                    openai_tools.append(tool.to_openai_tool())
                    openai_tools.append(tool.get_file_reader().to_openai_tool())
                else:
                    openai_tools.append(tool.to_openai_tool())
        elif self._active_skill:
            # Stage 2: Skill is active but no methods selected yet, expose skill entry with method enum
            logger.info(f"2️⃣ [Skill Stage 2] Exposing skill '{self._active_skill.name}' with method selection (enum)")
            openai_tools.append(self._active_skill.to_skill_entry_tool(include_method_param=True))
            
            # Also include regular tools
            for tool in self.tools:
                if isinstance(tool, NativeObjectTool):
                    openai_tools.extend(tool.to_openai_tools())
                elif isinstance(tool, SkillObjectTool):
                    # Skip other skills when one is active
                    continue
                elif isinstance(tool, ClaudeSkill):
                    openai_tools.append(tool.to_openai_tool())
                    openai_tools.append(tool.get_file_reader().to_openai_tool())
                else:
                    openai_tools.append(tool.to_openai_tool())
        else:
            # Stage 1: No skill active, expose skill entry tools (no method param) + regular tools
            if self._skill_map:
                logger.info(f"1️⃣ [Skill Stage 1] Exposing {len(self._skill_map)} skill(s) to LLM")
            for skill_tool in self._skill_map.values():
                openai_tools.append(skill_tool.to_skill_entry_tool(include_method_param=False))
            
            for tool in self.tools:
                if isinstance(tool, NativeObjectTool):
                    openai_tools.extend(tool.to_openai_tools())
                elif isinstance(tool, SkillObjectTool):
                    # SkillObjectTool already handled above
                    continue
                elif isinstance(tool, ClaudeSkill):
                    openai_tools.append(tool.to_openai_tool())
                    openai_tools.append(tool.get_file_reader().to_openai_tool())
                else:
                    openai_tools.append(tool.to_openai_tool())
        
        logger.debug(f"📊 [Tool List] Returning {len(openai_tools)} tool(s)")
        return openai_tools
    
    async def execute_loop(
        self, 
        provider: Any,  # LLM Provider
        messages: List["Message"],
        model_uri: str
    ) -> tuple["Message", List["Message"]]:
        """
        Execute LLM + MCP tool calling loop (non-streaming).
        
        Flow:
        1. Call LLM with tools
        2. If LLM returns tool_calls -> execute tools -> add results to messages -> loop back to step 1
        3. If LLM returns final answer (no tool_calls) -> return
        
        Note: If model doesn't support function calling, gracefully fallback to non-tool mode.
        
        Args:
            provider: LLM Provider instance
            messages: Message list
            model_uri: Model URI (not used, provider already has model)
        
        Returns:
            tuple: (final_message, all_new_messages)
                - final_message: Final assistant response (after all tool calls completed)
                - all_new_messages: All messages added during this loop (including intermediate AIMessage+ToolMessage)
        
        Raises:
            Exception: Max iteration reached or other errors
        """
        from ..message import AIMessage, ToolMessage
        
        # Reset active skill and selected methods at the start of each conversation turn
        self._active_skill = None
        self._selected_methods = []
        
        current_messages = messages.copy()
        new_messages = []  # Track all new messages added during this loop
        all_attachments: List[_Attachment] = []  # Accumulate Attachment objects across all tool calls

        # Convert tools to OpenAI format
        openai_tools = self._get_openai_tools()
        
        for iteration in range(self.max_iterations):
            # Step 1: Call LLM with tools
            logger.debug(f"💬 [Tool Loop] Iteration {iteration}: Calling LLM with {len(openai_tools)} tools...")
            try:
                # Anthropic (and some other providers) reject tools=[] — only pass
                # the parameter when there is at least one tool to send.
                _send_kwargs: Dict[str, Any] = {"messages": current_messages, "stream": False}
                if openai_tools:
                    _send_kwargs["tools"] = openai_tools
                response = await asyncio.to_thread(provider.send, **_send_kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                # Check if model doesn't support function calling
                if _is_no_tool_support_error(e):
                    logger.warning(f"⚠️ [Tool] Model doesn't support function calling, gracefully degrading...")
                    logger.debug(f"📝 [Tool Loop] Error message: {str(e)}")
                    # Graceful degradation: call without tools and return
                    response = await asyncio.to_thread(
                        provider.send,
                        messages=current_messages,
                        stream=False
                    )
                    final_msg = AIMessage(
                        content=response.content if hasattr(response, 'content') else str(response),
                        metadata=getattr(response, 'metadata', Metadata()),
                        attachments=all_attachments,
                    )
                    new_messages.append(final_msg)
                    return final_msg, new_messages
                else:
                    # Other errors, re-raise
                    logger.error(f"❌ [Tool] LLM call failed: {str(e)}")
                    raise
            
            # Step 2: Check if LLM wants to call tools
            tool_calls = getattr(response, 'tool_calls', None)
            
            logger.debug(f"📊 [Tool Loop] Iteration {iteration}: tool_calls_count={len(tool_calls) if tool_calls else 0}")
            
            if not tool_calls:
                # No tool calls -> LLM finished, return final answer
                logger.info(f"ℹ️ [Tool] No tool calls in this iteration, LLM returned final answer")
                logger.debug(f"✅ [Tool Loop] No tool calls, finishing...")
                final_msg = AIMessage(
                    content=response.content if hasattr(response, 'content') else str(response),
                    metadata=getattr(response, 'metadata', Metadata()),
                    attachments=all_attachments,
                )
                new_messages.append(final_msg)
                return final_msg, new_messages
            
            logger.info(f"🔧 [Tool] LLM wants to call {len(tool_calls)} tool(s): {[tc.function.name for tc in tool_calls]}")
            logger.debug(f"📤 [Tool Loop] Calling tools: {[tc.function.name for tc in tool_calls]}")
            
            # Step 3: Execute tools in parallel
            logger.debug(f"⏳ [Tool Loop] Executing {len(tool_calls)} tools...")
            tool_results = await self._execute_tools_parallel(tool_calls)
            logger.debug(f"📥 [Tool Loop] Tool results: {[r.content[:50] + '...' if len(r.content) > 50 else r.content for r in tool_results]}")
            
            # Step 4: Add assistant message (with tool_calls) to conversation
            # Preserve reasoning_content so providers like DeepSeek (thinking mode)
            # can pass it back in the next round as required by their API.
            assistant_msg = AIMessage(
                content=response.content if hasattr(response, 'content') else "",
                tool_calls=tool_calls,
                reasoning_content=getattr(response, 'reasoning_content', None),
                metadata=getattr(response, 'metadata', Metadata())
            )
            current_messages.append(assistant_msg)
            new_messages.append(assistant_msg)
            
            # Step 5: Add tool results to conversation; accumulate attachment URLs.
            for result in tool_results:
                all_attachments.extend(result.attachments)
                tool_msg = ToolMessage(
                    content=result.content,
                    tool_call_id=result.call_id
                )
                current_messages.append(tool_msg)
                new_messages.append(tool_msg)
            
            # Step 6: Update tool list for next iteration if skill was activated
            if self._active_skill:
                # Skill has been activated, next iteration will use skill's internal tools
                openai_tools = self._get_openai_tools()
                logger.debug(f"🌟 [Skill] Updated tool list for activated skill: {self._active_skill.name}")
            
            logger.debug(f"🔁 [Tool Loop] Loop continues to iteration {iteration + 1}...")
            # Step 6: Loop back to step 1 (LLM will see tool results and decide next action)
        
        # Max iteration reached (possible infinite loop)
        raise Exception(
            f"Max tool call iterations ({self.max_iterations}) reached. "
            "The conversation may be stuck in a loop."
        )
    
    async def execute_loop_stream(self, provider: Any, messages: List["Message"], model_uri: str):
        """
        Execute LLM + MCP tool calling loop with streaming support.
        
        Flow:
        1. Call LLM with tools (streaming)
        2. Yield content chunks in real-time
        3. Accumulate tool_calls from delta
        4. If finish_reason == 'tool_calls' -> execute tools -> loop back to step 1
        5. If finish_reason == 'stop' -> done
        
        Args:
            provider: LLM Provider instance
            messages: Message list
            model_uri: Model URI
        
        Yields:
            tuple: (MessageChunk, all_new_messages_so_far)
                - MessageChunk: Streaming chunks
                - all_new_messages_so_far: All messages added during this loop (for conv.messages sync)
        
        Raises:
            Exception: Max iteration reached or other errors
        """
        from ..message import AIMessage, ToolMessage, MessageChunk
        
        # Reset active skill and selected methods at the start of each conversation turn
        self._active_skill = None
        self._selected_methods = []
        
        current_messages = messages.copy()
        new_messages = []  # Track all new messages added during this loop
        
        # Convert tools to OpenAI format (same as execute_loop)
        openai_tools = self._get_openai_tools()
        
        for iteration in range(self.max_iterations):
            accumulated_content = ""
            accumulated_reasoning_content = ""
            accumulated_tool_calls = []
            finish_reason = None
            last_metadata: Optional[Dict] = None
            try:
                # Get stream iterator synchronously in thread
                def _get_stream():
                    # Anthropic rejects tools=[] — only include when non-empty
                    _kwargs: Dict[str, Any] = {"messages": current_messages, "stream": True}
                    if openai_tools:
                        _kwargs["tools"] = openai_tools
                    return provider.send(**_kwargs)
                
                stream = await asyncio.to_thread(_get_stream)
            except Exception as e:
                error_msg = str(e).lower()
                if _is_no_tool_support_error(e):
                    logger.warning(f"⚠️ [Tool] Model doesn't support function calling, gracefully degrading to streaming mode...")
                    logger.debug(f"📝 [Tool Loop] Error message: {str(e)}")
                    # Graceful degradation: streaming without tools
                    def _get_fallback_stream():
                        return provider.send(
                            messages=current_messages,
                            stream=True
                        )
                    stream = await asyncio.to_thread(_get_fallback_stream)
                    
                    # Yield all chunks from the fallback stream (using converter)
                    fallback_content = ""
                    for provider_chunk in stream:
                        unified_chunk = provider.converter.from_provider_chunk(provider_chunk)
                        if unified_chunk.content:
                            fallback_content += unified_chunk.content
                            yield MessageChunk(
                                content=unified_chunk.content,
                                is_final=False
                            ), new_messages
                    
                    # Construct final message with accumulated fallback content
                    final_msg = AIMessage(content=fallback_content)
                    new_messages.append(final_msg)
                    
                    yield MessageChunk(content="", is_final=True, final_message=final_msg), new_messages
                    return
                else:
                    logger.error(f"❌ [Tool] LLM call failed: {str(e)}")
                    raise
            
            # Step 2: Process streaming chunks (Manager only handles UnifiedStreamChunk)
            for provider_chunk in stream:
                # Convert provider chunk to unified format
                unified_chunk = provider.converter.from_provider_chunk(provider_chunk)
                
                # Accumulate reasoning_content (required by DeepSeek thinking mode on next round)
                if unified_chunk.reasoning_content:
                    accumulated_reasoning_content += unified_chunk.reasoning_content

                # Handle regular content
                if unified_chunk.content:
                    accumulated_content += unified_chunk.content
                    yield MessageChunk(
                        content=unified_chunk.content,
                        is_final=False,
                        metadata={"iteration": iteration}
                    ), new_messages
                
                # Handle tool_calls delta (accumulate)
                for tc_delta in unified_chunk.tool_calls_delta:
                    index = tc_delta.index
                    
                    # Ensure list is large enough
                    while len(accumulated_tool_calls) <= index:
                        accumulated_tool_calls.append({
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        })
                    
                    # Update accumulated tool call (incremental)
                    if tc_delta.id:
                        accumulated_tool_calls[index]["id"] = tc_delta.id
                    if tc_delta.type:
                        accumulated_tool_calls[index]["type"] = tc_delta.type
                    if tc_delta.function_name:
                        accumulated_tool_calls[index]["function"]["name"] = tc_delta.function_name
                    if tc_delta.function_arguments:
                        accumulated_tool_calls[index]["function"]["arguments"] += tc_delta.function_arguments
                
                # Update finish_reason if present
                if unified_chunk.finish_reason:
                    finish_reason = unified_chunk.finish_reason
                
                # Track last metadata (usage arrives in the final chunk)
                if unified_chunk.metadata:
                    last_metadata = unified_chunk.metadata
            
            # Step 3: Check finish_reason
            logger.debug(f"📊 [Tool Loop] Iteration {iteration}: finish_reason={finish_reason}, tool_calls_count={len(accumulated_tool_calls)}")
            
            if finish_reason == "tool_calls" and accumulated_tool_calls:
                logger.info(f"🔧 [Tool] LLM wants to call {len(accumulated_tool_calls)} tool(s): {[tc['function']['name'] for tc in accumulated_tool_calls]}")
                logger.debug(f"📤 [Tool Loop] Calling tools: {[tc['function']['name'] for tc in accumulated_tool_calls]}")
                
                # Convert to proper tool_call objects
                from ..message import ChatCompletionMessageToolCall, Function
                tool_calls_objects = [
                    ChatCompletionMessageToolCall(
                        id=tc["id"],
                        type="function",
                        function=Function(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"]
                        )
                    )
                    for tc in accumulated_tool_calls
                ]
                
                # Execute tools
                logger.debug(f"⏳ [Tool Loop] Executing {len(tool_calls_objects)} tools...")
                tool_results = await self._execute_tools_parallel(tool_calls_objects)
                logger.debug(f"📥 [Tool Loop] Tool results: {[r.content[:50] + '...' if len(r.content) > 50 else r.content for r in tool_results]}")
                
                # Add assistant message (with tool_calls), preserving LLM usage metadata
                # Also preserve reasoning_content for providers like DeepSeek (thinking mode)
                # that require it to be passed back in the next round.
                assistant_msg = AIMessage(
                    content=accumulated_content,
                    tool_calls=tool_calls_objects,
                    reasoning_content=accumulated_reasoning_content or None,
                    metadata=_dict_to_metadata(last_metadata)
                )
                current_messages.append(assistant_msg)
                new_messages.append(assistant_msg)
                
                # Add tool results
                for result in tool_results:
                    tool_msg = ToolMessage(
                        content=result.content,
                        tool_call_id=result.call_id
                    )
                    current_messages.append(tool_msg)
                    new_messages.append(tool_msg)
                
                # Update tool list for next iteration if skill was activated
                if self._active_skill:
                    openai_tools = self._get_openai_tools()
                    logger.debug(f"🌟 [Skill] Updated tool list for activated skill: {self._active_skill.name}")
                
                logger.debug(f"🔁 [Tool Loop] Loop continues to iteration {iteration + 1}...")
                # Loop continues (next iteration will call LLM with tool results)
            else:
                # No tool calls or finish_reason != 'tool_calls' -> done
                logger.info(f"ℹ️ [Tool] No tool calls in this iteration, LLM returned final answer")
                logger.debug(f"✅ [Tool Loop] No tool calls, finishing...")
                
                # Add final AIMessage to new_messages, preserving LLM usage metadata
                final_msg = AIMessage(
                    content=accumulated_content,
                    metadata=_dict_to_metadata(last_metadata)
                )
                new_messages.append(final_msg)
                
                yield MessageChunk(content="", is_final=True, final_message=final_msg), new_messages
                return
        
        # Max iteration reached
        raise Exception(
            f"Max tool call iterations ({self.max_iterations}) reached. "
            "The conversation may be stuck in a loop."
        )
    
    async def execute_loop_with_events(
        self,
        provider: Any,
        messages: List["Message"],
        model_uri: str
    ):
        """
        Execute LLM + tool calling loop with event stream support.
        
        This method provides complete observability by yielding events for:
        - LLM content generation (MessageChunk)
        - Tool call initiation (ToolCallStartEvent)
        - Tool call completion (ToolCallEndEvent)
        
        Flow:
        1. Call LLM with tools (streaming)
        2. Yield MessageChunk chunks in real-time
        3. Accumulate tool_calls from delta
        4. If finish_reason == 'tool_calls':
           - Yield ToolCallStartEvent for each tool
           - Execute tools in parallel
           - Yield ToolCallEndEvent for each result
           - Loop back to step 1
        5. If finish_reason == 'stop' -> yield final MessageChunk and done
        
        Args:
            provider: LLM Provider instance
            messages: Message list
            model_uri: Model URI
        
        Yields:
            StreamEvent: MessageChunk, ToolCallStartEvent, ToolCallSuccessEvent, ToolCallErrorEvent, or ConversationCompleteEvent
        
        Raises:
            Exception: Max iteration reached or other errors
        """
        from ..message import AIMessage, ToolMessage, MessageChunk, ReasoningChunk, ToolCallStartEvent, ToolCallSuccessEvent, ToolCallErrorEvent, ToolCallCancelledEvent, ConversationCompleteEvent, ChatCompletionMessageToolCall, Function
        import json
        
        # Reset active skill and selected methods at the start of each conversation turn
        self._active_skill = None
        self._selected_methods = []
        
        current_messages = messages.copy()
        new_messages = []  # Track all messages created during this turn
        all_attachments: List[_Attachment] = []  # Accumulate Attachment objects across all tool calls
        
        # Convert tools to OpenAI format (same as execute_loop)
        openai_tools = self._get_openai_tools()
        
        for iteration in range(self.max_iterations):
            accumulated_content = ""
            accumulated_reasoning_content = ""
            accumulated_tool_calls = []
            finish_reason = None
            last_metadata: Optional[Dict] = None
            
            # Step 1: Call LLM with streaming
            logger.debug(f"💬 [Tool Loop] Iteration {iteration}: Calling LLM with {len(openai_tools)} tools (streaming with events)...")
            try:
                # Get stream iterator synchronously in thread
                def _get_stream():
                    # Anthropic rejects tools=[] — only include when non-empty
                    _kwargs: Dict[str, Any] = {"messages": current_messages, "stream": True}
                    if openai_tools:
                        _kwargs["tools"] = openai_tools
                    return provider.send(**_kwargs)
                
                stream = await asyncio.to_thread(_get_stream)
            except Exception as e:
                error_msg = str(e).lower()
                if _is_no_tool_support_error(e):
                    logger.warning(f"⚠️ [Tool] Model doesn't support function calling, gracefully degrading to event streaming mode...")
                    logger.debug(f"📝 [Tool Loop] Error message: {str(e)}")
                    # Graceful degradation: streaming without tools
                    def _get_fallback_stream():
                        return provider.send(
                            messages=current_messages,
                            stream=True
                        )
                    stream = await asyncio.to_thread(_get_fallback_stream)
                    
                    # Yield all content events from the fallback stream
                    fallback_content = ""
                    for provider_chunk in stream:
                        # Use converter to handle different provider formats
                        chunk = provider.converter.from_provider_chunk(provider_chunk)
                        if isinstance(chunk, MessageChunk) and chunk.content:
                            fallback_content += chunk.content
                            yield chunk
                    
                    # Construct final message with accumulated content
                    final_msg = AIMessage(content=fallback_content)
                    current_messages.append(final_msg)
                    
                    yield MessageChunk(content="", is_final=True, final_message=final_msg)
                    return
                else:
                    logger.error(f"❌ [Tool] LLM call failed: {str(e)}")
                    raise
            
            # Step 2: Process streaming chunks (Manager only handles UnifiedStreamChunk)
            for provider_chunk in stream:
                # Convert provider chunk to unified format
                unified_chunk = provider.converter.from_provider_chunk(provider_chunk)
                
                # Handle reasoning content
                if unified_chunk.reasoning_content:
                    # Accumulate for pass-back (DeepSeek thinking mode requirement)
                    accumulated_reasoning_content += unified_chunk.reasoning_content
                    yield ReasoningChunk(
                        content=unified_chunk.reasoning_content,
                        is_final=False,
                        metadata=unified_chunk.metadata
                    )
                
                # Handle regular content
                if unified_chunk.content:
                    accumulated_content += unified_chunk.content
                    yield MessageChunk(
                        content=unified_chunk.content,
                        is_final=False
                    )
                
                # Handle tool_calls delta (accumulate)
                for tc_delta in unified_chunk.tool_calls_delta:
                    index = tc_delta.index
                    
                    # Ensure list is large enough
                    while len(accumulated_tool_calls) <= index:
                        accumulated_tool_calls.append({
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        })
                    
                    # Update accumulated tool call (incremental)
                    if tc_delta.id:
                        accumulated_tool_calls[index]["id"] = tc_delta.id
                    if tc_delta.type:
                        accumulated_tool_calls[index]["type"] = tc_delta.type
                    if tc_delta.function_name:
                        accumulated_tool_calls[index]["function"]["name"] = tc_delta.function_name
                    if tc_delta.function_arguments:
                        accumulated_tool_calls[index]["function"]["arguments"] += tc_delta.function_arguments
                
                # Update finish_reason if present
                if unified_chunk.finish_reason:
                    finish_reason = unified_chunk.finish_reason
                
                # Track last metadata (usage arrives in the final chunk)
                if unified_chunk.metadata:
                    last_metadata = unified_chunk.metadata
                
                # Check if final
                if unified_chunk.is_final:
                    finish_reason = finish_reason or "stop"
            
            # Step 3: Check finish_reason
            logger.debug(f"📊 [Tool Loop] Iteration {iteration}: finish_reason={finish_reason}, tool_calls_count={len(accumulated_tool_calls)}")
            
            if finish_reason == "tool_calls" and accumulated_tool_calls:
                logger.info(f"🔧 [Tool] LLM wants to call {len(accumulated_tool_calls)} tool(s): {[tc['function']['name'] for tc in accumulated_tool_calls]}")
                logger.debug(f"📤 [Tool Loop] Calling tools: {[tc['function']['name'] for tc in accumulated_tool_calls]}")
                
                # Convert to proper tool_call objects
                tool_calls_objects = [
                    ChatCompletionMessageToolCall(
                        id=tc["id"],
                        type="function",
                        function=Function(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"]
                        )
                    )
                    for tc in accumulated_tool_calls
                ]
                
                # Step 3b: Pre-resolve HITL for all tools (before emitting any events)
                # This ensures ToolCallStartEvent is only emitted for approved tools.
                responses_map: Dict[str, HITLDecision] = {}
                if self.hitl_handler is not None:
                    hitl_tasks = []
                    for tc in tool_calls_objects:
                        try:
                            tc_args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            tc_args = {"raw": tc.function.arguments}
                        request = HITLRequest(
                            tool_name=tc.function.name,
                            arguments=tc_args,
                            call_id=tc.id,
                        )
                        hitl_tasks.append(self.hitl_handler(request))
                    hitl_results = await asyncio.gather(*hitl_tasks)
                    for tc, result in zip(tool_calls_objects, hitl_results):
                        responses_map[tc.id] = result
                else:
                    # No HITL handler: all tools auto-approved with no overrides
                    for tc in tool_calls_objects:
                        responses_map[tc.id] = HITLDecision.allow()

                # Emit events per tool: ToolCallCancelledEvent or ToolCallStartEvent
                approved_tool_calls = []
                cancelled_results: List[ToolCallResult] = []
                for tc in tool_calls_objects:
                    try:
                        args_dict = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args_dict = {"raw": tc.function.arguments}

                    hitl_resp = responses_map[tc.id]
                    if hitl_resp.is_aborted:
                        # Cancelled by HITL handler
                        logger.info(f"ℹ️ [Tool] Tool call '{tc.function.name}' cancelled by HITL handler (events mode)")
                        yield ToolCallCancelledEvent(
                            tool_name=tc.function.name,
                            call_id=tc.id,
                        )
                        cancelled_results.append(ToolCallResult(
                            call_id=tc.id,
                            content=(
                                "Tool call was cancelled by user. "
                                "Do NOT call this tool or any other tools again. "
                                "Answer the user's request directly based on the existing "
                                "conversation without using tools."
                            ),
                            is_error=True,
                        ))
                    else:
                        # Approved — emit start event with (possibly overridden) args
                        effective_args = {**args_dict, **hitl_resp.overrides}
                        yield ToolCallStartEvent(
                            tool_name=tc.function.name,
                            call_id=tc.id,
                            arguments=effective_args,
                        )
                        approved_tool_calls.append(tc)

                # Execute approved tools in parallel (HITL already resolved)
                logger.debug(f"⏳ [Tool Loop] Executing {len(approved_tool_calls)} approved tool(s)...")
                executed_results = await self._execute_tools_parallel(
                    approved_tool_calls,
                    overrides_map={tc.id: responses_map[tc.id].overrides for tc in approved_tool_calls},
                )
                tool_results = cancelled_results + executed_results
                logger.debug(f"📥 [Tool Loop] Tool results: {[r.content[:50] + '...' if len(r.content) > 50 else r.content for r in tool_results]}")

                # Yield ToolCallSuccessEvent or ToolCallErrorEvent for executed tools only
                for result in executed_results:
                    tool_name_for_event = "unknown"
                    for tc in approved_tool_calls:
                        if tc.id == result.call_id:
                            tool_name_for_event = tc.function.name
                            break

                    if result.is_error:
                        yield ToolCallErrorEvent(
                            tool_name=tool_name_for_event,
                            call_id=result.call_id,
                            error=result.content,
                        )
                    else:
                        all_attachments.extend(result.attachments)
                        yield ToolCallSuccessEvent(
                            tool_name=tool_name_for_event,
                            call_id=result.call_id,
                            result=result.content,
                            attachments=result.attachments,
                        )

                # Add assistant message (with tool_calls), preserving LLM usage metadata
                # Also preserve reasoning_content for providers like DeepSeek (thinking mode)
                # that require it to be passed back in the next round.
                assistant_msg = AIMessage(
                    content=accumulated_content,
                    tool_calls=tool_calls_objects,
                    reasoning_content=accumulated_reasoning_content or None,
                    metadata=_dict_to_metadata(last_metadata)
                )
                current_messages.append(assistant_msg)
                new_messages.append(assistant_msg)
                
                # Add tool results
                for result in tool_results:
                    tool_msg = ToolMessage(
                        content=result.content,
                        tool_call_id=result.call_id
                    )
                    current_messages.append(tool_msg)
                    new_messages.append(tool_msg)
                
                # Update tool list for next iteration if skill was activated
                if self._active_skill:
                    openai_tools = self._get_openai_tools()
                    logger.debug(f"🌟 [Skill] Updated tool list for activated skill: {self._active_skill.name}")
                
                logger.debug(f"🔁 [Tool Loop] Loop continues to iteration {iteration + 1}...")
                # Loop continues (next iteration will call LLM with tool results)
            else:
                # No tool calls or finish_reason != 'tool_calls' -> done
                logger.info(f"ℹ️ [Tool] No tool calls in this iteration, LLM returned final answer")
                logger.debug(f"✅ [Tool Loop] No tool calls, finishing...")
                
                # Construct final AIMessage from accumulated content, preserving LLM usage metadata
                final_message = AIMessage(
                    content=accumulated_content,
                    metadata=_dict_to_metadata(last_metadata),
                    attachments=all_attachments,
                )
                new_messages.append(final_message)
                
                yield MessageChunk(content="", is_final=True, final_message=final_message, attachments=all_attachments)
                yield ConversationCompleteEvent(messages=new_messages)
                return
        
        # Max iteration reached
        raise Exception(
            f"Max tool call iterations ({self.max_iterations}) reached. "
            "The conversation may be stuck in a loop."
        )
    
    async def _execute_tools_parallel(self, tool_calls: List[Any], overrides_map: Optional[Dict[str, Optional[Dict[str, Any]]]] = None) -> List[ToolCallResult]:
        """
        并行执行多个工具调用

        Args:
            tool_calls: 工具调用列表
            overrides_map: Optional pre-resolved HITL results keyed by call_id.
                           If provided, _execute_single_tool skips its internal HITL.

        Returns:
            工具结果列表
        """
        tasks = [
            self._execute_single_tool(
                call,
                pre_hitl_overrides=overrides_map.get(call.id, _HITL_PENDING) if overrides_map is not None else _HITL_PENDING,
            )
            for call in tool_calls
        ]
        return await asyncio.gather(*tasks)

    async def _execute_single_tool(self, tool_call: Any, pre_hitl_overrides: Any = _HITL_PENDING) -> ToolCallResult:
        """
        执行单个工具调用
        
        Args:
            tool_call: 工具调用对象（包含 name, arguments, id）
        
        Returns:
            ToolCallResult: 工具执行结果
        """
        # 从 tool_map 中查找工具
        tool_name = tool_call.function.name if hasattr(tool_call, 'function') else tool_call.name
        call_id = tool_call.id
        arguments = tool_call.function.arguments if hasattr(tool_call, 'function') else tool_call.arguments
        
        logger.info(f"🔧 [Tool] Calling tool: {tool_name}")
        logger.debug(f"📨 [Tool] Tool call ID: {call_id}")
        logger.debug(f"📨 [Tool] Arguments: {arguments}")
        
        # 解析 arguments（可能是 JSON 字符串）
        if isinstance(arguments, str):
            import json
            try:
                arguments = json.loads(arguments)
                logger.debug(f"🔄 [Tool] Parsed arguments: {arguments}")
            except json.JSONDecodeError:
                logger.error(f"❌ [Tool] Invalid JSON arguments: {arguments}")
                return ToolCallResult(
                    call_id=call_id,
                    content=f"Error: Invalid JSON arguments: {arguments}",
                    is_error=True
                )
        
        # HITL interception
        if pre_hitl_overrides is _HITL_PENDING:
            # Not pre-resolved: run internal handler (execute_loop / execute_loop_stream paths)
            if self.hitl_handler is not None:
                request = HITLRequest(
                    tool_name=tool_name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                    call_id=call_id,
                )
                try:
                    response = await self.hitl_handler(request)
                except Exception as e:
                    logger.error(f"❌ [Tool] HITL handler failed for tool '{tool_name}': {e}")
                    return ToolCallResult(
                        call_id=call_id,
                        content=f"Error: HITL handler failed: {e}",
                        is_error=True,
                    )
                if response.is_aborted:
                    logger.info(f"ℹ️ [Tool] Tool call '{tool_name}' cancelled by HITL handler")
                    return ToolCallResult(
                        call_id=call_id,
                        content=(
                            "Tool call was cancelled by user. "
                            "Do NOT call this tool or any other tools again. "
                            "Answer the user's request directly based on the existing "
                            "conversation without using tools."
                        ),
                        is_error=True,
                    )
                if response.overrides:
                    arguments = {**arguments, **response.overrides}
        else:
            # Pre-resolved by execute_loop_with_events: apply overrides if any
            if pre_hitl_overrides:
                arguments = {**arguments, **pre_hitl_overrides}
        
        # Check if this is a skill entry call
        if tool_name in self._skill_map:
            skill_tool = self._skill_map[tool_name]
            method_name = arguments.get('method', None)
            
            if method_name:
                # Stage 2: LLM selected a method, record it and prepare Stage 3
                logger.info(f"2️⃣ [Skill Stage 2] Method '{method_name}' selected for skill '{tool_name}'")
                self._active_skill = skill_tool
                
                # Add to selected methods list (avoid duplicates)
                if method_name not in self._selected_methods:
                    self._selected_methods.append(method_name)
                
                # Return confirmation to LLM
                return ToolCallResult(
                    call_id=call_id,
                    content=f"Method '{method_name}' selected and ready to use. You can now call it directly.",
                    is_error=False
                )
            else:
                # Stage 1: Activate skill and return method summary
                logger.info(f"1️⃣ [Skill Stage 1] Activating skill '{tool_name}', returning method summary")
                self._active_skill = skill_tool
                
                # Generate method summary
                instruction = arguments.get('instruction', None)
                method_summary = skill_tool.generate_method_summary(instruction)
                
                logger.debug(f"📋 [Skill] Generated summary for {len(skill_tool.method_names)} methods")
                
                # Return method summary to LLM
                return ToolCallResult(
                    call_id=call_id,
                    content=method_summary,
                    is_error=False
                )
        
        # Stage 3: If a skill is active and methods selected, look up from selected methods
        if self._active_skill and self._selected_methods:
            if tool_name in self._selected_methods:
                try:
                    method_tool = self._active_skill.get_method_tool(tool_name)
                    tool = method_tool
                    logger.info(f"3️⃣ [Skill Stage 3] Calling selected method '{tool_name}'")
                except ValueError:
                    # Method not found, fall back to regular tool_map
                    tool = self._tool_map.get(tool_name)
            else:
                # Not a selected method, check regular tools
                tool = self._tool_map.get(tool_name)
        else:
            # No active skill, use regular tool_map
            tool = self._tool_map.get(tool_name)
        
        if not tool:
            logger.error(f"❌ [Tool] Tool not found: {tool_name}")
            return ToolCallResult(
                call_id=call_id,
                content=f"Error: Tool not found: {tool_name}",
                is_error=True
            )
        
        try:
            # 调用工具（支持 MCPTool 和 NativeFunctionTool）
            logger.debug(f"⚙️  [Tool] Executing tool: {tool_name}...")
            result = await tool.call(arguments, executor=self.executor)
        
            # Extract result content
            if hasattr(result, 'content'):
                content = str(result.content)
            elif isinstance(result, dict):
                import json
                content = json.dumps(result, ensure_ascii=False)
            else:
                content = str(result)

            # Extract attachment:// URLs from the tool output before the content
            # is fed to the LLM.  The real URLs are preserved as typed Attachment objects
            # and travel out-of-band to the caller via ToolCallSuccessEvent / MessageChunk.
            # The LLM still sees the full content so its response reads naturally; the
            # frontend relies on the attachments field, not on parsing the LLM text.
            attachments: List[_Attachment] = []
            for m in _ATTACHMENT_RE.finditer(content):
                attachments.append(_url_to_attachment(m.group(1)))

            logger.info(f"✅ [Tool] Tool '{tool_name}' succeeded")
            logger.debug(f"📦 [Tool] Result: {content[:200]}..." if len(content) > 200 else f"📦 [Tool] Result: {content}")

            return ToolCallResult(
                call_id=call_id,
                content=content,
                is_error=False,
                attachments=attachments,
            )
        except Exception as e:
            logger.error(f"❌ [Tool] Tool '{tool_name}' failed: {str(e)}")
            return ToolCallResult(
                call_id=call_id,
                content=f"Error: {str(e)}",
                is_error=True
            )
