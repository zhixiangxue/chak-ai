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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Union

from loguru import logger

if TYPE_CHECKING:
    from ..message import Message

# Import tool types for type hints
from .mcp.tool import MCPTool
from .native.function import NativeFunctionTool
from .native.object import NativeObjectTool
from .skills.object import SkillObjectTool


@dataclass
class ToolCallResult:
    """工具调用结果"""
    call_id: str
    content: str
    is_error: bool


@dataclass
class ToolCallApproval:
    """Tool call approval request info."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str


ToolApprovalHandler = Callable[[ToolCallApproval], Awaitable[bool]]

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
    
    def __init__(self, tools: List[Union[MCPTool, NativeFunctionTool, NativeObjectTool, SkillObjectTool]], max_iterations: int = 10, executor=None, approval_handler: Optional[ToolApprovalHandler] = None):
        """
        Args:
            tools: 工具列表（MCPTool、NativeFunctionTool、NativeObjectTool 或 SkillObjectTool）
            max_iterations: 最大迭代次数（防止无限循环）
            executor: 执行器实例（ThreadPoolExecutor/ProcessPoolExecutor）或 None（使用 asyncio）
            approval_handler: Optional approval handler for human-in-the-loop tool calls
        """
        self.tools = tools
        self._tool_map = self._build_tool_map()
        self._skill_map = self._build_skill_map()
        self._active_skill: Optional[SkillObjectTool] = None  # Track currently active skill instance
        self.max_iterations = max_iterations if max_iterations is not None else 10
        self.executor = executor
        self.approval_handler = approval_handler
    
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
        
        If a skill is active, return that skill's internal tools.
        Otherwise return skill entry tools + regular tools.
        
        Returns:
            OpenAI tool definition list
        """
        openai_tools = []
        
        if self._active_skill:
            # Stage 2: Return active skill's internal tools only
            openai_tools.extend(self._active_skill.to_openai_tools())
        else:
            # Stage 1: Skill entry tools + regular tools
            for skill_tool in self._skill_map.values():
                openai_tools.append(skill_tool.to_skill_entry_tool())
            
            for tool in self.tools:
                if isinstance(tool, NativeObjectTool):
                    openai_tools.extend(tool.to_openai_tools())
                elif isinstance(tool, SkillObjectTool):
                    # SkillObjectTool already handled above
                    continue
                else:
                    openai_tools.append(tool.to_openai_tool())
        
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
        
        # Reset active skill at the start of each conversation turn
        self._active_skill = None
        
        current_messages = messages.copy()
        new_messages = []  # Track all new messages added during this loop
        
        # Convert tools to OpenAI format
        openai_tools = self._get_openai_tools()
        
        for iteration in range(self.max_iterations):
            # Step 1: Call LLM with tools
            logger.debug(f"💬 [Tool Loop] Iteration {iteration}: Calling LLM with {len(openai_tools)} tools...")
            try:
                response = await asyncio.to_thread(
                    provider.send,
                    messages=current_messages,
                    tools=openai_tools,
                    stream=False
                )
            except Exception as e:
                error_msg = str(e).lower()
                # Check if model doesn't support function calling
                if any(keyword in error_msg for keyword in ['tool', 'function', 'not support', 'invalid']):
                    logger.warning(f"⚠️ [Tool] Model doesn't support function calling, gracefully degrading...")
                    logger.debug(f"📝 [Tool Loop] Error message: {str(e)}")
                    # Graceful degradation: call without tools and return
                    response = await asyncio.to_thread(
                        provider.send,
                        messages=current_messages,
                        stream=False
                    )
                    final_msg = AIMessage(
                        content=response.content if hasattr(response, 'content') else str(response)
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
                    content=response.content if hasattr(response, 'content') else str(response)
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
            assistant_msg = AIMessage(
                content=response.content if hasattr(response, 'content') else "",
                tool_calls=tool_calls
            )
            current_messages.append(assistant_msg)
            new_messages.append(assistant_msg)
            
            # Step 5: Add tool results to conversation
            for result in tool_results:
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
        
        current_messages = messages.copy()
        new_messages = []  # Track all new messages added during this loop
        openai_tools = []
        for tool in self.tools:
            if isinstance(tool, NativeObjectTool):
                openai_tools.extend(tool.to_openai_tools())
            else:
                openai_tools.append(tool.to_openai_tool())
        
        for iteration in range(self.max_iterations):
            accumulated_content = ""
            accumulated_tool_calls = []
            finish_reason = None
            
            # Step 1: Call LLM with streaming
            logger.debug(f"💬 [Tool Loop] Iteration {iteration}: Calling LLM with {len(openai_tools)} tools (streaming)...")
            try:
                # Get stream iterator synchronously in thread
                def _get_stream():
                    return provider.send(
                        messages=current_messages,
                        tools=openai_tools,
                        stream=True
                    )
                
                stream = await asyncio.to_thread(_get_stream)
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['tool', 'function', 'not support', 'invalid']):
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
                    for provider_chunk in stream:
                        unified_chunk = provider.converter.from_provider_chunk(provider_chunk)
                        if unified_chunk.content:
                            yield MessageChunk(
                                content=unified_chunk.content,
                                is_final=False
                            ), new_messages
                    
                    yield MessageChunk(content="", is_final=True), new_messages
                    return
                else:
                    logger.error(f"❌ [Tool] LLM call failed: {str(e)}")
                    raise
            
            # Step 2: Process streaming chunks (Manager only handles UnifiedStreamChunk)
            for provider_chunk in stream:
                # Convert provider chunk to unified format
                unified_chunk = provider.converter.from_provider_chunk(provider_chunk)
                
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
                
                # Add assistant message (with tool_calls)
                assistant_msg = AIMessage(
                    content=accumulated_content,
                    tool_calls=tool_calls_objects
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
                
                logger.debug(f"🔁 [Tool Loop] Loop continues to iteration {iteration + 1}...")
                # Loop continues (next iteration will call LLM with tool results)
            else:
                # No tool calls or finish_reason != 'tool_calls' -> done
                logger.info(f"ℹ️ [Tool] No tool calls in this iteration, LLM returned final answer")
                logger.debug(f"✅ [Tool Loop] No tool calls, finishing...")
                
                # Add final AIMessage to new_messages
                if accumulated_content:
                    final_msg = AIMessage(content=accumulated_content)
                    new_messages.append(final_msg)
                
                yield MessageChunk(content="", is_final=True), new_messages
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
            StreamEvent: MessageChunk, ToolCallStartEvent, ToolCallSuccessEvent, or ToolCallErrorEvent
        
        Raises:
            Exception: Max iteration reached or other errors
        """
        from ..message import AIMessage, ToolMessage, MessageChunk, ReasoningChunk, ToolCallStartEvent, ToolCallSuccessEvent, ToolCallErrorEvent, ChatCompletionMessageToolCall, Function
        import json
        
        current_messages = messages.copy()
        openai_tools = []
        for tool in self.tools:
            if isinstance(tool, NativeObjectTool):
                openai_tools.extend(tool.to_openai_tools())
            else:
                openai_tools.append(tool.to_openai_tool())
        
        for iteration in range(self.max_iterations):
            accumulated_content = ""
            accumulated_tool_calls = []
            finish_reason = None
            
            # Step 1: Call LLM with streaming
            logger.debug(f"💬 [Tool Loop] Iteration {iteration}: Calling LLM with {len(openai_tools)} tools (streaming with events)...")
            try:
                # Get stream iterator synchronously in thread
                def _get_stream():
                    return provider.send(
                        messages=current_messages,
                        tools=openai_tools,
                        stream=True
                    )
                
                stream = await asyncio.to_thread(_get_stream)
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['tool', 'function', 'not support', 'invalid']):
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
                    for provider_chunk in stream:
                        # Use converter to handle different provider formats
                        chunk = provider.converter.from_provider_chunk(provider_chunk)
                        if isinstance(chunk, MessageChunk) and chunk.content:
                            yield chunk
                    
                    yield MessageChunk(content="", is_final=True)
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
                
                # Yield ToolCallStartEvent for each tool
                for tc in tool_calls_objects:
                    try:
                        args_dict = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args_dict = {"raw": tc.function.arguments}
                    
                    yield ToolCallStartEvent(
                        tool_name=tc.function.name,
                        call_id=tc.id,
                        arguments=args_dict
                    )
                
                # Execute tools in parallel
                logger.debug(f"⏳ [Tool Loop] Executing {len(tool_calls_objects)} tools...")
                tool_results = await self._execute_tools_parallel(tool_calls_objects)
                logger.debug(f"📥 [Tool Loop] Tool results: {[r.content[:50] + '...' if len(r.content) > 50 else r.content for r in tool_results]}")
                
                # Yield ToolCallSuccessEvent or ToolCallErrorEvent for each result
                for result in tool_results:
                    # Find tool name by call_id
                    tool_name = "unknown"
                    for tc in tool_calls_objects:
                        if tc.id == result.call_id:
                            tool_name = tc.function.name
                            break
                    
                    if result.is_error:
                        yield ToolCallErrorEvent(
                            tool_name=tool_name,
                            call_id=result.call_id,
                            error=result.content
                        )
                    else:
                        yield ToolCallSuccessEvent(
                            tool_name=tool_name,
                            call_id=result.call_id,
                            result=result.content
                        )
                
                # Add assistant message (with tool_calls)
                current_messages.append(AIMessage(
                    content=accumulated_content,
                    tool_calls=tool_calls_objects
                ))
                
                # Add tool results
                for result in tool_results:
                    current_messages.append(ToolMessage(
                        content=result.content,
                        tool_call_id=result.call_id
                    ))
                
                logger.debug(f"🔁 [Tool Loop] Loop continues to iteration {iteration + 1}...")
                # Loop continues (next iteration will call LLM with tool results)
            else:
                # No tool calls or finish_reason != 'tool_calls' -> done
                logger.info(f"ℹ️ [Tool] No tool calls in this iteration, LLM returned final answer")
                logger.debug(f"✅ [Tool Loop] No tool calls, finishing...")
                yield MessageChunk(content="", is_final=True)
                return
        
        # Max iteration reached
        raise Exception(
            f"Max tool call iterations ({self.max_iterations}) reached. "
            "The conversation may be stuck in a loop."
        )
    
    async def _execute_tools_parallel(self, tool_calls: List[Any]) -> List[ToolCallResult]:
        """
        并行执行多个工具调用
        
        Args:
            tool_calls: 工具调用列表
        
        Returns:
            工具结果列表
        """
        tasks = [
            self._execute_single_tool(call)
            for call in tool_calls
        ]
        return await asyncio.gather(*tasks)
    
    async def _execute_single_tool(self, tool_call: Any) -> ToolCallResult:
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
        
        # 审批钩子：支持 human-in-the-loop 工具调用
        if self.approval_handler is not None:
            approval = ToolCallApproval(
                tool_name=tool_name,
                arguments=arguments if isinstance(arguments, dict) else {},
                call_id=call_id,
            )
            try:
                allowed = await self.approval_handler(approval)
            except Exception as e:
                logger.error(f"❌ [Tool] Approval handler failed for tool '{tool_name}': {e}")
                return ToolCallResult(
                    call_id=call_id,
                    content=f"Error: approval handler failed: {e}",
                    is_error=True,
                )

            if not allowed:
                logger.info(f"ℹ️ [Tool] Tool call '{tool_name}' was rejected by approval handler")
                return ToolCallResult(
                    call_id=call_id,
                    content=(
                        "Tool call was rejected by user. "
                        "Do NOT call this tool or any other tools again. "
                        "Answer the user's request directly based on the existing "
                        "conversation without using tools."
                    ),
                    is_error=True,
                )
        
        # Check if this is a skill entry call
        if tool_name in self._skill_map:
            logger.info(f"🌟 [Skill] Activating skill: {tool_name}")
            skill_tool = self._skill_map[tool_name]
            
            # Activate skill (store instance, not name)
            self._active_skill = skill_tool
            
            # Return activation message
            method_list = ", ".join(skill_tool.method_names)
            return ToolCallResult(
                call_id=call_id,
                content=f"Skill '{tool_name}' activated. Available operations: {method_list}",
                is_error=False
            )
        
        # If a skill is active, look up from skill's internal tools first
        if self._active_skill:
            try:
                method_tool = self._active_skill.get_method_tool(tool_name)
                tool = method_tool
            except ValueError:
                # Method not found in active skill, fall back to regular tool_map
                tool = self._tool_map.get(tool_name)
        else:
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
        
            # 提取结果内容
            if hasattr(result, 'content'):
                content = str(result.content)
            elif isinstance(result, dict):
                import json
                content = json.dumps(result, ensure_ascii=False)
            else:
                content = str(result)
        
            logger.info(f"✅ [Tool] Tool '{tool_name}' succeeded")
            logger.debug(f"📦 [Tool] Result: {content[:200]}..." if len(content) > 200 else f"📦 [Tool] Result: {content}")
        
            return ToolCallResult(
                call_id=call_id,
                content=content,
                is_error=False
            )
        except Exception as e:
            logger.error(f"❌ [Tool] Tool '{tool_name}' failed: {str(e)}")
            return ToolCallResult(
                call_id=call_id,
                content=f"Error: {str(e)}",
                is_error=True
            )
