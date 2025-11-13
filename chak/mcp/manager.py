"""
ToolManager 类：管理 LLM + MCP 工具调用循环

核心功能：
- 自动管理多轮工具调用循环
- 并行执行多个工具调用
- 自动重试和错误处理
- 与上下文策略集成
"""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

from loguru import logger

if TYPE_CHECKING:
    from ..message import Message
    from .tool import Tool


@dataclass
class ToolCallResult:
    """工具调用结果"""
    call_id: str
    content: str
    is_error: bool


class ToolManager:
    """
    Manage LLM + MCP tool calling loop.
    
    Auto-handles:
    1. Call LLM with tools
    2. If LLM returns tool_calls -> execute tools -> add results -> back to step 1
    3. If LLM doesn't call tools -> return final answer
    
    Example:
        manager = ToolManager(tools)
        response = await manager.execute_loop(
            provider=provider,
            messages=messages,
            model_uri="openai/gpt-4o"
        )
    """
    
    def __init__(self, tools: List["Tool"], max_iterations: int = 10):
        """
        Args:
            tools: 工具列表
            max_iterations: 最大迭代次数（防止无限循环）
        """
        self.tools = tools
        self._tool_map = {t.name: t for t in tools}
        self.max_iterations = max_iterations
    
    async def execute_loop(
        self, 
        provider: Any,  # LLM Provider
        messages: List["Message"],
        model_uri: str
    ) -> "Message":
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
            Message: Final assistant response (after all tool calls completed)
        
        Raises:
            Exception: Max iteration reached or other errors
        """
        from ..message import AIMessage, ToolMessage
        
        current_messages = messages.copy()
        
        # Convert tools to OpenAI format
        openai_tools = [t.to_openai_tool() for t in self.tools]
        
        for iteration in range(self.max_iterations):
            # Step 1: Call LLM with tools
            logger.debug(f"💬 [MCP Tool Loop] Iteration {iteration}: Calling LLM with {len(openai_tools)} tools...")
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
                    logger.warning(f"⚠️ [MCP] Model doesn't support function calling, gracefully degrading...")
                    logger.debug(f"📝 [MCP Tool Loop] Error message: {str(e)}")
                    # Graceful degradation: call without tools and return
                    response = await asyncio.to_thread(
                        provider.send,
                        messages=current_messages,
                        stream=False
                    )
                    return AIMessage(
                        content=response.content if hasattr(response, 'content') else str(response)
                    )
                else:
                    # Other errors, re-raise
                    logger.error(f"❌ [MCP] LLM call failed: {str(e)}")
                    raise
            
            # Step 2: Check if LLM wants to call tools
            tool_calls = getattr(response, 'tool_calls', None)
            
            logger.debug(f"📊 [MCP Tool Loop] Iteration {iteration}: tool_calls_count={len(tool_calls) if tool_calls else 0}")
            
            if not tool_calls:
                # No tool calls -> LLM finished, return final answer
                logger.info(f"ℹ️ [MCP] No tool calls in this iteration, LLM returned final answer")
                logger.debug(f"✅ [MCP Tool Loop] No tool calls, finishing...")
                return AIMessage(
                    content=response.content if hasattr(response, 'content') else str(response)
                )
            
            logger.info(f"🔧 [MCP] LLM wants to call {len(tool_calls)} tool(s): {[tc.function.name for tc in tool_calls]}")
            logger.debug(f"📤 [MCP Tool Loop] Calling tools: {[tc.function.name for tc in tool_calls]}")
            
            # Step 3: Execute tools in parallel
            logger.debug(f"⏳ [MCP Tool Loop] Executing {len(tool_calls)} tools...")
            tool_results = await self._execute_tools_parallel(tool_calls)
            logger.debug(f"📥 [MCP Tool Loop] Tool results: {[r.content[:50] + '...' if len(r.content) > 50 else r.content for r in tool_results]}")
            
            # Step 4: Add assistant message (with tool_calls) to conversation
            current_messages.append(AIMessage(
                content=response.content if hasattr(response, 'content') else "",
                tool_calls=tool_calls
            ))
            
            # Step 5: Add tool results to conversation
            for result in tool_results:
                current_messages.append(ToolMessage(
                    content=result.content,
                    tool_call_id=result.call_id
                ))
            
            logger.debug(f"🔁 [MCP Tool Loop] Loop continues to iteration {iteration + 1}...")
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
            MessageChunk: Streaming chunks
        
        Raises:
            Exception: Max iteration reached or other errors
        """
        from ..message import AIMessage, ToolMessage, MessageChunk
        
        current_messages = messages.copy()
        openai_tools = [t.to_openai_tool() for t in self.tools]
        
        for iteration in range(self.max_iterations):
            accumulated_content = ""
            accumulated_tool_calls = []
            finish_reason = None
            
            # Step 1: Call LLM with streaming
            logger.debug(f"💬 [MCP Tool Loop] Iteration {iteration}: Calling LLM with {len(openai_tools)} tools (streaming)...")
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
                    logger.warning(f"⚠️ [MCP] Model doesn't support function calling, gracefully degrading to streaming mode...")
                    logger.debug(f"📝 [MCP Tool Loop] Error message: {str(e)}")
                    # Graceful degradation: streaming without tools
                    def _get_fallback_stream():
                        return provider.send(
                            messages=current_messages,
                            stream=True
                        )
                    stream = await asyncio.to_thread(_get_fallback_stream)
                    
                    # Yield all chunks from the fallback stream
                    for chunk in stream:
                        choice = chunk.choices[0] if chunk.choices else None
                        if not choice:
                            continue
                        delta = choice.delta
                        if delta and hasattr(delta, 'content') and delta.content:
                            yield MessageChunk(
                                content=delta.content,
                                is_final=False
                            )
                    
                    yield MessageChunk(content="", is_final=True)
                    return
                else:
                    logger.error(f"❌ [MCP] LLM call failed: {str(e)}")
                    raise
            
            # Step 2: Process streaming chunks (in event loop friendly way)
            for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue
                
                delta = choice.delta
                finish_reason = choice.finish_reason
                
                # 2.1 Has content? -> yield immediately
                if delta and hasattr(delta, 'content') and delta.content:
                    accumulated_content += delta.content
                    yield MessageChunk(
                        content=delta.content,
                        is_final=False,
                        metadata={"iteration": iteration}
                    )
                
                # 2.2 Has tool_calls? -> accumulate
                if delta and hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        index = tc_delta.index
                        
                        # Ensure list is large enough
                        while len(accumulated_tool_calls) <= index:
                            accumulated_tool_calls.append({
                                "id": None,
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        
                        # Update accumulated tool call
                        if tc_delta.id:
                            accumulated_tool_calls[index]["id"] = tc_delta.id
                        if tc_delta.type:
                            accumulated_tool_calls[index]["type"] = tc_delta.type
                        if tc_delta.function:
                            if tc_delta.function.name:
                                accumulated_tool_calls[index]["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                accumulated_tool_calls[index]["function"]["arguments"] += tc_delta.function.arguments
            
            # Step 3: Check finish_reason
            logger.debug(f"📊 [MCP Tool Loop] Iteration {iteration}: finish_reason={finish_reason}, tool_calls_count={len(accumulated_tool_calls)}")
            
            if finish_reason == "tool_calls" and accumulated_tool_calls:
                logger.info(f"🔧 [MCP] LLM wants to call {len(accumulated_tool_calls)} tool(s): {[tc['function']['name'] for tc in accumulated_tool_calls]}")
                logger.debug(f"📤 [MCP Tool Loop] Calling tools: {[tc['function']['name'] for tc in accumulated_tool_calls]}")
                
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
                logger.debug(f"⏳ [MCP Tool Loop] Executing {len(tool_calls_objects)} tools...")
                tool_results = await self._execute_tools_parallel(tool_calls_objects)
                logger.debug(f"📥 [MCP Tool Loop] Tool results: {[r.content[:50] + '...' if len(r.content) > 50 else r.content for r in tool_results]}")
                
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
                
                logger.debug(f"🔁 [MCP Tool Loop] Loop continues to iteration {iteration + 1}...")
                # Loop continues (next iteration will call LLM with tool results)
            else:
                # No tool calls or finish_reason != 'tool_calls' -> done
                logger.info(f"ℹ️ [MCP] No tool calls in this iteration, LLM returned final answer")
                logger.debug(f"✅ [MCP Tool Loop] No tool calls, finishing...")
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
        
        logger.info(f"🔧 [MCP] Calling tool: {tool_name}")
        logger.debug(f"📨 [MCP] Tool call ID: {call_id}")
        logger.debug(f"📨 [MCP] Arguments: {arguments}")
        
        # 解析 arguments（可能是 JSON 字符串）
        if isinstance(arguments, str):
            import json
            try:
                arguments = json.loads(arguments)
                logger.debug(f"🔄 [MCP] Parsed arguments: {arguments}")
            except json.JSONDecodeError:
                logger.error(f"❌ [MCP] Invalid JSON arguments: {arguments}")
                return ToolCallResult(
                    call_id=call_id,
                    content=f"Error: Invalid JSON arguments: {arguments}",
                    is_error=True
                )
        
        tool = self._tool_map.get(tool_name)
        if not tool:
            logger.error(f"❌ [MCP] Tool not found: {tool_name}")
            return ToolCallResult(
                call_id=call_id,
                content=f"Error: Tool not found: {tool_name}",
                is_error=True
            )
        
        try:
            # 调用工具（自带重试）
            logger.debug(f"⚙️  [MCP] Executing tool: {tool_name}...")
            result = await tool.call(arguments)
            
            # 提取结果内容
            if hasattr(result, 'content'):
                content = str(result.content)
            elif isinstance(result, dict):
                import json
                content = json.dumps(result, ensure_ascii=False)
            else:
                content = str(result)
            
            logger.info(f"✅ [MCP] Tool '{tool_name}' succeeded")
            logger.debug(f"📦 [MCP] Result: {content[:200]}..." if len(content) > 200 else f"📦 [MCP] Result: {content}")
            
            return ToolCallResult(
                call_id=call_id,
                content=content,
                is_error=False
            )
        except Exception as e:
            logger.error(f"❌ [MCP] Tool '{tool_name}' failed: {str(e)}")
            return ToolCallResult(
                call_id=call_id,
                content=f"Error: {str(e)}",
                is_error=True
            )
