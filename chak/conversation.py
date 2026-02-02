import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from enum import Enum
from typing import TYPE_CHECKING, List, Dict, Any, Iterator, Union, Optional, AsyncIterator

from .attachment import Attachment
from .context.handlers import BaseContextHandler, NoopContextHandler
from .message import Message, MessageChunk, ReasoningChunk, HumanMessage, AIMessage, SystemMessage, ToolMessage, _current_turn_id
from .providers import create_provider
from .providers.types import ProviderCategory
from .schemas import Reasoning
from .utils.uri import parse as parse_uri

if TYPE_CHECKING:
    from .tools.mcp.tool import MCPTool
    from .tools.manager import ToolManager, ToolApprovalHandler


class ToolExecutor(str, Enum):
    """Tool execution mode."""
    ASYNCIO = "asyncio"  # Use asyncio.to_thread (default, best for IO-bound)
    THREAD = "thread"    # Use ThreadPoolExecutor
    PROCESS = "process"  # Use ProcessPoolExecutor (for CPU-bound tasks)


class Conversation:
    """
    Chat conversation that follows your desired flow:
    URI -> parse -> dict -> ProviderConfig -> Provider -> client
    
    Conversation专用于LLM类型的provider，用于文本对话交互。
    """
    
    # 类常量：指定Conversation只使用LLM类型的provider
    PROVIDER_CATEGORY = ProviderCategory.LLM

    def __init__(
        self, 
        model_uri: str, 
        api_key: str,
        id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        context_handler: Optional[BaseContextHandler] = None,
        tools: Optional[List["MCPTool"]] = None,
        tool_executor: ToolExecutor = ToolExecutor.ASYNCIO,
        tool_approval_handler: Optional["ToolApprovalHandler"] = None,
        **kwargs
    ):
        """
        Initialize conversation from URI.

        Flow:
        1. Parse URI to get components
        2. Create provider-specific config from parsed dict + kwargs
        3. Create provider with that config
        4. Provider initializes its client
        
        Args:
            model_uri: Model URI string (e.g., "bailian@https://...:qwen-plus")
            api_key: API key for authentication
            id: Unique conversation ID (auto-generated if not provided)
            system_prompt: Optional system prompt to initialize the conversation.
                          If you need structured content, use \n\n to separate sections.
            context_handler: Context management handler (default: NoopContextHandler)
            tools: Optional list of MCP tools or native functions (requires async asend() method)
            tool_executor: Tool execution mode (default: ToolExecutor.ASYNCIO)
                          - ASYNCIO: Best for IO-bound tasks (API calls, DB queries)
                          - THREAD: ThreadPoolExecutor for sync blocking operations
                          - PROCESS: ProcessPoolExecutor for CPU-bound tasks
            **kwargs: Additional configuration parameters
        
        Example:
            >>> # Simple system prompt
            >>> conv = Conversation(
            ...     model_uri="openai:gpt-4",
            ...     api_key="sk-...",
            ...     system_prompt="You are a helpful assistant."
            ... )
            >>> 
            >>> # Structured system prompt
            >>> system_prompt = (
            ...     "You are a helpful assistant.\n\n"
            ...     "Rules:\n"
            ...     "- Always respond in Chinese\n"
            ...     "- Be concise and professional"
            ... )
            >>> conv = Conversation(
            ...     model_uri="openai:gpt-4",
            ...     api_key="sk-...",
            ...     system_prompt=system_prompt
            ... )
        """
        self.model_uri = model_uri
        self.api_key = api_key
        self.id = id or str(uuid.uuid4())
        self.messages = []
        self.attachments: List[Attachment] = []  # Session-level attachment tracking
        
        # Tool management
        self._raw_tools: List = []  # Store original tools
        self._tool_manager: Optional["ToolManager"] = None
        
        # Tool executor configuration
        self._tool_executor = tool_executor
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._tool_approval_handler: Optional["ToolApprovalHandler"] = tool_approval_handler
        
        # Initialize tools if provided
        if tools:
            self.add_tools(tools)
        
        # Initialize system prompt
        self._initial_system_message = self._normalize_system_message(system_prompt)
        if self._initial_system_message:
            self.messages.append(self._initial_system_message)
        
        # Initialize context handler
        self.context_handler = context_handler or NoopContextHandler()

        # 1. Parse URI to dict
        parsed = parse_uri(model_uri)

        # 2. Build config dict (URI params + kwargs + model)
        config_dict = self._build_config_dict(parsed, kwargs)

        # 3. Create provider with LLM category
        self.provider = create_provider(
            parsed['provider'],
            config_dict,
            category=self.PROVIDER_CATEGORY
        )
        
        # Store provider name and model name for easy access
        self._provider_name = parsed['provider']
        self._model_name = parsed['model']

    def _normalize_system_message(self, system_prompt: Optional[str]) -> Optional[SystemMessage]:
        """
        Convert system prompt string to SystemMessage object.
        
        Args:
            system_prompt: System prompt string
            
        Returns:
            SystemMessage object, or None if input is empty
        """
        if not system_prompt:
            return None
        
        if not isinstance(system_prompt, str):
            raise TypeError(f"system_prompt must be str, got {type(system_prompt)}")
        
        return SystemMessage(content=system_prompt)
    
    def set_tool_executor(self, executor: ToolExecutor) -> None:
        """
        Change tool execution mode.
        
        Args:
            executor: New execution mode (ToolExecutor.ASYNCIO/THREAD/PROCESS)
        
        Example:
            >>> conv = Conversation(..., tool_executor=ToolExecutor.ASYNCIO)
            >>> # Switch to process pool for CPU-bound tasks
            >>> conv.set_tool_executor(ToolExecutor.PROCESS)
        """
        self._tool_executor = executor
        # Update tool manager's executor if it exists
        if self._tool_manager:
            self._tool_manager.executor = self._get_executor()
    
    def _get_executor(self, override: Optional[ToolExecutor] = None):
        """
        Get executor for current mode (with optional override).
        
        Args:
            override: Optional executor to use instead of default
        
        Returns:
            Executor instance or None for asyncio mode
        """
        mode = override or self._tool_executor
        
        if mode == ToolExecutor.PROCESS:
            if not self._process_pool:
                self._process_pool = ProcessPoolExecutor()
            return self._process_pool
        elif mode == ToolExecutor.THREAD:
            if not self._thread_pool:
                self._thread_pool = ThreadPoolExecutor()
            return self._thread_pool
        else:  # ASYNCIO or invalid (default to asyncio)
            return None
    
    def get_tools(self) -> List:
        """
        Get current tools list (original format).
        
        Returns:
            List of tools in the format they were added (functions, objects, MCPTool instances)
        
        Example:
            >>> conv = Conversation(..., tools=[my_func, my_obj])
            >>> tools = conv.get_tools()
            >>> print(tools)  # [<function my_func>, <MyClass object>]
        """
        return self._raw_tools.copy()
    
    def add_tools(self, tools: List) -> None:
        """
        Add tools to the conversation.
        
        Supports:
        - Functions (callable)
        - Objects (with public methods)
        - MCPTool instances
        
        Note: If a tool with the same name already exists, it will be replaced.
        
        Args:
            tools: List of tools to add
        
        Example:
            >>> def my_func(x: int) -> int:
            ...     return x * 2
            >>> 
            >>> conv = Conversation(...)
            >>> conv.add_tools([my_func])
        """
        self._raw_tools.extend(tools)
        self._rebuild_tool_manager()
    
    def remove_tools(self, tools: List) -> None:
        """
        Remove tools from the conversation by reference.
        
        Args:
            tools: List of tool objects to remove (same objects that were added)
        
        Example:
            >>> my_func = lambda x: x + 1
            >>> conv.add_tools([my_func])
            >>> conv.remove_tools([my_func])  # Remove by reference
            >>> 
            >>> # Or get tools first
            >>> tools = conv.get_tools()
            >>> conv.remove_tools([tools[0]])  # Remove first tool
        """
        for tool in tools:
            if tool in self._raw_tools:
                self._raw_tools.remove(tool)
        self._rebuild_tool_manager()
    
    def clear_tools(self) -> None:
        """
        Clear all tools from the conversation.
        
        Example:
            >>> conv = Conversation(..., tools=[func1, func2])
            >>> conv.clear_tools()
            >>> len(conv.get_tools())  # 0
        """
        self._raw_tools.clear()
        self._tool_manager = None
    
    def _rebuild_tool_manager(self) -> None:
        """
        Rebuild ToolManager with current tools.
        
        Called automatically when tools are added/removed.
        """
        if not self._raw_tools:
            self._tool_manager = None
            return
        
        from .tools import wrap_tools
        from .tools.manager import ToolManager
        
        wrapped_tools = wrap_tools(self._raw_tools)
        executor = self._get_executor()
        self._tool_manager = ToolManager(
            wrapped_tools,
            executor=executor,
            approval_handler=self._tool_approval_handler,
        )
    
    def _build_config_dict(self, parsed_uri: Dict, kwargs: Dict) -> Dict[str, Any]:
        """Build configuration dictionary from URI and kwargs."""
        config_dict = {}

        # Core config from URI
        config_dict['api_key'] = self.api_key
        config_dict['model'] = parsed_uri['model']

        # Add base_url from URI if present
        if parsed_uri['base_url']:
            config_dict['base_url'] = parsed_uri['base_url']

        # Add parameters from URI query string
        config_dict.update(parsed_uri['params'])

        # Add/override with kwargs (kwargs have higher priority)
        config_dict.update(kwargs)

        return config_dict

    def add_messages(self, messages: List[Union[Message, Dict[str, str]]]) -> None:
        """
        Batch add messages to conversation history for restoring previous conversations.
        
        Args:
            messages: List of messages, can be Message objects or dicts
                     Dict format: {"role": "user", "content": "hello"}
        
        Example:
            >>> conv = Conversation(...)
            >>> # Restore conversation history
            >>> conv.add_messages([
            ...     {"role": "user", "content": "Hello"},
            ...     {"role": "assistant", "content": "Hi! How can I help you?"},
            ...     {"role": "user", "content": "Tell me about yourself"}
            ... ])
        """
        for msg in messages:
            if isinstance(msg, dict):
                # Convert dict to Message object
                role = msg['role']
                content = msg.get('content')
                
                if role == "user":
                    self.messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    self.messages.append(AIMessage(content=content))
                elif role == "system":
                    self.messages.append(SystemMessage(content=content))
                elif role == "tool":
                    self.messages.append(ToolMessage(content=content))
                else:
                    raise ValueError(f"Invalid role: {role}")
            elif isinstance(msg, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
                # Already a Message object, add directly
                self.messages.append(msg)
            else:
                raise TypeError(f"Message must be Message object or dict, got: {type(msg)}")

    def send(
            self,
            message: Union[str, Message],
            attachments: Optional[List[Attachment]] = None,
            stream: bool = False,
            timeout: Optional[int] = None,
            returns: Optional[type] = None,
            reasoning: Optional[Union[Reasoning, dict]] = None,
            **kwargs
    ) -> Union[Message, Iterator[Union[MessageChunk, ReasoningChunk]], Any]:
        """
        Send message (sync, no MCP tools support).
        
        Supports:
        - ✅ Streaming
        - ✅ Non-streaming
        - ✅ Multimodal (images, audio)
        - ✅ Reasoning mode (for compatible models)
        - ❌ MCP tools (not supported)
        
        For MCP tool usage, please use: await conv.asend(message)
        
        Args:
            message: Message content (str will be converted to HumanMessage)
            attachments: Optional list of Attachment objects (images, audio, etc.)
            stream: Enable streaming
            timeout: Request timeout in seconds. If None, uses provider's default timeout (30s)
            returns: Optional Pydantic model class for structured output. When provided,
                    forces LLM to return data matching this schema via function calling
            **kwargs: Additional LLM parameters
        
        Returns:
            - If stream=False: Complete Message
            - If stream=True: Iterator[Union[MessageChunk, ReasoningChunk]]
              You can distinguish chunk types using isinstance() or match-case
        
        Raises:
            RuntimeError: If tools are configured
        
        Examples:
            # Simple usage
            response = conv.send("Hello")
            
            # With streaming - distinguish answer vs reasoning chunks
            for chunk in conv.send("Hello", stream=True):
                if isinstance(chunk, MessageChunk):
                    print(chunk.content, end="")  # Answer content
                elif isinstance(chunk, ReasoningChunk):
                    print(f"[Thinking: {chunk.content}]")  # Reasoning content
            
            # With image
            from chak.attachment import Image
            response = conv.send(
                "What's in this image?",
                attachments=[Image("https://example.com/photo.jpg")]
            )
            
            # With custom timeout
            response = conv.send(
                "Analyze this document",
                attachments=[PDF("large.pdf")],
                timeout=120
            )
            
            # Advanced: Send specific message type
            conv.send(SystemMessage(content="You are helpful"))
            conv.send(HumanMessage(content="Hello"))
        """
        # Set turn ID for this entire send operation
        turn_id = str(uuid.uuid4())
        token = _current_turn_id.set(turn_id)
        
        try:
            # Check if tools are configured
            if self._raw_tools:
                raise RuntimeError(
                    "MCP tools require async execution. "
                    "Please use: await conv.asend(message)"
                )
            
            # Check if structured output is requested
            if returns is not None:
                raise RuntimeError(
                    "Structured output (returns parameter) requires async execution. "
                    "Please use: await conv.asend(message, returns=YourModel)"
                )
            
            # Merge timeout into kwargs if specified
            if timeout is not None:
                kwargs['timeout'] = timeout
            
            # Check if in async context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    raise RuntimeError(
                        "Cannot use sync send() in async context. "
                        "Please use: await conv.asend(message)"
                    )
            except RuntimeError:
                pass
            
            # Convert str to HumanMessage and merge attachments if present
            if isinstance(message, str):
                if attachments:
                    # Create multimodal message
                    content_parts = [{"type": "text", "text": message}]
                    for att in attachments:
                        if att.mime_type.is_image():
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": att.source}
                            })
                        elif att.mime_type.is_audio():
                            content_parts.append({
                                "type": "input_audio",
                                "input_audio": {
                                    "data": att.source,
                                    "format": att.mime_type.subtype
                                }
                            })
                        elif att.mime_type.is_video():
                            content_parts.append({
                                "type": "video",
                                "video": {"url": att.source}
                            })
                        elif att.mime_type.is_document() or att.reader:
                            # Document types (PDF, DOC, Excel, TXT, Link, etc.)
                            # Need to read and extract text content first
                            doc_result = att.read()  # Sync read for sync method
                            if doc_result and doc_result.content:
                                # Add extracted text as text part
                                doc_text = doc_result.content
                                # Include metadata info if available
                                if doc_result.meta:
                                    meta_str = ", ".join([f"{k}: {v}" for k, v in doc_result.meta.items() if k != "error"])
                                    if meta_str:
                                        doc_text = f"[Document metadata: {meta_str}]\n\n{doc_text}"
                                content_parts.append({
                                    "type": "text",
                                    "text": doc_text
                                })
                    user_message = HumanMessage(content=content_parts, attachments=list(attachments) if attachments else [])
                else:
                    # Simple text message
                    user_message = HumanMessage(content=message)
            else:
                # User provided a Message object directly
                user_message = message
            
            self.messages.append(user_message)
            
            # Track attachments at conversation level
            if attachments:
                self.attachments.extend(attachments)

            # Apply context handler
            messages_to_send = self._apply_context_handler()

            # Add reasoning to kwargs if provided
            if reasoning is not None:
                kwargs['reasoning'] = reasoning

            # Normal LLM call (no tools)
            if stream:
                return self._send_stream(messages_to_send, **kwargs)
            else:
                return self._send_nonstream(messages_to_send, **kwargs)
        finally:
            # Reset turn ID context
            _current_turn_id.reset(token)
    
    async def asend(
            self,
            message: Union[str, Message],
            attachments: Optional[List[Attachment]] = None,
            stream: bool = False,
            event: bool = False,
            timeout: Optional[int] = None,
            returns: Optional[type] = None,
            tool_executor: Optional[ToolExecutor] = None,
            reasoning: Optional[Union[Reasoning, dict]] = None,
            **kwargs
    ) -> Union[Message, AsyncIterator[Union[MessageChunk, ReasoningChunk]], AsyncIterator['StreamEvent'], Any, None]:
        """
        Send message (async, full featured).
        
        Supports:
        - ✅ Streaming
        - ✅ Non-streaming
        - ✅ Multimodal (images, audio)
        - ✅ MCP tools (both modes)
        - ✅ Structured output (returns parameter)
        - ✅ Event stream (for tool observability)
        - ✅ Reasoning mode (for compatible models)
        
        Args:
            message: Message content (str will be converted to HumanMessage)
            attachments: Optional list of Attachment objects (images, audio, etc.)
            stream: Enable streaming (ignored if event=True)
            event: Enable event stream mode (returns MessageChunk + ReasoningChunk + ToolCall events)
                  When True, you can observe tool calls in real-time using isinstance() or match-case.
                  Note: event=True will override stream parameter.
            timeout: Request timeout in seconds. If None, uses provider's default timeout (30s)
            returns: Optional Pydantic model class for structured output. When provided,
                    forces LLM to return data matching this schema via function calling.
                    Returns None if extraction fails.
            tool_executor: Optional override for tool execution mode (for this call only)
            reasoning: Optional reasoning configuration dict (e.g., {"effort": "medium"})
                      for compatible models (OpenAI o1/o3, Bailian QwQ).
            **kwargs: Additional LLM parameters
        
        Returns:
            - If event=False and stream=False and returns=None: Complete Message
            - If event=False and stream=True: AsyncIterator[Union[MessageChunk, ReasoningChunk]]
            - If event=True: AsyncIterator[StreamEvent] (MessageChunk/ReasoningChunk/ToolCallStartEvent/ToolCallSuccessEvent/ToolCallErrorEvent)
            - If returns is provided: Validated Pydantic model instance or None if failed
        
        Examples:
            # Non-streaming
            response = await conv.asend("Hello")
            
            # Streaming - handle both answer and reasoning chunks
            async for chunk in await conv.asend("Hello", stream=True):
                if isinstance(chunk, MessageChunk):
                    print(chunk.content, end="")  # Answer content
                elif isinstance(chunk, ReasoningChunk):
                    print(f"[Thinking: {chunk.content}]")  # Reasoning content
            
            # Event stream (with tool observability)
            async for event in await conv.asend("What's the weather?", event=True):
                match event:
                    case MessageChunk(content=text):
                        print(text, end="")
                    case ReasoningChunk(content=reasoning):
                        print(f"[Reasoning: {reasoning}]")
                    case ToolCallStartEvent(tool_name=name, arguments=args):
                        print(f"\n🔧 Calling {name}")
                    case ToolCallEndEvent(tool_name=name, result=res):
                        print(f"✅ Result: {res[:100]}")
            
            # With image
            from chak.attachment import Image
            response = await conv.asend(
                "What's in this image?",
                attachments=[Image("https://example.com/photo.jpg")]
            )
            
            # With custom timeout
            response = await conv.asend(
                "Analyze this document",
                attachments=[TXT("large.txt")],
                timeout=120
            )
            
            # With MCP tools (non-streaming)
            conv = Conversation("gpt-4", tools=mcp_tools)
            response = await conv.asend("What's the weather?")
            
            # With MCP tools (streaming)
            conv = Conversation("gpt-4", tools=mcp_tools)
            async for chunk in await conv.asend("What's the weather?", stream=True):
                print(chunk.content, end="")
            
            # Advanced: Send specific message type
            await conv.asend(SystemMessage(content="You are helpful"))
            await conv.asend(HumanMessage(content="Hello"))
        """
        # Set turn ID for this entire asend operation
        turn_id = str(uuid.uuid4())
        token = _current_turn_id.set(turn_id)
        
        try:
            # Merge timeout into kwargs if specified
            if timeout is not None:
                kwargs['timeout'] = timeout
            
            # Handle structured output (returns parameter)
            if returns is not None:
                try:
                    return await self._asend_with_structured_output(
                        message=message,
                        attachments=attachments,
                        returns=returns,
                        **kwargs
                    )
                except Exception:
                    # Structured output failed, return None
                    return None
        
                # Convert str to HumanMessage and merge attachments if present
            if isinstance(message, str):
                if attachments:
                    # Create multimodal message
                    content_parts = [{"type": "text", "text": message}]
                    for att in attachments:
                        if att.mime_type.is_image():
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": att.source}
                            })
                        elif att.mime_type.is_audio():
                            content_parts.append({
                                "type": "input_audio",
                                "input_audio": {
                                    "data": att.source,
                                    "format": att.mime_type.subtype
                                }
                            })
                        elif att.mime_type.is_video():
                            content_parts.append({
                                "type": "video",
                                "video": {"url": att.source}
                            })
                        elif att.mime_type.is_document() or att.reader:
                            # Document types (PDF, DOC, Excel, TXT, Link, etc.)
                            # Need to read and extract text content first
                            doc_result = await att.aread()  # Async read for async method
                            if doc_result and doc_result.content:
                                # Add extracted text as text part
                                doc_text = doc_result.content
                                # Include metadata info if available
                                if doc_result.meta:
                                    meta_str = ", ".join([f"{k}: {v}" for k, v in doc_result.meta.items() if k != "error"])
                                    if meta_str:
                                        doc_text = f"[Document metadata: {meta_str}]\n\n{doc_text}"
                                content_parts.append({
                                    "type": "text",
                                    "text": doc_text
                                })
                    user_message = HumanMessage(content=content_parts, attachments=list(attachments) if attachments else [])
                else:
                    # Simple text message
                    user_message = HumanMessage(content=message)
            else:
                # User provided a Message object directly
                user_message = message
            
            self.messages.append(user_message)
            
            # Track attachments at conversation level
            if attachments:
                self.attachments.extend(attachments)

            # Apply context handler
            messages_to_send = self._apply_context_handler()

            # Check if event stream mode is requested
            if event:
                # Event stream mode: returns MessageChunk + ToolCall events
                # This overrides stream parameter and provides full observability
                return self._asend_with_events(messages_to_send, tool_executor, **kwargs)

            # Check if tools are configured
            if self._tool_manager:
                # Temporarily override executor if specified for this call
                original_executor = None
                if tool_executor is not None:
                    original_executor = self._tool_manager.executor
                    self._tool_manager.executor = self._get_executor(tool_executor)
                
                try:
                    # MCP tools mode
                    if stream:
                        return self._asend_stream_with_tools(messages_to_send, **kwargs)
                    else:
                        return await self._asend_nonstream_with_tools(messages_to_send, **kwargs)
                finally:
                    # Restore original executor
                    if original_executor is not None:
                        self._tool_manager.executor = original_executor
            else:
                # Add reasoning to kwargs if provided
                if reasoning is not None:
                    kwargs['reasoning'] = reasoning
                
                # Normal LLM mode
                if stream:
                    return self._asend_stream(messages_to_send, **kwargs)
                else:
                    return await self._asend_nonstream(messages_to_send, **kwargs)
        finally:
            # Reset turn ID context
            _current_turn_id.reset(token)
    
    async def _asend_stream(self, messages: List[Message], **kwargs) -> AsyncIterator[Union[MessageChunk, ReasoningChunk]]:
        """Handle async streaming response with support for both answer and reasoning chunks."""
        # Get provider chunks (sync iterator)
        def _get_sync_chunks():
            return self.provider.send(
                messages=messages,
                stream=True,
                **kwargs
            )
        
        # Run in thread to avoid blocking
        provider_chunks = await asyncio.to_thread(_get_sync_chunks)

        # Convert to standard chunks and collect content
        complete_content = ""
        complete_reasoning_content = ""
        last_chunk_was_final = False
        last_chunk_metadata = {}
        
        for provider_chunk in provider_chunks:
            unified_chunk = self.provider.converter.from_provider_chunk(provider_chunk)
            
            # Handle reasoning content
            if unified_chunk.reasoning_content:
                complete_reasoning_content += unified_chunk.reasoning_content
                if unified_chunk.metadata:
                    last_chunk_metadata.update(unified_chunk.metadata)
                yield ReasoningChunk(
                    content=unified_chunk.reasoning_content,
                    is_final=False,
                    metadata=unified_chunk.metadata
                )
            
            # Handle regular content
            if unified_chunk.content:
                complete_content += unified_chunk.content
                if unified_chunk.metadata:
                    last_chunk_metadata = unified_chunk.metadata
                yield MessageChunk(
                    content=unified_chunk.content,
                    is_final=False,
                    metadata=unified_chunk.metadata
                )
            
            # Check if final
            if unified_chunk.is_final:
                last_chunk_was_final = True

        # Only send additional final chunk if provider didn't send one
        if complete_content and not last_chunk_was_final:
            final_message = AIMessage(
                content=complete_content,
                reasoning_content=complete_reasoning_content if complete_reasoning_content else None,
                metadata=last_chunk_metadata
            )
            self.messages.append(final_message)
            
            # Send a special final chunk containing the complete message
            yield MessageChunk(
                content="",
                is_final=True,
                final_message=final_message
            )
        elif complete_content:
            # Provider sent final chunk, just save the message
            final_message = AIMessage(
                content=complete_content,
                reasoning_content=complete_reasoning_content if complete_reasoning_content else None,
                metadata=last_chunk_metadata
            )
            self.messages.append(final_message)
    
    async def _asend_stream_with_tools(self, messages: List[Message], **kwargs) -> AsyncIterator[Union[MessageChunk, ReasoningChunk]]:
        """Handle async streaming response with MCP tools, supporting both answer and reasoning chunks."""
        if not self._tool_manager:
            raise RuntimeError("Tool manager not initialized")
        
        all_new_messages = []
        
        # Use tool manager's streaming loop
        async for chunk, new_messages in self._tool_manager.execute_loop_stream(
            provider=self.provider,
            messages=messages,
            model_uri=self.model_uri
        ):
            # Sync new_messages to all_new_messages
            all_new_messages = new_messages
            yield chunk
        
        # Save all new messages (including intermediate AIMessage + ToolMessage)
        self.messages.extend(all_new_messages)
    
    async def _asend_nonstream_with_tools(self, messages: List[Message], **kwargs) -> Message:
        """Handle async non-streaming response with MCP tools."""
        if not self._tool_manager:
            raise RuntimeError("Tool manager not initialized")
        
        final_message, new_messages = await self._tool_manager.execute_loop(
            provider=self.provider,
            messages=messages,
            model_uri=self.model_uri
        )
        # Add all new messages (including intermediate AIMessage + ToolMessage) to conversation history
        self.messages.extend(new_messages)
        return final_message
    
    async def _asend_with_structured_output(
        self, 
        message: Union[str, Message],
        attachments: Optional[List[Attachment]],
        returns: type,
        **kwargs
    ) -> Any:
        """
        Handle structured output by converting Pydantic model to tool calling.
        This method:
        1. Creates a virtual tool from the Pydantic model schema
        2. Forces LLM to call this tool
        3. Parses and validates the result
        4. Returns the Pydantic model instance
        
        Supports:
        - BaseModel: returns=MyModel
        - List[BaseModel]: returns=List[MyModel]
        - Dict[str, BaseModel]: returns=Dict[str, MyModel]
        
        Args:
            message: User message
            attachments: Optional attachments
            returns: Pydantic BaseModel class or generic type (List/Dict)
            **kwargs: Additional LLM parameters
            
        Returns:
            Instance of the Pydantic model (or List/Dict of instances)
        """
        from pydantic import BaseModel, RootModel
        from typing import get_origin, get_args
        import json
        
        # Check if returns is a generic type (List/Dict)
        origin = get_origin(returns)
        is_wrapped = False
        original_returns = returns
        
        if origin is list:
            # List[BaseModel] -> wrap with RootModel
            args = get_args(returns)
            if not args or not (isinstance(args[0], type) and issubclass(args[0], BaseModel)):
                raise TypeError(
                    f"List type must contain a Pydantic BaseModel, got List[{args[0] if args else 'Unknown'}]"
                )
            # Create RootModel wrapper
            returns = RootModel[original_returns]
            is_wrapped = True
        elif origin is dict:
            # Dict[str, BaseModel] -> wrap with RootModel
            args = get_args(returns)
            if len(args) != 2 or args[0] != str or not (isinstance(args[1], type) and issubclass(args[1], BaseModel)):
                raise TypeError(
                    f"Dict type must be Dict[str, BaseModel], got Dict[{args[0] if len(args) > 0 else 'Unknown'}, {args[1] if len(args) > 1 else 'Unknown'}]"
                )
            # Create RootModel wrapper
            returns = RootModel[original_returns]
            is_wrapped = True
        elif not (isinstance(returns, type) and issubclass(returns, BaseModel)):
            # Not a BaseModel and not a supported generic type
            raise TypeError(
                f"returns must be a Pydantic BaseModel, List[BaseModel], or Dict[str, BaseModel], got {type(returns)}"
            )
        
        # Convert str to HumanMessage (same logic as regular asend)
        if isinstance(message, str):
            if attachments:
                # Create multimodal message
                content_parts = [{"type": "text", "text": message}]
                for att in attachments:
                    if att.mime_type.is_image():
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": att.source}
                        })
                    elif att.mime_type.is_audio():
                        content_parts.append({
                            "type": "input_audio",
                            "input_audio": {
                                "data": att.source,
                                "format": att.mime_type.subtype
                            }
                        })
                    elif att.mime_type.is_video():
                        content_parts.append({
                            "type": "video",
                            "video": {"url": att.source}
                        })
                    elif att.mime_type.is_document() or att.reader:
                        doc_result = await att.aread()
                        if doc_result and doc_result.content:
                            doc_text = doc_result.content
                            if doc_result.meta:
                                meta_str = ", ".join([f"{k}: {v}" for k, v in doc_result.meta.items() if k != "error"])
                                if meta_str:
                                    doc_text = f"[Document metadata: {meta_str}]\n\n{doc_text}"
                            content_parts.append({
                                "type": "text",
                                "text": doc_text
                            })
                user_message = HumanMessage(content=content_parts, attachments=list(attachments) if attachments else [])
            else:
                user_message = HumanMessage(content=message)
        else:
            user_message = message
        
        self.messages.append(user_message)
        
        if attachments:
            self.attachments.extend(attachments)
        
        # Apply context handler
        messages_to_send = self._apply_context_handler()
        
        # Generate tool schema from Pydantic model
        tool_schema = self._generate_tool_schema_from_model(returns)
        
        # Add tool to kwargs and force its usage
        kwargs['tools'] = [tool_schema]
        kwargs['tool_choice'] = {
            "type": "function",
            "function": {"name": tool_schema["function"]["name"]}
        }
        
        # Call LLM with forced tool calling
        response = await asyncio.to_thread(
            self.provider.send,
            messages=messages_to_send,
            **kwargs
        )
        
        # Extract tool call result
        # Note: provider.send returns AIMessage, not raw ChatCompletion
        if isinstance(response, AIMessage):
            # Direct AIMessage from provider
            if not response.tool_calls:
                raise RuntimeError(
                    "LLM did not return a tool call. "
                    "This may happen if the model doesn't support function calling."
                )
            tool_call = response.tool_calls[0]
        elif hasattr(response, 'choices') and response.choices:
            # Raw ChatCompletion format (fallback)
            message_obj = response.choices[0].message
            if not hasattr(message_obj, 'tool_calls') or not message_obj.tool_calls:
                raise RuntimeError(
                    "LLM did not return a tool call. "
                    "This may happen if the model doesn't support function calling."
                )
            tool_call = message_obj.tool_calls[0]
        else:
            raise RuntimeError("Unexpected response format from LLM")
        
        # Parse arguments and validate with Pydantic
        try:
            arguments = json.loads(tool_call.function.arguments)
            result = returns.model_validate(arguments)
            
            # Unwrap RootModel if it was auto-wrapped
            if is_wrapped:
                result = result.root
            
            # Save the AI message to conversation history
            if isinstance(response, AIMessage):
                # Response is already AIMessage, save it directly
                self.messages.append(response)
            else:
                # Fallback: create AIMessage from ChatCompletion
                ai_message = AIMessage(
                    content=response.choices[0].message.content or "",
                    tool_calls=response.choices[0].message.tool_calls
                )
                self.messages.append(ai_message)
            
            # Add a virtual tool response to complete the tool calling flow
            # This prevents LLM from expecting a tool response in the next turn
            tool_response = ToolMessage(
                content="Structured data extracted successfully",
                tool_call_id=tool_call.id
            )
            self.messages.append(tool_response)
            
            return result
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse tool call arguments: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to validate response with Pydantic model: {e}")
    
    def _generate_tool_schema_from_model(self, model: type) -> Dict[str, Any]:
        """
        Generate OpenAI tool schema from Pydantic model.
        
        Args:
            model: Pydantic BaseModel class (or RootModel for List/Dict)
            
        Returns:
            OpenAI tool definition dict
        """
        from pydantic import RootModel
        
        schema = model.model_json_schema()
        
        # Extract description from model docstring or use default
        # For RootModel, extract from the wrapped type
        if hasattr(model, '__pydantic_generic_metadata__'):
            # This is a RootModel, use generic description
            model_name = "ExtractedData"
            description = "Correctly extracted structured data with all required parameters"
        else:
            model_name = model.__name__
            description = model.__doc__ or f"Correctly extracted `{model.__name__}` with all required parameters"
            description = description.strip()
        
        return {
            "type": "function",
            "function": {
                "name": model_name,
                "description": description,
                "parameters": schema
            }
        }
    
    async def _asend_nonstream(self, messages: List[Message], **kwargs) -> Message:
        """Handle async non-streaming response without tools."""
        # Use asyncio.to_thread to wrap sync provider call
        response = await asyncio.to_thread(
            self.provider.send,
            messages=messages,
            stream=False,
            **kwargs
        )
        # Convert to AIMessage
        if not isinstance(response, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
            ai_response = AIMessage(
                content=response.content,  # type: ignore
                reasoning_content=response.reasoning_content,  # type: ignore
                tool_calls=response.tool_calls,  # type: ignore
                refusal=response.refusal,  # type: ignore
                metadata=response.metadata  # type: ignore
            )
        else:
            ai_response = response  # type: ignore
        self.messages.append(ai_response)
        return ai_response
    
    async def _asend_with_events(
        self,
        messages: List[Message],
        tool_executor: Optional[ToolExecutor],
        **kwargs
    ) -> AsyncIterator['StreamEvent']:
        """
        Return unified event stream (content + tool calls).
        
        This method provides complete observability of the conversation flow,
        including LLM content generation and tool execution details.
        
        Yields:
            StreamEvent: MessageChunk, ReasoningChunk, ToolCallStartEvent, ToolCallSuccessEvent, or ToolCallErrorEvent
        """
        from .message import MessageChunk, ReasoningChunk
        
        if self._tool_manager:
            # Temporarily override executor if specified
            original_executor = None
            if tool_executor is not None:
                original_executor = self._tool_manager.executor
                self._tool_manager.executor = self._get_executor(tool_executor)
            
            try:
                # Has tools: return full event stream from tool manager
                async for event in self._tool_manager.execute_loop_with_events(
                    provider=self.provider,
                    messages=messages,
                    model_uri=self.model_uri
                ):
                    yield event
            finally:
                # Restore original executor
                if original_executor is not None:
                    self._tool_manager.executor = original_executor
        else:
            # No tools: only return content events (both MessageChunk and ReasoningChunk)
            async for chunk in self._asend_stream(messages, **kwargs):
                # Directly yield the chunk (whether MessageChunk or ReasoningChunk)
                yield chunk

    def _send_nonstream(self, messages: List[Message], **kwargs) -> Message:
        """Handle non-streaming response."""
        response = self.provider.send(
            messages=messages,
            stream=False,
            **kwargs
        )
        # Convert to AIMessage
        if not isinstance(response, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
            ai_response = AIMessage(
                content=response.content,  # type: ignore
                reasoning_content=response.reasoning_content,  # type: ignore
                tool_calls=response.tool_calls,  # type: ignore
                refusal=response.refusal,  # type: ignore
                metadata=response.metadata  # type: ignore
            )
        else:
            ai_response = response  # type: ignore
        self.messages.append(ai_response)
        return ai_response
    
    def _send_stream(self, messages: List[Message], **kwargs) -> Iterator[Union[MessageChunk, ReasoningChunk]]:
        """Handle streaming response with support for both answer and reasoning chunks."""
        # Get provider chunks (model is already in provider config)
        provider_chunks = self.provider.send(
            messages=messages,
            stream=True,
            **kwargs
        )

        # Convert to standard chunks and collect content
        complete_content = ""
        complete_reasoning_content = ""
        last_chunk_was_final = False
        last_chunk_metadata = {}
        
        for provider_chunk in provider_chunks:
            unified_chunk = self.provider.converter.from_provider_chunk(provider_chunk)
            
            # Handle reasoning content
            if unified_chunk.reasoning_content:
                complete_reasoning_content += unified_chunk.reasoning_content
                if unified_chunk.metadata:
                    last_chunk_metadata.update(unified_chunk.metadata)
                yield ReasoningChunk(
                    content=unified_chunk.reasoning_content,
                    is_final=False,
                    metadata=unified_chunk.metadata
                )
            
            # Handle regular content
            if unified_chunk.content:
                complete_content += unified_chunk.content
                if unified_chunk.metadata:
                    last_chunk_metadata = unified_chunk.metadata
                yield MessageChunk(
                    content=unified_chunk.content,
                    is_final=False,
                    metadata=unified_chunk.metadata
                )
            
            # Check if final
            if unified_chunk.is_final:
                last_chunk_was_final = True

        # Only send additional final chunk if provider didn't send one
        if complete_content and not last_chunk_was_final:
            final_message = AIMessage(
                content=complete_content,
                reasoning_content=complete_reasoning_content if complete_reasoning_content else None,
                metadata=last_chunk_metadata
            )
            self.messages.append(final_message)
            
            # Send a special final chunk containing the complete message
            yield MessageChunk(
                content="",
                is_final=True,
                final_message=final_message
            )
        elif complete_content:
            # Provider sent final chunk, just save the message
            final_message = AIMessage(
                content=complete_content,
                reasoning_content=complete_reasoning_content if complete_reasoning_content else None,
                metadata=last_chunk_metadata
            )
            self.messages.append(final_message)

    def _apply_context_handler(self) -> List[Message]:
        """
        Apply context handler to process messages.
        
        Returns:
            Context messages for this round (handler output)
        """
        if not self.messages:
            return []
        
        # Call handler with messages and conversation id
        context_messages = self.context_handler(
            messages=self.messages,
            conversation_id=self.id
        )
        
        return context_messages
    
    def clear(self):
        """Clear conversation history."""
        self.messages.clear()
    
    def reset(self):
        """
        Reset conversation to initial state.
        
        Clear all message history but preserve the initial system message.
        This is very useful when using Conversation as a tool to avoid message pollution.
        
        Example:
            >>> conv = Conversation(
            ...     model_uri="openai:gpt-4",
            ...     api_key="sk-...",
            ...     system_prompt="You are a helpful assistant."
            ... )
            >>> conv.send("Hello")
            >>> conv.send("How are you?")
            >>> len(conv.messages)  # 3 (1 system + 2 conversations)
            >>> conv.reset()
            >>> len(conv.messages)  # 1 (only system message)
        """
        self.messages.clear()
        if self._initial_system_message:
            self.messages.append(self._initial_system_message)
        
        # Reset context handler cache (if handler has reset method)
        if hasattr(self.context_handler, 'reset') and callable(getattr(self.context_handler, 'reset')):
            self.context_handler.reset()  # type: ignore

    def stats(self) -> Dict[str, Any]:
        """
        Get conversation statistics.
        
        Returns:
            Dictionary containing:
            - total_messages: Total number of messages
            - by_type: Message count by type
            - total_tokens: Total tokens (displayed as xxK format)
            - input_tokens: Input tokens
            - output_tokens: Output tokens
        
        Example:
            >>> conv.stats()
            {
                'total_messages': 10,
                'by_type': {
                    'user': 5,
                    'assistant': 4,
                    'context': 1
                },
                'total_tokens': 12543,
                'input_tokens': 8234,
                'output_tokens': 4309
            }
        """
        stats = {
            'total_messages': len(self.messages),
            'by_type': {},
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0
        }
        
        # Count messages by type
        for msg in self.messages:
            msg_type = msg.role
            stats['by_type'][msg_type] = stats['by_type'].get(msg_type, 0) + 1
            
            # Count tokens (from metadata.usage)
            usage = getattr(msg.metadata, 'usage', None)
            if usage is not None:
                stats['total_tokens'] += usage.total_tokens
                stats['input_tokens'] += usage.prompt_tokens
                stats['output_tokens'] += usage.completion_tokens
        
        return stats
    
    def get_messages(
        self,
        turn_ids: Optional[Union[str, List[str]]] = None,
        turns: Optional[Union[int, tuple[int, int]]] = None,
        messages: Optional[Union[int, tuple[int, int]]] = None,
        roles: Optional[Union[str, List[str]]] = None,
        has_tool_calls: Optional[bool] = None,
        has_attachments: Optional[bool] = None,
        message_ids: Optional[Union[str, List[str]]] = None,
    ) -> List[Message]:
        """
        Get filtered messages from conversation.
        
        All filters are combined with AND logic.
        
        Args:
            turn_ids: Filter by turn ID(s)
            turns: Filter by turn index/range
                - Positive int: First N turns (e.g., 3 = first 3 turns)
                - Negative int: Last N turns (e.g., -3 = last 3 turns)
                - Tuple: Turn range (start, end) - e.g., (1, 4) = turns 1-3 (0-indexed)
            messages: Filter by message index/range
                - Positive int: First N messages
                - Negative int: Last N messages
                - Tuple: Message range (start, end) - standard Python slice
            roles: Filter by role(s) - "user", "assistant", "system", "tool"
            has_tool_calls: Filter messages with/without tool calls
            has_attachments: Filter messages with/without attachments
            message_ids: Filter by message ID(s)
        
        Returns:
            Filtered list of messages
        
        Examples:
            # Last 3 turns
            conv.get_messages(turns=-3)
            
            # First 2 turns
            conv.get_messages(turns=2)
            
            # Turn 1 to 3 (0-indexed)
            conv.get_messages(turns=(1, 4))
            
            # Last 10 messages
            conv.get_messages(messages=-10)
            
            # First 5 messages
            conv.get_messages(messages=5)
            
            # Message 10 to 20
            conv.get_messages(messages=(10, 20))
            
            # All user messages
            conv.get_messages(roles="user")
            
            # Get messages from specific turns
            conv.get_messages(turn_ids=["abc-123", "def-456"])
            
            # All assistant messages with tool calls
            conv.get_messages(roles="assistant", has_tool_calls=True)
            
            # Last 2 turns, only user and assistant
            conv.get_messages(turns=-2, roles=["user", "assistant"])
        """
        result = list(self.messages)
        
        # Filter by messages index/range first (before other filters)
        if messages is not None:
            if isinstance(messages, tuple):
                # Range: (start, end)
                start, end = messages
                result = result[start:end]
            elif messages > 0:
                # Positive: first N messages
                result = result[:messages]
            else:
                # Negative: last N messages
                result = result[messages:]
        
        # Filter by turns index/range
        if turns is not None:
            # Get unique turn IDs in order
            seen = set()
            turn_ids_in_order = []
            for msg in result:
                if msg.turn_id and msg.turn_id not in seen:
                    turn_ids_in_order.append(msg.turn_id)
                    seen.add(msg.turn_id)
            
            # Select turns based on parameter type
            if isinstance(turns, tuple):
                # Range: (start, end)
                start, end = turns
                selected_turn_ids = set(turn_ids_in_order[start:end])
            elif turns > 0:
                # Positive: first N turns
                selected_turn_ids = set(turn_ids_in_order[:turns])
            else:
                # Negative: last N turns
                selected_turn_ids = set(turn_ids_in_order[turns:])
            
            result = [msg for msg in result if msg.turn_id in selected_turn_ids]
        
        # Filter by turn_ids
        if turn_ids is not None:
            if isinstance(turn_ids, str):
                turn_ids = [turn_ids]
            turn_id_set = set(turn_ids)
            result = [msg for msg in result if msg.turn_id in turn_id_set]
        
        # Filter by roles
        if roles is not None:
            if isinstance(roles, str):
                roles = [roles]
            role_set = set(roles)
            result = [msg for msg in result if msg.role in role_set]
        
        # Filter by message_ids
        if message_ids is not None:
            if isinstance(message_ids, str):
                message_ids = [message_ids]
            msg_id_set = set(message_ids)
            result = [msg for msg in result if msg.id in msg_id_set]
        
        # Filter by has_tool_calls
        if has_tool_calls is not None:
            if has_tool_calls:
                result = [msg for msg in result if msg.tool_calls]
            else:
                result = [msg for msg in result if not msg.tool_calls]
        
        # Filter by has_attachments
        if has_attachments is not None:
            if has_attachments:
                result = [msg for msg in result if msg.attachments]
            else:
                result = [msg for msg in result if not msg.attachments]
        
        return result
    
    @property
    def user_messages(self) -> List[Message]:
        """Get all user messages."""
        return self.get_messages(roles="user")
    
    @property
    def assistant_messages(self) -> List[Message]:
        """Get all assistant messages."""
        return self.get_messages(roles="assistant")
    
    @property
    def tool_messages(self) -> List[Message]:
        """Get all tool messages."""
        return self.get_messages(roles="tool")
    
    @property
    def turns(self) -> List[str]:
        """
        Get all unique turn IDs in chronological order.
        
        Returns:
            List of turn IDs
        
        Example:
            >>> turn_ids = conv.turns
            >>> print(f"Total turns: {len(turn_ids)}")
            >>> # Get first turn's messages
            >>> first_turn_msgs = conv.get_messages(turn_ids=turn_ids[0])
        """
        seen = set()
        turn_ids = []
        for msg in self.messages:
            if msg.turn_id and msg.turn_id not in seen:
                turn_ids.append(msg.turn_id)
                seen.add(msg.turn_id)
        return turn_ids
    
    @property
    def provider_name(self) -> str:
        """
        Get provider name.
        
        Returns:
            Provider name (e.g., 'openai', 'bailian', 'anthropic')
        
        Example:
            >>> conv = Conversation("openai/gpt-4", api_key="...")
            >>> print(conv.provider_name)  # 'openai'
        """
        return self._provider_name
    
    @property
    def model_name(self) -> str:
        """
        Get model name.
        
        Returns:
            Model name (e.g., 'gpt-4', 'qwen-plus', 'claude-3-opus')
        
        Example:
            >>> conv = Conversation("bailian/qwen-plus", api_key="...")
            >>> print(conv.model_name)  # 'qwen-plus'
        """
        return self._model_name

    def close(self):
        """Close the provider and cleanup executors."""
        if hasattr(self, 'provider'):
            self.provider.close()
        
        # Shutdown executor pools
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
