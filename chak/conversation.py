import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from enum import Enum
from typing import TYPE_CHECKING, List, Dict, Any, Iterator, Union, Optional, AsyncIterator

from .attachment import Attachment
from .context.strategies import BaseContextStrategy, NoopStrategy
from .context.strategies.base import StrategyRequest
from .message import Message, MessageChunk, HumanMessage, AIMessage, SystemMessage, ToolMessage, MarkerMessage
from .providers import create_provider
from .providers.types import ProviderCategory
from .utils.uri import parse as parse_uri

if TYPE_CHECKING:
    from .tools.mcp.tool import MCPTool
    from .tools.manager import ToolManager


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
        system_message: Optional[str] = None,
        context_strategy: Optional[BaseContextStrategy] = None,
        tools: Optional[List["MCPTool"]] = None,
        tool_executor: ToolExecutor = ToolExecutor.ASYNCIO,
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
            system_message: Optional system message to initialize the conversation.
                          If you need structured content, use \n\n to separate sections.
            context_strategy: Context management strategy (default: NoopStrategy)
            tools: Optional list of MCP tools or native functions (requires async asend() method)
            tool_executor: Tool execution mode (default: ToolExecutor.ASYNCIO)
                          - ASYNCIO: Best for IO-bound tasks (API calls, DB queries)
                          - THREAD: ThreadPoolExecutor for sync blocking operations
                          - PROCESS: ProcessPoolExecutor for CPU-bound tasks
            **kwargs: Additional configuration parameters
        
        Example:
            >>> # Simple system message
            >>> conv = Conversation(
            ...     model_uri="openai:gpt-4",
            ...     api_key="sk-...",
            ...     system_message="You are a helpful assistant."
            ... )
            >>> 
            >>> # Structured system message
            >>> system_prompt = (
            ...     "You are a helpful assistant.\n\n"
            ...     "Rules:\n"
            ...     "- Always respond in Chinese\n"
            ...     "- Be concise and professional"
            ... )
            >>> conv = Conversation(
            ...     model_uri="openai:gpt-4",
            ...     api_key="sk-...",
            ...     system_message=system_prompt
            ... )
        """
        self.model_uri = model_uri
        self.api_key = api_key
        self.messages = []
        self.attachments: List[Attachment] = []  # Session-level attachment tracking
        
        # Tool management
        self._raw_tools: List = []  # Store original tools
        self._tool_manager: Optional["ToolManager"] = None
        
        # Tool executor configuration
        self._tool_executor = tool_executor
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        
        # Initialize tools if provided
        if tools:
            self.add_tools(tools)
        
        # Initialize system message
        self._initial_system_message = self._normalize_system_message(system_message)
        if self._initial_system_message:
            self.messages.append(self._initial_system_message)
        
        # Initialize context strategy
        self.context_strategy = context_strategy or NoopStrategy()

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

    def _normalize_system_message(self, system_message: Optional[str]) -> Optional[SystemMessage]:
        """
        Convert system message string to SystemMessage object.
        
        Args:
            system_message: System message string
            
        Returns:
            SystemMessage object, or None if input is empty
        """
        if not system_message:
            return None
        
        if not isinstance(system_message, str):
            raise TypeError(f"system_message must be str, got {type(system_message)}")
        
        return SystemMessage(content=system_message)
    
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
        self._tool_manager = ToolManager(wrapped_tools, executor=executor)
    
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
                elif role == "context":
                    metadata = msg.get('metadata', {})
                    if isinstance(metadata, dict):
                        self.messages.append(MarkerMessage(content=content, metadata=metadata))
                    else:
                        self.messages.append(MarkerMessage(content=content))
                else:
                    raise ValueError(f"Invalid role: {role}")
            elif isinstance(msg, (HumanMessage, AIMessage, SystemMessage, ToolMessage, MarkerMessage)):
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
            **kwargs
    ) -> Union[Message, Iterator[MessageChunk], Any]:
        """
        Send message (sync, no MCP tools support).
        
        Supports:
        - ✅ Streaming
        - ✅ Non-streaming
        - ✅ Multimodal (images, audio)
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
            - If stream=True: Iterator[MessageChunk]
        
        Raises:
            RuntimeError: If tools are configured
        
        Examples:
            # Simple usage
            response = conv.send("Hello")
            
            # With streaming
            for chunk in conv.send("Hello", stream=True):
                print(chunk.content, end="")
            
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

        # Apply context strategy
        messages_to_send = self._apply_context_strategy()

        # Normal LLM call (no tools)
        if stream:
            return self._send_stream(messages_to_send, **kwargs)
        else:
            return self._send_nonstream(messages_to_send, **kwargs)
    
    async def asend(
            self,
            message: Union[str, Message],
            attachments: Optional[List[Attachment]] = None,
            stream: bool = False,
            event: bool = False,
            timeout: Optional[int] = None,
            returns: Optional[type] = None,
            tool_executor: Optional[ToolExecutor] = None,
            **kwargs
    ) -> Union[Message, AsyncIterator[MessageChunk], AsyncIterator['StreamEvent'], Any, None]:
        """
        Send message (async, full featured).
        
        Supports:
        - ✅ Streaming
        - ✅ Non-streaming
        - ✅ Multimodal (images, audio)
        - ✅ MCP tools (both modes)
        - ✅ Structured output (returns parameter)
        - ✅ Event stream (for tool observability)
        
        Args:
            message: Message content (str will be converted to HumanMessage)
            attachments: Optional list of Attachment objects (images, audio, etc.)
            stream: Enable streaming (ignored if event=True)
            event: Enable event stream mode (returns MessageChunk + ToolCall events)
                  When True, you can observe tool calls in real-time using isinstance() or match-case.
                  Note: event=True will override stream parameter.
            timeout: Request timeout in seconds. If None, uses provider's default timeout (30s)
            returns: Optional Pydantic model class for structured output. When provided,
                    forces LLM to return data matching this schema via function calling.
                    Returns None if extraction fails.
            tool_executor: Optional override for tool execution mode (for this call only)
            **kwargs: Additional LLM parameters
        
        Returns:
            - If event=False and stream=False and returns=None: Complete Message
            - If event=False and stream=True: AsyncIterator[MessageChunk]
            - If event=True: AsyncIterator[StreamEvent] (MessageChunk/ToolCallStartEvent/ToolCallSuccessEvent/ToolCallErrorEvent)
            - If returns is provided: Validated Pydantic model instance or None if failed
        
        Examples:
            # Non-streaming
            response = await conv.asend("Hello")
            
            # Streaming
            async for chunk in await conv.asend("Hello", stream=True):
                print(chunk.content, end="")
            
            # Event stream (with tool observability)
            async for event in await conv.asend("What's the weather?", event=True):
                match event:
                    case MessageChunk(content=text):
                        print(text, end="")
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

        # Apply context strategy
        messages_to_send = self._apply_context_strategy()

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
            # Normal LLM mode
            if stream:
                return self._asend_stream(messages_to_send, **kwargs)
            else:
                return await self._asend_nonstream(messages_to_send, **kwargs)
    
    async def _asend_stream(self, messages: List[Message], **kwargs) -> AsyncIterator[MessageChunk]:
        """Handle async streaming response."""
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
        last_chunk_was_final = False
        last_chunk_metadata = {}
        
        for provider_chunk in provider_chunks:
            chunk = self.provider.converter.from_provider_chunk(provider_chunk)
            complete_content += chunk.content
            
            # Check if this chunk is already marked as final
            if chunk.is_final:
                last_chunk_was_final = True
            
            # Save metadata from last chunk (may contain usage info)
            if chunk.metadata:
                last_chunk_metadata = chunk.metadata
            
            yield chunk

        # Only send additional final chunk if provider didn't send one
        if complete_content and not last_chunk_was_final:
            final_message = AIMessage(
                content=complete_content,
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
                metadata=last_chunk_metadata
            )
            self.messages.append(final_message)
    
    async def _asend_stream_with_tools(self, messages: List[Message], **kwargs) -> AsyncIterator[MessageChunk]:
        """Handle async streaming response with MCP tools."""
        if not self._tool_manager:
            raise RuntimeError("Tool manager not initialized")
        
        complete_content = ""
        
        # Use tool manager's streaming loop
        async for chunk in self._tool_manager.execute_loop_stream(
            provider=self.provider,
            messages=messages,
            model_uri=self.model_uri
        ):
            complete_content += chunk.content
            yield chunk
        
        # Save final message
        if complete_content:
            final_message = AIMessage(content=complete_content)
            self.messages.append(final_message)
    
    async def _asend_nonstream_with_tools(self, messages: List[Message], **kwargs) -> Message:
        """Handle async non-streaming response with MCP tools."""
        if not self._tool_manager:
            raise RuntimeError("Tool manager not initialized")
        
        response = await self._tool_manager.execute_loop(
            provider=self.provider,
            messages=messages,
            model_uri=self.model_uri
        )
        self.messages.append(response)
        return response
    
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
        
        Args:
            message: User message
            attachments: Optional attachments
            returns: Pydantic BaseModel class
            **kwargs: Additional LLM parameters
            
        Returns:
            Instance of the Pydantic model
        """
        from pydantic import BaseModel
        import json
        
        # Validate returns is a Pydantic model
        if not (isinstance(returns, type) and issubclass(returns, BaseModel)):
            raise TypeError(
                f"returns must be a Pydantic BaseModel subclass, got {type(returns)}"
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
        
        # Apply context strategy
        messages_to_send = self._apply_context_strategy()
        
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
            model: Pydantic BaseModel class
            
        Returns:
            OpenAI tool definition dict
        """
        schema = model.model_json_schema()
        
        # Extract description from model docstring or use default
        description = model.__doc__ or f"Correctly extracted `{model.__name__}` with all required parameters"
        description = description.strip()
        
        return {
            "type": "function",
            "function": {
                "name": model.__name__,
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
        if not isinstance(response, (HumanMessage, AIMessage, SystemMessage, ToolMessage, MarkerMessage)):
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
            StreamEvent: MessageChunk, ToolCallStartEvent, ToolCallSuccessEvent, or ToolCallErrorEvent
        """
        from .message import MessageChunk
        
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
            # No tools: only return content events
            async for chunk in self._asend_stream(messages, **kwargs):
                yield MessageChunk(
                    content=chunk.content,
                    is_final=chunk.is_final,
                    metadata=chunk.metadata,
                    final_message=chunk.final_message
                )

    def _send_nonstream(self, messages: List[Message], **kwargs) -> Message:
        """Handle non-streaming response."""
        response = self.provider.send(
            messages=messages,
            stream=False,
            **kwargs
        )
        # Convert to AIMessage
        if not isinstance(response, (HumanMessage, AIMessage, SystemMessage, ToolMessage, MarkerMessage)):
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
    
    def _send_stream(self, messages: List[Message], **kwargs) -> Iterator[MessageChunk]:
        """Handle streaming response."""
        # Get provider chunks (model is already in provider config)
        provider_chunks = self.provider.send(
            messages=messages,
            stream=True,
            **kwargs
        )

        # Convert to standard chunks and collect content
        complete_content = ""
        last_chunk_was_final = False
        last_chunk_metadata = {}
        
        for provider_chunk in provider_chunks:
            chunk = self.provider.converter.from_provider_chunk(provider_chunk)
            complete_content += chunk.content
            
            # Check if this chunk is already marked as final
            if chunk.is_final:
                last_chunk_was_final = True
            
            # Save metadata from last chunk (may contain usage info)
            if chunk.metadata:
                last_chunk_metadata = chunk.metadata
            
            yield chunk

        # Only send additional final chunk if provider didn't send one
        if complete_content and not last_chunk_was_final:
            final_message = AIMessage(
                content=complete_content,
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
                metadata=last_chunk_metadata
            )
            self.messages.append(final_message)

    def _apply_context_strategy(self) -> List[Message]:
        """
        Apply context strategy to process messages.
        
        Returns:
            Complete processed message list (strategy may insert markers)
        """
        if not self.messages:
            return []
        
        # Build strategy request
        request = StrategyRequest(messages=self.messages)
        
        # Get strategy response
        response = self.context_strategy.process(request)
        
        # Update messages (may include markers)
        self.messages = response.messages
        
        # Extract messages to send: system messages + last marker (inclusive) → end
        messages_to_send = self._extract_messages_to_send(response.messages)
        
        # Convert MarkerMessage to SystemMessage for LLM compatibility
        messages_for_llm = self._prepare_for_llm(messages_to_send)
        
        return messages_for_llm
    
    def _extract_messages_to_send(self, messages: List[Message]) -> List[Message]:
        """
        Extract messages to send from complete message list.
        
        Extraction rules:
        - Always include all system messages
        - If markers exist: last marker (inclusive) → last message
        - If no markers: all conversation messages
        
        Args:
            messages: Complete message list
            
        Returns:
            Messages to send to LLM
        """
        if not messages:
            return []
        
        # 1. Extract system messages
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        
        # 2. Find last marker
        last_marker_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], MarkerMessage):
                last_marker_idx = i
                break
        
        # 3. Extract messages based on marker presence
        if last_marker_idx is not None:
            # Has marker: from last marker to end
            context_messages = messages[last_marker_idx:]
        else:
            # No marker: all non-system messages
            context_messages = [
                m for m in messages 
                if not isinstance(m, SystemMessage)
            ]
        
        # 4. Combine: system messages + context messages
        return list(system_messages) + list(context_messages)
    
    def _prepare_for_llm(self, messages: List[Message]) -> List[Message]:
        """
        Prepare messages for LLM by converting MarkerMessage to SystemMessage.
        
        Args:
            messages: Messages to send
            
        Returns:
            Messages with context role converted to system
        """
        result: List[Message] = []
        for msg in messages:
            if isinstance(msg, MarkerMessage):
                # Convert to SystemMessage for LLM compatibility
                result.append(SystemMessage(content=msg.content))
            else:
                result.append(msg)
        return result

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
            ...     system_message="You are a helpful assistant."
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
        
        # Reset context strategy cache (if strategy has reset method)
        # Some strategies (like SummarizationStrategy) may have cache that needs cleanup
        if hasattr(self.context_strategy, 'reset') and callable(getattr(self.context_strategy, 'reset')):
            self.context_strategy.reset()  # type: ignore

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
                'total_tokens': '12.5K',
                'input_tokens': '8.2K',
                'output_tokens': '4.3K'
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
            
            # Count tokens (from metadata)
            if 'usage' in msg.metadata:
                usage = msg.metadata['usage']
                # Handle both dict and object types (e.g., CompletionUsage)
                if isinstance(usage, dict):
                    stats['total_tokens'] += usage.get('total_tokens', 0)
                    stats['input_tokens'] += usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
                    stats['output_tokens'] += usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
                elif hasattr(usage, 'total_tokens'):
                    # Object type (e.g., CompletionUsage from OpenAI SDK)
                    stats['total_tokens'] += getattr(usage, 'total_tokens', 0)
                    stats['input_tokens'] += getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'input_tokens', 0)
                    stats['output_tokens'] += getattr(usage, 'completion_tokens', 0) or getattr(usage, 'output_tokens', 0)
        
        # Format token counts (use K for numbers over 1000)
        stats['total_tokens'] = self._format_tokens(stats['total_tokens'])
        stats['input_tokens'] = self._format_tokens(stats['input_tokens'])
        stats['output_tokens'] = self._format_tokens(stats['output_tokens'])
        
        return stats
    
    def _format_tokens(self, tokens: int) -> str:
        """
        Format token count, use K notation for numbers over 1000.
        
        Args:
            tokens: Token count
            
        Returns:
            Formatted string
        """
        if tokens >= 1000:
            return f"{tokens / 1000:.1f}K"
        return str(tokens)
    
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
