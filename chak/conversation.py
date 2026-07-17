import asyncio
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from enum import Enum
from typing import TYPE_CHECKING, List, Dict, Any, Iterator, Union, Optional, AsyncIterator

from .attachment import Attachment
from .context.handlers import BaseContextHandler, NoopContextHandler
from .message import Message, MessageChunk, ReasoningChunk, FailoverChunk, HumanMessage, AIMessage, SystemMessage, ToolMessage, _current_turn_id
from .metadata import ProviderTrace
from .providers import create_provider
from .providers.llm.resilient import FallbackOn, ResilientProvider
from .providers.types import ProviderCategory
from .schemas import Reasoning
from .utils.uri import parse as parse_uri

if TYPE_CHECKING:
    from .tools.mcp.tool import MCPTool
    from .tools.manager import ToolManager, HITLHandler, HITLRequest


def _merge_stream_metadata(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge an incoming chunk metadata dict into the accumulated base.

    Token counts are summed so that prompt_tokens (from message_start) and
    completion_tokens (from message_delta) are both captured correctly even
    though they arrive in separate content-less chunks.

    Since chak normalizes all providers into the canonical disjoint-bucket
    contract (see chak.metadata.Usage), total_tokens is recomputed from the
    four disjoint sums here — no provider-specific branching needed.
    """
    merged = {**base, **incoming}
    old_usage = (base.get('usage') or {})
    new_usage = (incoming.get('usage') or {})
    if old_usage or new_usage:
        pt = (old_usage.get('prompt_tokens') or 0) + (new_usage.get('prompt_tokens') or 0)
        ct = (old_usage.get('completion_tokens') or 0) + (new_usage.get('completion_tokens') or 0)
        cc = (old_usage.get('cache_creation_input_tokens') or 0) + (new_usage.get('cache_creation_input_tokens') or 0)
        cr = (old_usage.get('cache_read_input_tokens') or 0) + (new_usage.get('cache_read_input_tokens') or 0)
        merged['usage'] = {
            'prompt_tokens': pt,
            'completion_tokens': ct,
            'total_tokens': pt + ct + cc + cr,
            'cache_creation_input_tokens': cc,
            'cache_read_input_tokens': cr,
        }
    return merged


class ToolExecutor(str, Enum):
    """Tool execution mode."""
    ASYNCIO = "asyncio"  # Use asyncio.to_thread (default, best for IO-bound)
    THREAD = "thread"    # Use ThreadPoolExecutor
    PROCESS = "process"  # Use ProcessPoolExecutor (for CPU-bound tasks)


class _ToolConfig:
    """Fluent configuration namespace for tool-related settings.

    Usage:
        conv.tool.verbose.on()              # enable verbose tool logging
        conv.tool.loop.max(10000)           # raise max tool-call iterations
        conv.tool.executor.use(ToolExecutor.THREAD)  # switch execution mode
    """

    class Verbose:
        """Fluent toggle for tool-call verbose logging."""

        def __init__(self):
            self._enabled = False

        @property
        def enabled(self) -> bool:
            return self._enabled

        def on(self) -> None:
            """Enable verbose tool logging (show arguments and results)."""
            self._enabled = True

        def off(self) -> None:
            """Disable verbose tool logging (default, only show tool name and status)."""
            self._enabled = False

        def __bool__(self) -> bool:
            return self._enabled

    class Loop:
        """Fluent config for the tool-calling loop."""

        def __init__(self):
            self._max_iterations = 50
            # Back-reference set by Conversation after construction, so that
            # fluent changes can propagate to the live ToolManager (mirrors
            # Executor.use() below).
            self._owner: Optional["Conversation"] = None

        @property
        def max_iterations(self) -> int:
            return self._max_iterations

        def max(self, n: int) -> None:
            """Set the maximum number of tool-call iterations.

            The loop will raise an error if the LLM keeps requesting tool calls
            beyond this limit, preventing infinite loops.

            Args:
                n: Maximum iterations (must be >= 1). Default is 50.
            """
            if n < 1:
                raise ValueError("max_iterations must be >= 1")
            self._max_iterations = n
            # Propagate to live ToolManager if Conversation is already initialized.
            # ToolManager caches max_iterations at construction time, so without
            # this hop a post-construction ``conv.tool.loop.max(...)`` would be
            # silently ignored by the running loop.
            if self._owner is not None and self._owner._tool_manager is not None:
                self._owner._tool_manager.max_iterations = n

        def unlimited(self) -> None:
            """Remove the iteration limit entirely.

            Use with caution — a stuck LLM will loop forever without this
            safety net.
            """
            import sys
            self._max_iterations = sys.maxsize
            # Propagate to live ToolManager (see ``max`` for rationale).
            if self._owner is not None and self._owner._tool_manager is not None:
                self._owner._tool_manager.max_iterations = sys.maxsize

    class Executor:
        """Fluent config for tool execution mode."""

        def __init__(self):
            self._mode = ToolExecutor.ASYNCIO
            # Back-reference set by Conversation after construction, so that
            # fluent changes can propagate to the live ToolManager.
            self._owner: Optional["Conversation"] = None

        @property
        def mode(self) -> "ToolExecutor":
            return self._mode

        def use(self, mode: "ToolExecutor") -> None:
            """Set the tool execution mode.

            Args:
                mode: ToolExecutor.ASYNCIO (default), THREAD, or PROCESS
            """
            if not isinstance(mode, ToolExecutor):
                raise TypeError(f"Expected ToolExecutor, got {type(mode).__name__}")
            self._mode = mode
            # Propagate to live ToolManager if Conversation is already initialized
            if self._owner is not None:
                self._owner._tool_executor = mode
                if self._owner._tool_manager:
                    self._owner._tool_manager.executor = self._owner._get_executor()

    def __init__(self):
        self.verbose = _ToolConfig.Verbose()
        self.loop = _ToolConfig.Loop()
        self.executor = _ToolConfig.Executor()


class _FallbackConfig:
    """Fluent config for provider fallback strategy.

    Usage:
        conv.fallback.on(FallbackOn.RETRYABLE_ERRORS)
    """

    def __init__(self):
        self._mode = FallbackOn.ALL_ERRORS

    @property
    def mode(self) -> FallbackOn:
        return self._mode

    def on(self, mode: FallbackOn) -> None:
        """Set the fallback trigger condition.

        Args:
            mode: FallbackOn.ALL_ERRORS (default) or RETRYABLE_ERRORS
        """
        if not isinstance(mode, FallbackOn):
            raise TypeError(f"Expected FallbackOn, got {type(mode).__name__}")
        self._mode = mode


class _HookPoint:
    """A single hook registration point (before_send or after_send).

    Supports fluent registration via direct call:
        conv.hook.before_send(my_callback)
        conv.hook.before_send([cb1, cb2])
    """

    def __init__(self):
        self._callbacks: list = []

    def __call__(self, callbacks):
        """Register one or more callbacks.

        Args:
            callbacks: A single callable or a list of callables.
                       Each callback must have the signature:
                       async def callback(conv, request, **send_kwargs) -> None
        """
        if callable(callbacks):
            self._callbacks.append(callbacks)
        elif isinstance(callbacks, list):
            self._callbacks.extend(callbacks)
        else:
            raise TypeError(
                f"Expected a callable or list of callables, got {type(callbacks).__name__}"
            )

    async def _invoke(self, conv, request, **send_kwargs):
        """Execute all registered callbacks in registration order.

        Exceptions propagate to the caller — hooks that need to abort
        the current operation should raise directly.
        """
        for cb in self._callbacks:
            await cb(conv, request, **send_kwargs)


class _HookGroup:
    """Fluent configuration namespace for lifecycle hooks.

    Usage:
        conv.hook.before_send(budget_checker)
        conv.hook.after_send([logger, metrics])
    """

    def __init__(self):
        self.before_send = _HookPoint()
        self.after_send = _HookPoint()


class _CreateSignal:
    """Class-level signal emitted after each Conversation instance is fully built.

    A general extensibility point — any external tool (observability,
    metrics, tracing, debugging) can subscribe without monkey-patching or
    subclassing. Chak core itself never imports any subscriber; the
    dependency direction is strictly one-way (subscriber → core).

    Usage::

        from chak.conversation import Conversation

        Conversation.on_create.subscribe(lambda conv: my_hook(conv))
    """

    def __init__(self):
        self._listeners: list = []

    def subscribe(self, fn) -> None:
        """Register a listener callable. Duplicates are silently ignored."""
        if fn not in self._listeners:
            self._listeners.append(fn)

    def _emit(self, instance) -> None:
        """Notify all listeners. Errors are swallowed so that a buggy
        listener can never crash conversation creation."""
        for fn in self._listeners:
            try:
                fn(instance)
            except Exception:
                pass


class Conversation:
    """
    Chat conversation that follows your desired flow:
    URI -> parse -> dict -> ProviderConfig -> Provider -> client
    
    Conversation专用于LLM类型的provider，用于文本对话交互。
    """
    
    # 类常量：指定Conversation只使用LLM类型的provider
    PROVIDER_CATEGORY = ProviderCategory.LLM

    # Class-level creation signal. External tools subscribe to this to
    # get notified after every Conversation is fully initialized — without
    # monkey-patching __init__. See _CreateSignal docstring.
    on_create = _CreateSignal()

    @staticmethod
    async def _build_human_message(message: Union[str, "Message"], attachments: Optional[List["Attachment"]] = None) -> "HumanMessage":
        """Build a HumanMessage from a string and optional attachments.

        If ``message`` is already a Message object, returns it as-is.
        If no attachments are provided, returns a simple text HumanMessage.
        Otherwise, builds a multimodal content array by reading document
        attachments (async).

        This is the single source of truth for str → HumanMessage conversion
        across all async send paths.
        """
        if not isinstance(message, str):
            return message  # type: ignore[return-value]
        if not attachments:
            return HumanMessage(content=message)

        content_parts: list = [{"type": "text", "text": message}]
        for att in attachments:
            if att.mime_type.is_image():
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": att.source},
                })
            elif att.mime_type.is_audio():
                content_parts.append({
                    "type": "input_audio",
                    "input_audio": {
                        "data": att.source,
                        "format": att.mime_type.subtype,
                    },
                })
            elif att.mime_type.is_video():
                content_parts.append({
                    "type": "video",
                    "video": {"url": att.source},
                })
            elif att.mime_type.is_document() or att.reader:
                doc_result = await att.aread()
                if doc_result and doc_result.content:
                    doc_text = doc_result.content
                    if doc_result.meta:
                        meta_str = ", ".join(
                            f"{k}: {v}" for k, v in doc_result.meta.items()
                            if k != "error"
                        )
                        if meta_str:
                            doc_text = f"[Document metadata: {meta_str}]\n\n{doc_text}"
                    content_parts.append({
                        "type": "text",
                        "text": doc_text,
                    })

        return HumanMessage(
            content=content_parts,
            attachments=list(attachments) if attachments else [],
        )

    def __init__(
        self, 
        model_uri: str, 
        api_key: str,
        id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        context_handler: Optional[BaseContextHandler] = None,
        tools: Optional[List["MCPTool"]] = None,
        tool_executor: Optional[ToolExecutor] = None,
        hitl_handler: Optional["HITLHandler"] = None,
        fallback_on: Optional[FallbackOn] = None,
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
            tool_executor: [Deprecated] Use conv.tool.executor.use() instead.
            fallback_on: [Deprecated] Use conv.fallback.on() instead.
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
        self.title: Optional[str] = None
        self.messages = []
        self.attachments: List[Attachment] = []  # Session-level attachment tracking
        
        # Tool management
        self._raw_tools: List = []  # Store original tools
        self._tool_manager: Optional["ToolManager"] = None
        
        # Initialize fluent configs first
        self.tool = _ToolConfig()
        self.fallback = _FallbackConfig()
        self.hook = _HookGroup()

        # Handle deprecated `tool_executor` parameter
        if tool_executor is not None:
            warnings.warn(
                "Parameter 'tool_executor' is deprecated and will be removed in v0.5. "
                "Use conv.tool.executor.use(ToolExecutor.XXX) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.tool.executor.use(tool_executor)

        # Handle deprecated `fallback_on` parameter
        if fallback_on is not None:
            warnings.warn(
                "Parameter 'fallback_on' is deprecated and will be removed in v0.5. "
                "Use conv.fallback.on(FallbackOn.XXX) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.fallback.on(fallback_on)

        # Tool executor configuration (read from fluent config)
        self._tool_executor = self.tool.executor.mode
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._hitl_handler: Optional["HITLHandler"] = hitl_handler
        # Wire back-reference so fluent Executor.use() can propagate to live ToolManager
        self.tool.executor._owner = self
        # Same for Loop.max() / Loop.unlimited() — without this the ToolManager
        # would keep the max_iterations snapshotted at construction time.
        self.tool.loop._owner = self
        
        # Initialize tools if provided
        if tools:
            self.add_tools(tools)
        
        # Initialize system prompt
        self._initial_system_message = self._normalize_system_message(system_prompt)
        if self._initial_system_message:
            self.messages.append(self._initial_system_message)
        
        # Initialize context handler
        self.context_handler = context_handler or NoopContextHandler()

        fallbacks = kwargs.pop('fallbacks', None)

        primary_parsed = parse_uri(model_uri)

        if fallbacks:
            primary_config = self._build_config_dict(primary_parsed, kwargs, api_key=api_key)
            primary_config['provider_name'] = primary_parsed['provider']
            primary_provider = create_provider(
                primary_parsed['provider'],
                primary_config,
                category=self.PROVIDER_CATEGORY
            )
            fallback_providers = []
            for fallback in fallbacks:
                if not isinstance(fallback, dict):
                    raise TypeError("fallback model spec must be a dict")

                fallback_spec = dict(fallback)
                fallback_model_uri = fallback_spec.pop('model_uri', None)
                if not fallback_model_uri:
                    raise ValueError("fallback model spec requires 'model_uri'")
                fallback_api_key = fallback_spec.pop('api_key', self.api_key)
                nested_kwargs = fallback_spec.pop('kwargs', None)
                fallback_kwargs: Dict[str, Any] = {}
                if nested_kwargs is not None:
                    if not isinstance(nested_kwargs, dict):
                        raise TypeError("fallback model 'kwargs' must be a dict")
                    fallback_kwargs.update(nested_kwargs)
                fallback_kwargs.update(fallback_spec)

                fallback_parsed = parse_uri(fallback_model_uri)
                fallback_config = self._build_config_dict(fallback_parsed, fallback_kwargs, api_key=fallback_api_key)
                fallback_config['provider_name'] = fallback_parsed['provider']
                fallback_provider = create_provider(
                    fallback_parsed['provider'],
                    fallback_config,
                    category=self.PROVIDER_CATEGORY
                )
                fallback_providers.append(fallback_provider)
            self.provider = ResilientProvider(
                primary_provider,
                fallback_providers,
                fallback_on=self.fallback.mode,
            )
        else:
            config_dict = self._build_config_dict(primary_parsed, kwargs)
            config_dict['provider_name'] = primary_parsed['provider']
            self.provider = create_provider(
                primary_parsed['provider'],
                config_dict,
                category=self.PROVIDER_CATEGORY
            )
        
        # Store provider name and model name for easy access
        self._provider_name = primary_parsed['provider']
        self._model_name = primary_parsed['model']

        # Notify class-level creation signal — subscribers (e.g. inspector)
        # can observe new conversations without monkey-patching.
        type(self).on_create._emit(self)

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
        
        .. deprecated::
            Use ``conv.tool.executor.use(ToolExecutor.XXX)`` instead.
        
        Args:
            executor: New execution mode (ToolExecutor.ASYNCIO/THREAD/PROCESS)
        """
        warnings.warn(
            "set_tool_executor() is deprecated and will be removed in v0.5. "
            "Use conv.tool.executor.use(ToolExecutor.XXX) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.tool.executor.use(executor)
    
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

    def _purge_turn(self, turn_id: str, att_snap: int) -> None:
        """Drop every message produced in *turn_id* and rewind attachments.

        Called from the outermost ``except`` of every send/asend entry point
        so a half-finished turn (provider 4xx/5xx, transient network error,
        tool failure, etc.) leaves ``self.messages`` / ``self.attachments``
        byte-identical to the state before the call.  This is what lets an
        upper-layer retry simply re-invoke ``conv.asend(text)`` without
        duplicating the user message or accumulating partial tool-loop
        messages.
        """
        if turn_id is not None:
            self.messages[:] = [
                m for m in self.messages if getattr(m, "turn_id", None) != turn_id
            ]
        if len(self.attachments) > att_snap:
            del self.attachments[att_snap:]
    
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

    def get_provider_traces(self) -> List[ProviderTrace]:
        """Collect all non-empty provider traces from messages in this conversation.

        Returns:
            List of ProviderTrace objects, one per message that has a trace.
            Empty list if no messages have provider traces.

        Example:
            >>> traces = conv.get_provider_traces()
            >>> for t in traces:
            ...     if t.fallback_used:
            ...         print(f"Failover: {t.primary_provider} -> {t.resolved_provider}")
        """
        traces: List[ProviderTrace] = []
        for msg in self.messages:
            metadata = getattr(msg, "metadata", None)
            if metadata is not None:
                trace = getattr(metadata, "provider_trace", None)
                if trace is not None:
                    traces.append(trace)
        return traces
    
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
            max_iterations=self.tool.loop.max_iterations,
            executor=executor,
            hitl_handler=self._hitl_handler,
            verbose=self.tool.verbose,
        )
    
    def _build_config_dict(self, parsed_uri: Dict, kwargs: Dict, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Build configuration dictionary from URI and kwargs."""
        config_dict = {}

        # Core config from URI
        config_dict['api_key'] = api_key or self.api_key
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
        # Snapshot attachments length so we can rewind on failure (messages
        # are tagged with turn_id and purged separately).
        att_snap = len(self.attachments)

        # Build send_kwargs snapshot for hooks (read-only)
        send_kwargs = {
            'timeout': timeout,
            'stream': stream,
            'event': False,
            'returns': returns,
            'reasoning': reasoning,
            **kwargs,
        }
        
        try:
            # Check if tools are configured
            if self._raw_tools:
                raise RuntimeError(
                    "Tool calling requires async execution. "
                    "Please use asyncio.run(conv.asend(message)) or await conv.asend(message) in async context."
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
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise RuntimeError(
                    "Cannot use sync send() in async context. "
                    "Please use: await conv.asend(message)"
                )
            
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

            # before_send hook — fires before message is appended to history
            asyncio.run(self.hook.before_send._invoke(self, user_message, **send_kwargs))
            
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
                # Wrap so iteration-time errors also rollback the turn.
                def _stream_wrap():
                    try:
                        yield from self._send_stream(messages_to_send, **kwargs)
                    except BaseException:
                        self._purge_turn(turn_id, att_snap)
                        raise
                    finally:
                        asyncio.run(self.hook.after_send._invoke(self, user_message, **send_kwargs))
                return _stream_wrap()
            else:
                result = self._send_nonstream(messages_to_send, **kwargs)
                asyncio.run(self.hook.after_send._invoke(self, user_message, **send_kwargs))
                return result
        except BaseException:
            # All-or-nothing: drop everything this turn produced so the caller
            # can safely retry the same conv.send(text) call.
            self._purge_turn(turn_id, att_snap)
            raise
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
        # Snapshot attachments length so we can rewind on failure (messages
        # are tagged with turn_id and purged separately).
        att_snap = len(self.attachments)
        
        # Build send_kwargs snapshot for hooks (read-only)
        stream_send_kwargs = {
            'timeout': timeout,
            'stream': stream,
            'event': event,
            'returns': returns,
            'reasoning': reasoning,
            **kwargs,
        }

        # Prepare user_message once for all paths (hooks + streaming + non-streaming)
        user_message = await self._build_human_message(message, attachments)

        # before_send hook — fires once for all modes
        await self.hook.before_send._invoke(self, user_message, **stream_send_kwargs)

        # Check if event stream mode is requested FIRST (before try block)
        # This must be handled separately because it returns an async generator
        if event:
            # Event stream mode: returns MessageChunk + ToolCall events
            # This overrides stream parameter and provides full observability
            # NOTE: Cannot use try-finally here for ContextVar management because
            # we're returning a generator. The generator execution happens AFTER
            # this function returns, so finally would reset ContextVar too early.
            # Solution: consume the generator within the try block using yield from
            async def _event_stream_wrapper():
                try:
                    async for evt in self._asend_with_events_impl(
                        message=user_message,
                        attachments=attachments,
                        tool_executor=tool_executor,
                        **kwargs
                    ):
                        yield evt
                except BaseException:
                    self._purge_turn(turn_id, att_snap)
                    raise
                finally:
                    await self.hook.after_send._invoke(self, user_message, **stream_send_kwargs)
                    _current_turn_id.reset(token)
            
            return _event_stream_wrapper()
        
        # Check if stream mode is requested (also returns async generator)
        if stream:
            async def _stream_wrapper():
                try:
                    async for chunk in self._asend_stream_impl(
                        message=user_message,
                        attachments=attachments,
                        tool_executor=tool_executor,
                        **kwargs
                    ):
                        yield chunk
                except BaseException:
                    self._purge_turn(turn_id, att_snap)
                    raise
                finally:
                    await self.hook.after_send._invoke(self, user_message, **stream_send_kwargs)
                    _current_turn_id.reset(token)
            
            return _stream_wrapper()
        
        try:
            # Merge timeout into kwargs if specified
            if timeout is not None:
                kwargs['timeout'] = timeout

            # Build send_kwargs snapshot for hooks (read-only)
            send_kwargs = {
                'timeout': timeout,
                'stream': stream,
                'event': event,
                'returns': returns,
                'reasoning': reasoning,
                **kwargs,
            }

            # Structured output without tools: fast path.
            # user_message was already built and before_send already fired above.
            if returns is not None and not self._tool_manager:
                self.messages.append(user_message)
                if attachments:
                    self.attachments.extend(attachments)

                result = None
                try:
                    result = await self._asend_with_structured_output(
                        message=user_message,
                        attachments=attachments,
                        returns=returns,
                        **kwargs
                    )
                except Exception as e:
                    # Structured output failed, log and return None.
                    # Purge so the caller's retry doesn't see a stale
                    # user_message accumulating each attempt.
                    from .utils.logger import logger
                    logger.warning(f"Structured output failed: {type(e).__name__}: {e}")
                    self._purge_turn(turn_id, att_snap)
                    return None

                # after_send runs OUTSIDE the try block so hook exceptions
                # propagate upward (e.g. BudgetExceededError from budget guard).
                await self.hook.after_send._invoke(self, user_message, **send_kwargs)
                return result

            # Append the user_message (prepared above in the hook section)
            self.messages.append(user_message)

            # Track attachments at conversation level
            if attachments:
                self.attachments.extend(attachments)

            # Apply context handler
            messages_to_send = self._apply_context_handler()

            # Check if tools are configured
            if self._tool_manager:
                # Temporarily override executor if specified for this call
                original_executor = None
                if tool_executor is not None:
                    original_executor = self._tool_manager.executor
                    self._tool_manager.executor = self._get_executor(tool_executor)

                try:
                    if returns is not None:
                        # Two-phase flow: run the tool-calling loop first so that
                        # ClaudeSkill (and any other tools) can activate normally,
                        # then run a schema-forced extraction pass on the resulting
                        # conversation history.
                        result = None
                        try:
                            await self._asend_nonstream_with_tools(messages_to_send, **kwargs)
                            # Re-apply context handler so extraction sees updated history.
                            # The tool loop always ends with an AIMessage (final answer).
                            # Most LLMs refuse a tool_choice-forced call when the
                            # conversation already ends on an AI turn, causing all 3
                            # retry attempts to return plain text instead of a tool call.
                            # A bridge HumanMessage restores the expected
                            # human-turn → assistant-tool-call pattern without
                            # polluting self.messages (the bridge lives only in the
                            # local copy inside _run_extraction_loop).
                            #
                            # Crucially, we quote the LLM's own last response verbatim
                            # so it understands the task is pure reformatting, not
                            # new reasoning.  This prevents "complete answer" refusals
                            # where the model declines to call the tool because it
                            # believes it already finished.
                            extraction_messages = list(self._apply_context_handler())
                            last_ai_content = next(
                                (
                                    msg.content
                                    for msg in reversed(extraction_messages)
                                    if getattr(msg, 'role', None) == 'assistant'
                                    and isinstance(msg.content, str)
                                    and msg.content.strip()
                                ),
                                "",
                            )
                            if last_ai_content:
                                bridge_content = (
                                    "Based on the following content you just provided, "
                                    "please call the required function to structure it "
                                    "into the requested format. "
                                    "Do not add or remove information — only restructure:\n\n"
                                    f'\'\'\'\n{last_ai_content}\n\'\'\''
                                )
                            else:
                                bridge_content = (
                                    "Based on your analysis above, please call the required "
                                    "function to provide the structured output."
                                )
                            extraction_messages.append(HumanMessage(content=bridge_content))
                            result = await self._run_extraction_loop(extraction_messages, returns, **kwargs)
                        except Exception as e:
                            from .utils.logger import logger
                            logger.warning(f"Structured output failed: {type(e).__name__}: {e}")
                            # Purge so the caller's retry doesn't see partial
                            # tool-loop messages accumulating each attempt.
                            self._purge_turn(turn_id, att_snap)
                            return None

                        # after_send runs OUTSIDE the try block so hook exceptions
                        # propagate upward (e.g. BudgetExceededError from budget guard).
                        await self.hook.after_send._invoke(self, user_message, **send_kwargs)
                        return result
                    else:
                        # Normal tool-calling mode
                        result = await self._asend_nonstream_with_tools(messages_to_send, **kwargs)
                        await self.hook.after_send._invoke(self, user_message, **send_kwargs)
                        return result
                finally:
                    # Restore original executor
                    if original_executor is not None:
                        self._tool_manager.executor = original_executor
            else:
                # Add reasoning to kwargs if provided
                if reasoning is not None:
                    kwargs['reasoning'] = reasoning

                # Normal LLM mode (non-stream only, stream handled by wrapper above)
                result = await self._asend_nonstream(messages_to_send, **kwargs)
                await self.hook.after_send._invoke(self, user_message, **send_kwargs)
                return result
        except BaseException:
            # All-or-nothing: drop everything this turn produced so the caller
            # can safely retry the same conv.asend(text) call.
            self._purge_turn(turn_id, att_snap)
            raise
        finally:
            # Reset turn ID context
            _current_turn_id.reset(token)
    
    async def _asend_stream(self, messages: List[Message], **kwargs) -> AsyncIterator[Union[MessageChunk, ReasoningChunk, FailoverChunk]]:
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
            if isinstance(provider_chunk, FailoverChunk):
                complete_content = ""
                complete_reasoning_content = ""
                last_chunk_was_final = False
                last_chunk_metadata = {}
                yield provider_chunk
                continue

            unified_chunk = self.provider.converter.from_provider_chunk(provider_chunk)
            
            # Always accumulate metadata regardless of content.
            # Anthropic sends prompt_tokens in message_start and completion_tokens
            # in message_delta — both arrive as content-less chunks, so we must
            # track metadata outside the content/reasoning guards.
            if unified_chunk.metadata:
                last_chunk_metadata = _merge_stream_metadata(last_chunk_metadata, unified_chunk.metadata)
            
            # Handle reasoning content
            if unified_chunk.reasoning_content:
                complete_reasoning_content += unified_chunk.reasoning_content
                yield ReasoningChunk(
                    content=unified_chunk.reasoning_content,
                    is_final=False,
                    metadata=unified_chunk.metadata
                )
            
            # Handle regular content
            if unified_chunk.content:
                complete_content += unified_chunk.content
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
            # Provider sent final chunk, save the message AND send final MessageChunk
            final_message = AIMessage(
                content=complete_content,
                reasoning_content=complete_reasoning_content if complete_reasoning_content else None,
                metadata=last_chunk_metadata
            )
            self.messages.append(final_message)
            
            # Send final MessageChunk with final_message so callers can capture it
            yield MessageChunk(
                content="",
                is_final=True,
                final_message=final_message
            )
    
    async def _asend_nonstream_with_tools(self, messages: List[Message], **kwargs) -> Message:
        """Handle async non-streaming response with MCP tools."""
        if not self._tool_manager:
            raise RuntimeError("Tool manager not initialized")
        
        final_message, _ = await self._tool_manager.execute_loop(
            provider=self.provider,
            messages=messages,
            model_uri=self.model_uri,
            round_context_fn=self._make_round_context_fn(),
            history=self.messages,
        )
        # ``history=self.messages`` above makes the tool loop append every
        # intermediate AIMessage/ToolMessage directly into conv.messages as it
        # runs, so external observers (inspector, hooks, debuggers) see them
        # arrive one by one instead of all at once at the end.
        return final_message
    
    async def _run_extraction_loop(
        self,
        current_messages: List[Message],
        returns: type,
        **kwargs
    ) -> Any:
        """
        Run the structured output extraction loop against an existing message list.

        Expects self.messages to already contain the user message (and any
        tool-calling round-trips if applicable).  On success, appends the
        extraction AIMessage and a virtual ToolMessage to self.messages.

        Args:
            current_messages: Messages to send to the LLM for extraction.
            returns: Pydantic BaseModel class or generic type (List/Dict).
            **kwargs: Additional LLM parameters.

        Returns:
            Instance of the Pydantic model (or List/Dict of instances).
        """
        from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError as PydanticValidationError, create_model
        from typing import get_origin, get_args
        import json

        # Resolve generic wrappers (List[Model], Dict[str, Model])
        origin = get_origin(returns)
        unwrap_field = None
        original_returns = returns

        if origin is list:
            args = get_args(returns)
            if not args or not (isinstance(args[0], type) and issubclass(args[0], BaseModel)):
                raise TypeError(
                    f"List type must contain a Pydantic BaseModel, "
                    f"got List[{args[0] if args else 'Unknown'}]"
                )
            returns = create_model(
                "ExtractedData",
                __config__=ConfigDict(extra="forbid"),
                items=(original_returns, Field(..., description="Extracted items")),
            )
            unwrap_field = "items"
        elif origin is dict:
            args = get_args(returns)
            if len(args) != 2 or args[0] != str or not (isinstance(args[1], type) and issubclass(args[1], BaseModel)):
                raise TypeError(
                    f"Dict type must be Dict[str, BaseModel], "
                    f"got Dict[{args[0] if len(args) > 0 else 'Unknown'}, "
                    f"{args[1] if len(args) > 1 else 'Unknown'}]"
                )
            returns = RootModel[original_returns]
            unwrap_field = "root"
        elif not (isinstance(returns, type) and issubclass(returns, BaseModel)):
            raise TypeError(
                f"returns must be a Pydantic BaseModel, List[BaseModel], or "
                f"Dict[str, BaseModel], got {type(returns)}"
            )

        # Dispatch on provider capability: if the active provider declares
        # native support for OpenAI-style ``response_format=json_schema``,
        # route through the alternative extraction loop. This exists to
        # rescue structured output on models where forced ``tool_choice``
        # is broken (e.g. Moonshot ``kimi-k3``, whose always-on thinking
        # rejects ``tool_choice='specified'``). Providers that don't
        # override the capability method inherit ``False`` from Provider
        # base, so the classic tool-call flow below stays unchanged for
        # everyone else -- zero behavior change for currently-working paths.
        #
        # Defensive accessors: unit tests inject duck-typed provider mocks
        # that don't necessarily expose ``.config`` or the capability
        # method, so we must not crash when they're missing -- absent
        # capability just means "classic path", same as the real default.
        provider_config = getattr(self.provider, "config", None)
        active_model = getattr(provider_config, "model", "") or ""
        capability_fn = getattr(
            self.provider, "supports_json_schema_response_format", None
        )
        if callable(capability_fn) and capability_fn(active_model):
            return await self._run_extraction_loop_via_response_format(
                current_messages, returns, unwrap_field, **kwargs
            )

        # Generate tool schema and force its usage
        tool_schema = self._generate_tool_schema_from_model(returns)
        extraction_kwargs = dict(kwargs)
        extraction_kwargs['tools'] = [tool_schema]
        extraction_kwargs['tool_choice'] = {
            "type": "function",
            "function": {"name": tool_schema["function"]["name"]}
        }

        # ── Instructor-style extraction loop ──────────────────────────────
        # Each attempt may extend messages with error feedback so the LLM
        # can self-correct.  Up to _MAX_RETRIES tries.
        _MAX_RETRIES = 3
        tool_name = tool_schema["function"]["name"]
        messages = list(current_messages)
        last_error: Exception = RuntimeError("No attempts were made")

        for attempt in range(_MAX_RETRIES):
            # ── Step 1: LLM call ──────────────────────────────────────────
            try:
                response = await asyncio.to_thread(
                    self.provider.send,
                    messages=messages,
                    **extraction_kwargs
                )
            except Exception as e:
                status_code = getattr(e, "status_code", None)
                if status_code is not None and 400 <= status_code < 500:
                    raise
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    continue
                raise last_error

            # ── Step 2: extract tool_call ─────────────────────────────────
            tool_call = None
            if isinstance(response, AIMessage) and response.tool_calls:
                tool_call = response.tool_calls[0]
            elif hasattr(response, 'choices') and response.choices:
                tc = getattr(response.choices[0].message, 'tool_calls', None)
                if tc:
                    tool_call = tc[0]

            if tool_call is None:
                last_error = RuntimeError(
                    f"LLM did not call the '{tool_name}' tool "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES})"
                )
                if attempt < _MAX_RETRIES - 1:
                    ai_content = response.content if isinstance(response, AIMessage) else ""
                    messages = messages + [
                        AIMessage(content=ai_content or ""),
                        HumanMessage(
                            content=(
                                f"You must respond by calling the '{tool_name}' function "
                                f"to provide your answer in the required structured format. "
                                f"Please try again."
                            )
                        ),
                    ]
                continue

            # ── Step 3: parse + validate ──────────────────────────────────
            try:
                arguments = json.loads(tool_call.function.arguments)
                # Try direct validation first. Some providers (e.g. Anthropic)
                # return nested objects as JSON-encoded strings when the schema
                # contains $ref / oneOf. In that case, fall back to recursively
                # decoding string values that are valid JSON objects/arrays.
                def _decode_json_strings(obj):
                    if isinstance(obj, dict):
                        return {k: _decode_json_strings(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_decode_json_strings(v) for v in obj]
                    if isinstance(obj, str):
                        try:
                            parsed = json.loads(obj)
                            if isinstance(parsed, (dict, list)):
                                return _decode_json_strings(parsed)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    return obj

                def _validate_with_string_decode(candidate):
                    """Validate ``candidate`` directly, falling back to a
                    recursive JSON-string decode. Encapsulates the
                    Anthropic-style "nested objects as strings" fallback so we
                    can reuse it inside the envelope-unwrap loop below.
                    """
                    try:
                        return returns.model_validate(candidate)
                    except PydanticValidationError:
                        return returns.model_validate(_decode_json_strings(candidate))

                try:
                    result = _validate_with_string_decode(arguments)
                except PydanticValidationError as outer_err:
                    # Envelope-unwrap fallback.
                    #
                    # Some providers -- observed with DeepSeek on complex
                    # nested schemas under long contexts -- return the tool
                    # payload wrapped inside an envelope, e.g.
                    #
                    #   {"requirement": {...actual payload...}}
                    #   {"data": {...actual payload...}, "file": "unknown"}
                    #
                    # The wrap key is *not* reliably the tool function name.
                    # In practice it often echoes the Pydantic-generated JSON
                    # Schema ``title`` (which defaults to the model class
                    # name), and sometimes it is a generic word like "data".
                    # We therefore avoid hardcoding key names and instead try
                    # each top-level dict value; the first one that validates
                    # wins. This only fires after direct validation has
                    # already failed, so it never affects the happy path.
                    #
                    # When we succeed via this path we surface a WARNING with
                    # the outer key set so operators know the model is not
                    # conforming to the schema. When we fail we log the raw
                    # arguments string (truncated) so post-hoc investigation
                    # doesn't require re-instrumenting chak.
                    from .utils.logger import logger

                    result = None
                    unwrapped_key: Optional[str] = None
                    if isinstance(arguments, dict):
                        for _k, _v in arguments.items():
                            if not isinstance(_v, dict):
                                continue
                            try:
                                result = _validate_with_string_decode(_v)
                                unwrapped_key = _k
                                break
                            except PydanticValidationError:
                                continue

                    if result is None:
                        _raw = tool_call.function.arguments
                        logger.warning(
                            f"Structured output validation failed for tool "
                            f"{tool_name!r}. Raw arguments ({len(_raw)} chars, "
                            f"truncated to 2000): {_raw[:2000]}"
                        )
                        # Preserve the original error so the retry-feedback
                        # message the LLM sees describes the actual field-
                        # level mismatch rather than a spurious secondary
                        # failure from the unwrap probe.
                        raise outer_err

                    logger.warning(
                        f"Structured output: recovered from envelope wrap "
                        f"for tool {tool_name!r}. Provider returned outer "
                        f"keys={list(arguments.keys())}, unwrapped "
                        f"key={unwrapped_key!r}. The model is not conforming "
                        f"to the schema; consider tightening the prompt or "
                        f"switching model."
                    )

                # Unwrap internal wrappers for generic returns
                if unwrap_field is not None:
                    result = getattr(result, unwrap_field)

                # Save the extraction messages to conversation history
                if isinstance(response, AIMessage):
                    self.messages.append(response)
                else:
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

            except (json.JSONDecodeError, PydanticValidationError, ValueError) as e:
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    messages = messages + [
                        AIMessage(content=None, tool_calls=response.tool_calls)
                        if isinstance(response, AIMessage)
                        else AIMessage(content=""),
                        ToolMessage(
                            content=(
                                f"Validation error in your response:\n{e}\n"
                                f"Please fix the error and call '{tool_name}' again "
                                f"with correct values."
                            ),
                            tool_call_id=tool_call.id,
                        ),
                    ]
                continue

        raise last_error

    async def _run_extraction_loop_via_response_format(
        self,
        current_messages: List[Message],
        returns: type,
        unwrap_field: Optional[str],
        **kwargs
    ) -> Any:
        """Alternative structured-output loop using ``response_format=json_schema``.

        Chosen by ``_run_extraction_loop`` when the active provider declares
        ``supports_json_schema_response_format(model) is True``. This exists
        because the default forced-``tool_choice`` path is fundamentally
        incompatible with providers/models that keep reasoning always on
        (notably Moonshot's ``kimi-k3`` family). Instead of asking the model
        to call a specific tool, we constrain the entire response body to
        match a JSON schema — a first-class OpenAI-compat feature that
        coexists with thinking.

        The validation/retry shape mirrors the tool-call path (up to
        ``_MAX_RETRIES`` attempts, self-corrective feedback on failure,
        Anthropic-style JSON-string decode fallback, envelope-unwrap
        fallback for models that put the payload inside a wrapper dict)
        so behavior stays predictable regardless of which path is taken.

        Args:
            current_messages: Messages to send to the LLM (already through
                the context handler by the caller, same as the tool-call path).
            returns: The Pydantic model class to validate against — already
                wrapped for ``List[T]``/``Dict[str, T]`` inputs so we always
                pass the model itself, not the generic type.
            unwrap_field: If ``returns`` is a chak-generated wrapper (list
                or dict), the attribute name to unwrap after validation
                (``"items"`` or ``"root"``). None for direct BaseModel returns.
            **kwargs: Additional LLM parameters passed through to
                ``provider.send``. ``tools``/``tool_choice`` are stripped
                to prevent double-configuration when a caller mixes flows.

        Returns:
            The validated (and unwrapped, when applicable) return value.

        Raises:
            The last error encountered after exhausting retries.
        """
        from pydantic import RootModel, ValidationError as PydanticValidationError
        import json

        # Reuse the same name/description convention as the tool-call path
        # so logs/traces stay comparable across the two flows.
        schema = returns.model_json_schema()
        if isinstance(returns, type) and issubclass(returns, RootModel):
            schema_name = "ExtractedData"
        else:
            schema_name = returns.__name__

        extraction_kwargs = dict(kwargs)
        # Strip any caller-supplied tool config: mixing forced tool_choice
        # with response_format is either rejected by the API or produces
        # ambiguous behavior, and we want the wire request to be exactly
        # what we intend for this code path.
        extraction_kwargs.pop("tools", None)
        extraction_kwargs.pop("tool_choice", None)
        extraction_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
            },
        }

        _MAX_RETRIES = 3
        messages = list(current_messages)
        last_error: Exception = RuntimeError("No attempts were made")

        for attempt in range(_MAX_RETRIES):
            # ── Step 1: LLM call ──────────────────────────────────────────
            try:
                response = await asyncio.to_thread(
                    self.provider.send,
                    messages=messages,
                    **extraction_kwargs
                )
            except Exception as e:
                # Same policy as the tool-call path: 4xx are terminal
                # (bad request / auth / etc — retrying won't help), other
                # errors get up to _MAX_RETRIES chances.
                status_code = getattr(e, "status_code", None)
                if status_code is not None and 400 <= status_code < 500:
                    raise
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    continue
                raise last_error

            # ── Step 2: pull the JSON payload out of message.content ──────
            # response_format guarantees the entire content is a JSON
            # object matching our schema. Handle both chak's normalized
            # AIMessage and the raw SDK ChatCompletion shape defensively.
            if isinstance(response, AIMessage):
                raw_content = response.content or ""
            elif hasattr(response, "choices") and response.choices:
                raw_content = getattr(response.choices[0].message, "content", "") or ""
            else:
                raw_content = ""

            if not raw_content.strip():
                last_error = RuntimeError(
                    f"Empty response.content on attempt {attempt + 1}/{_MAX_RETRIES}"
                )
                if attempt < _MAX_RETRIES - 1:
                    messages = messages + [
                        AIMessage(content=""),
                        HumanMessage(
                            content=(
                                f"Your previous response was empty. You must respond "
                                f"with a JSON object matching the '{schema_name}' schema."
                            )
                        ),
                    ]
                continue

            # ── Step 3: parse + validate (same fallbacks as tool-call path) ──
            try:
                arguments = json.loads(raw_content)

                def _decode_json_strings(obj):
                    # Anthropic-style workaround also lives in the tool-call
                    # path (see there for full rationale). Kept in sync so
                    # behavior is symmetric across the two flows.
                    if isinstance(obj, dict):
                        return {k: _decode_json_strings(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_decode_json_strings(v) for v in obj]
                    if isinstance(obj, str):
                        try:
                            parsed = json.loads(obj)
                            if isinstance(parsed, (dict, list)):
                                return _decode_json_strings(parsed)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    return obj

                def _validate_with_string_decode(candidate):
                    try:
                        return returns.model_validate(candidate)
                    except PydanticValidationError:
                        return returns.model_validate(_decode_json_strings(candidate))

                try:
                    result = _validate_with_string_decode(arguments)
                except PydanticValidationError as outer_err:
                    # Envelope-unwrap fallback -- see the tool-call path for
                    # the full explanation. Same probe strategy so we don't
                    # regress models that already relied on it.
                    from .utils.logger import logger

                    result = None
                    unwrapped_key: Optional[str] = None
                    if isinstance(arguments, dict):
                        for _k, _v in arguments.items():
                            if not isinstance(_v, dict):
                                continue
                            try:
                                result = _validate_with_string_decode(_v)
                                unwrapped_key = _k
                                break
                            except PydanticValidationError:
                                continue

                    if result is None:
                        logger.warning(
                            f"Structured output validation failed for schema "
                            f"{schema_name!r} (response_format path). Raw "
                            f"content ({len(raw_content)} chars, truncated to "
                            f"2000): {raw_content[:2000]}"
                        )
                        raise outer_err

                    logger.warning(
                        f"Structured output: recovered from envelope wrap for "
                        f"{schema_name!r} (response_format path). Provider "
                        f"returned outer keys={list(arguments.keys())}, "
                        f"unwrapped key={unwrapped_key!r}. The model is not "
                        f"conforming to the schema; consider tightening the "
                        f"prompt or switching model."
                    )

                if unwrap_field is not None:
                    result = getattr(result, unwrap_field)

                # ── Step 4: record on conversation history ────────────────
                # No virtual ToolMessage this time -- the extraction wasn't
                # a tool call, so history only gets the assistant reply.
                if isinstance(response, AIMessage):
                    self.messages.append(response)
                else:
                    self.messages.append(AIMessage(content=raw_content))

                return result

            except (json.JSONDecodeError, PydanticValidationError, ValueError) as e:
                last_error = e
                if attempt < _MAX_RETRIES - 1:
                    # Retry feedback: unlike the tool-call path there is no
                    # tool_call_id to attach to, so we use a plain user turn
                    # describing the failure. Include the raw content so the
                    # model can see what it just produced.
                    messages = messages + [
                        AIMessage(content=raw_content),
                        HumanMessage(
                            content=(
                                f"Your previous response failed schema validation:\n{e}\n"
                                f"Please respond again with a valid JSON object matching "
                                f"the '{schema_name}' schema."
                            )
                        ),
                    ]
                continue

        raise last_error

    async def _asend_with_structured_output(
        self,
        message: Union[str, Message],
        attachments: Optional[List[Attachment]],
        returns: type,
        **kwargs
    ) -> Any:
        """
        Handle structured output (no tools configured).

        The caller has already appended the user_message to self.messages.
        This method only applies context handling and runs extraction.

        Supports:
        - BaseModel: returns=MyModel
        - List[BaseModel]: returns=List[MyModel]
        - Dict[str, BaseModel]: returns=Dict[str, MyModel]

        Args:
            message: Pre-built HumanMessage (or raw str for backward compat)
            attachments: Optional attachments (for backward compat str path)
            returns: Pydantic BaseModel class or generic type (List/Dict)
            **kwargs: Additional LLM parameters

        Returns:
            Instance of the Pydantic model (or List/Dict of instances)
        """
        # message is already a HumanMessage built by the caller; the str
        # path is retained only for backward compatibility.
        if isinstance(message, str):
            user_message = await self._build_human_message(message, attachments)
            self.messages.append(user_message)
            if attachments:
                self.attachments.extend(attachments)

        # Apply context handler
        messages_to_send = self._apply_context_handler()

        return await self._run_extraction_loop(messages_to_send, returns, **kwargs)
    
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
        
        # Extract description from model docstring or use default.
        # Previously this branched on ``hasattr(model, '__pydantic_generic_metadata__')``
        # as a RootModel probe, but that attribute exists on every Pydantic v2
        # BaseModel subclass, so the check was tautological and every structured
        # output ended up advertised to the LLM as a generic ``ExtractedData``
        # tool. That erased useful semantic hints (e.g. ``Requirement``,
        # ``UserProfile``) from the tool name and description. We now dispatch
        # on ``issubclass(model, RootModel)`` instead, which is the actual
        # distinguishing property.
        if isinstance(model, type) and issubclass(model, RootModel):
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
    
    async def _asend_stream_impl(
        self,
        message: Union[str, Message],
        attachments: Optional[List[Attachment]],
        tool_executor: Optional[ToolExecutor],
        **kwargs
    ) -> AsyncIterator[Union[MessageChunk, ReasoningChunk]]:
        """
        Internal implementation of stream mode.
        
        This method handles the full workflow: convert message, apply context handler,
        then delegate to tool manager or provider streaming.
        
        Yields:
            Union[MessageChunk, ReasoningChunk]: Streaming chunks
        """
        from .message import MessageChunk, ReasoningChunk, AIMessage
        
        # message is already a HumanMessage built by the caller (asend)
        self.messages.append(message)
        
        # Track attachments at conversation level
        if attachments:
            self.attachments.extend(attachments)

        # Apply context handler
        messages_to_send = self._apply_context_handler()
        
        # Delegate to tool manager or provider streaming
        if self._tool_manager:
            # Temporarily override executor if specified
            original_executor = None
            if tool_executor is not None:
                original_executor = self._tool_manager.executor
                self._tool_manager.executor = self._get_executor(tool_executor)
            
            try:
                # Has tools: use tool manager's streaming loop.
                # ``history=self.messages`` makes intermediate messages surface
                # in conv.messages as they are produced (see execute_loop docs).
                async for chunk, _ in self._tool_manager.execute_loop_stream(
                    provider=self.provider,
                    messages=messages_to_send,
                    model_uri=self.model_uri,
                    round_context_fn=self._make_round_context_fn(),
                    history=self.messages,
                ):
                    yield chunk
            finally:
                # Restore original executor
                if original_executor is not None:
                    self._tool_manager.executor = original_executor
        else:
            # No tools: use provider streaming directly
            accumulated_content = ""
            accumulated_reasoning = ""
            
            async for chunk in self._asend_stream(messages_to_send, **kwargs):
                if isinstance(chunk, MessageChunk):
                    accumulated_content += chunk.content or ""
                elif isinstance(chunk, ReasoningChunk):
                    accumulated_reasoning += chunk.content or ""
                yield chunk
            
            # Save final message after streaming
            from .message import AIMessage
            final_msg = AIMessage(
                content=accumulated_content,
                reasoning_content=accumulated_reasoning if accumulated_reasoning else None
            )
            self.messages.append(final_msg)
    
    async def _asend_with_events_impl(
        self,
        message: Union[str, Message],
        attachments: Optional[List[Attachment]],
        tool_executor: Optional[ToolExecutor],
        **kwargs
    ) -> AsyncIterator['StreamEvent']:
        """
        Internal implementation of event stream mode.
        
        This method handles the full workflow: convert message, apply context handler,
        then delegate to tool manager's event stream.
        
        Yields:
            StreamEvent: MessageChunk, ReasoningChunk, ToolCallStartEvent, ToolCallSuccessEvent, or ToolCallErrorEvent
        """
        from .message import MessageChunk, ReasoningChunk, ConversationCompleteEvent, _current_turn_id
        
        # message is already a HumanMessage built by the caller (asend)
        self.messages.append(message)
        
        # Track attachments at conversation level
        if attachments:
            self.attachments.extend(attachments)

        # Apply context handler
        messages_to_send = self._apply_context_handler()
        
        # Delegate to tool manager's event stream
        if self._tool_manager:
            # Temporarily override executor if specified
            original_executor = None
            if tool_executor is not None:
                original_executor = self._tool_manager.executor
                self._tool_manager.executor = self._get_executor(tool_executor)
            
            try:
                # Has tools: return full event stream from tool manager.
                # ``history=self.messages`` lets intermediate messages appear
                # in conv.messages incrementally as the tool loop progresses.
                async for event in self._tool_manager.execute_loop_with_events(
                    provider=self.provider,
                    messages=messages_to_send,
                    model_uri=self.model_uri,
                    round_context_fn=self._make_round_context_fn(),
                    history=self.messages,
                ):
                    # Intercept ConversationCompleteEvent (internal use only)
                    if isinstance(event, ConversationCompleteEvent):
                        # Messages were already appended live via history; the
                        # event still fires as an end-of-turn marker for any
                        # internal consumers that care about it.
                        pass
                    else:
                        yield event
            finally:
                # Restore original executor
                if original_executor is not None:
                    self._tool_manager.executor = original_executor
        else:
            # No tools: only return content events (both MessageChunk and ReasoningChunk)
            async for chunk in self._asend_stream(messages_to_send, **kwargs):
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
    
    def _send_stream(self, messages: List[Message], **kwargs) -> Iterator[Union[MessageChunk, ReasoningChunk, FailoverChunk]]:
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
            if isinstance(provider_chunk, FailoverChunk):
                complete_content = ""
                complete_reasoning_content = ""
                last_chunk_was_final = False
                last_chunk_metadata = {}
                yield provider_chunk
                continue

            unified_chunk = self.provider.converter.from_provider_chunk(provider_chunk)
            
            # Always accumulate metadata regardless of content.
            # Anthropic sends prompt_tokens in message_start and completion_tokens
            # in message_delta — both arrive as content-less chunks, so we must
            # track metadata outside the content/reasoning guards.
            if unified_chunk.metadata:
                last_chunk_metadata = _merge_stream_metadata(last_chunk_metadata, unified_chunk.metadata)
            
            # Handle reasoning content
            if unified_chunk.reasoning_content:
                complete_reasoning_content += unified_chunk.reasoning_content
                yield ReasoningChunk(
                    content=unified_chunk.reasoning_content,
                    is_final=False,
                    metadata=unified_chunk.metadata
                )
            
            # Handle regular content
            if unified_chunk.content:
                complete_content += unified_chunk.content
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
            # Provider sent final chunk, save the message AND send final MessageChunk
            final_message = AIMessage(
                content=complete_content,
                reasoning_content=complete_reasoning_content if complete_reasoning_content else None,
                metadata=last_chunk_metadata
            )
            self.messages.append(final_message)
            
            # Send final MessageChunk with final_message so callers can capture it
            yield MessageChunk(
                content="",
                is_final=True,
                final_message=final_message
            )

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
    
    def _make_round_context_fn(self):
        """Build the round-scoped context callback passed to the tool loop.

        Returns a closure ``fn(current_messages, round_index) -> messages``
        that delegates to ``self.context_handler.call_for_round(...)``.
        The default :meth:`BaseContextHandler.handle_round` is a no-op, so
        handlers that don't opt in cost nothing beyond a deep-copy per
        round (kept for symmetry with turn-level compression).
        """
        handler = self.context_handler
        conv_id = self.id

        def _round_context_fn(current_messages: List[Message], round_index: int) -> List[Message]:
            return handler.call_for_round(
                current_messages,
                conversation_id=conv_id,
                round_index=round_index,
            )

        return _round_context_fn
    
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
            - model_uri: Full model URI (e.g. 'anthropic/claude-sonnet-4-6')
            - provider: Provider name (e.g. 'anthropic')
            - model: Model name extracted from URI
            - total_messages: Total number of messages
            - by_type: Message count by type
            - total_tokens: Total tokens used (prompt + completion, excludes cache reads)
            - input_tokens: Prompt tokens (non-cached portion)
            - output_tokens: Completion tokens
            - cache_creation_tokens: Tokens written to prompt cache (billed at ~1.25x)
            - cache_read_tokens: Tokens read from prompt cache (billed at ~0.1x)
        
        Example:
            >>> conv.stats()
            {
                'model_uri': 'anthropic/claude-sonnet-4-6',
                'provider': 'anthropic',
                'model': 'claude-sonnet-4-6',
                'total_messages': 10,
                'by_type': {'user': 5, 'assistant': 4, 'tool': 1},
                'total_tokens': 1081,
                'input_tokens': 764,
                'output_tokens': 317,
                'cache_creation_tokens': 2542,
                'cache_read_tokens': 7626
            }
        """
        stats = {
            'model_uri': self.model_uri,
            'provider': self._provider_name,
            'model': self._model_name,
            'total_messages': len(self.messages),
            'by_type': {},
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'cache_creation_tokens': 0,
            'cache_read_tokens': 0,
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
                stats['cache_creation_tokens'] += usage.cache_creation_input_tokens
                stats['cache_read_tokens'] += usage.cache_read_input_tokens
        
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

    def dump(self) -> List[dict]:
        """Serialize conversation messages to a JSON-serializable list of dicts."""
        return [m.model_dump(mode="json") for m in self.messages]

    def load(self, data: List[dict]) -> None:
        """Restore conversation messages from a serialized list of dicts."""
        _ROLE_MAP = {
            "user": HumanMessage,
            "assistant": AIMessage,
            "system": SystemMessage,
            "tool": ToolMessage,
        }
        messages = []
        for m in data:
            role = m.get("role")
            cls = _ROLE_MAP.get(role)
            if cls is None:
                raise ValueError(f"Unknown message role '{role}'")
            messages.append(cls.model_validate(m))
        self.messages = messages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
