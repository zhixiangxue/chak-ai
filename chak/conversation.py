from typing import List, Dict, Any, Iterator, Union, Literal, Optional
from .utils.uri import parse as parse_uri
from .providers import create_provider
from .providers.types import ProviderCategory
from .message import Message, MessageChunk, HumanMessage, AIMessage, SystemMessage, ToolMessage, MarkerMessage
from .context.strategies import BaseContextStrategy, NoopStrategy
from .context.strategies.base import StrategyRequest


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
        context_strategy: Optional[BaseContextStrategy] = None,
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
            context_strategy: Context management strategy (default: NoopStrategy)
            **kwargs: Additional configuration parameters
        """
        self.model_uri = model_uri
        self.api_key = api_key
        self.messages = []
        
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
        批量添加消息到对话历史，用于恢复历史对话。
        
        Args:
            messages: 消息列表，可以是Message对象列表或字典列表
                     字典格式: {"role": "user", "content": "hello"}
        
        Example:
            >>> conv = Conversation(...)
            >>> # 恢复历史对话
            >>> conv.add_messages([
            ...     {"role": "user", "content": "你好"},
            ...     {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
            ...     {"role": "user", "content": "介绍一下你自己"}
            ... ])
        """
        for msg in messages:
            if isinstance(msg, dict):
                # 如果是字典，转换为Message对象
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
                    raise ValueError(f"无效的角色: {role}")
            elif isinstance(msg, (HumanMessage, AIMessage, SystemMessage, ToolMessage, MarkerMessage)):
                # 如果已经是Message对象，直接添加
                self.messages.append(msg)
            else:
                raise TypeError(f"消息必须是Message对象或字典，收到: {type(msg)}")

    def send(
            self,
            message: str,
            role: Literal["user", "assistant", "system", "tool"] = "user",
            stream: bool = False,
            **kwargs
    ) -> Union[Message, Iterator[MessageChunk]]:
        """Send message to the provider."""
        # Create user message based on role
        if role == "user":
            user_message = HumanMessage(content=message)
        elif role == "assistant":
            user_message = AIMessage(content=message)
        elif role == "system":
            user_message = SystemMessage(content=message)
        elif role == "tool":
            user_message = ToolMessage(content=message)
        else:
            user_message = HumanMessage(content=message)
        
        self.messages.append(user_message)

        # Apply context strategy
        messages_to_send = self._apply_context_strategy()

        # Send to provider (model is already in provider config)
        if stream:
            return self._send_stream(messages_to_send, **kwargs)
        else:
            response = self.provider.send(
                messages=messages_to_send,
                stream=False,
                **kwargs
            )
            # Response is Message (not Iterator)
            if not isinstance(response, (HumanMessage, AIMessage, SystemMessage, ToolMessage, MarkerMessage)):
                # Old Message type, convert to AIMessage
                ai_response = AIMessage(
                    content=response.content,  # type: ignore
                    reasoning_content=response.reasoning_content,  # type: ignore
                    tool_calls=response.tool_calls,  # type: ignore
                    refusal=response.refusal  # type: ignore
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
        for provider_chunk in provider_chunks:
            chunk = self.provider.converter.from_provider_chunk(provider_chunk)
            complete_content += chunk.content
            yield chunk

        # Create complete message from chunks and send final chunk
        if complete_content:
            final_message = AIMessage(
                content=complete_content
            )
            self.messages.append(final_message)
            
            # 发送一个特殊的 final chunk，包含完整消息
            yield MessageChunk(
                content="",
                is_final=True,
                final_message=final_message
            )

    def _apply_context_strategy(self) -> List[Message]:
        """
        Apply context strategy to process messages.
        
        Returns:
            Processed message list according to the strategy
        """
        if not self.messages:
            return []
        
        # Build strategy request
        request = StrategyRequest(messages=self.messages)
        
        # Get strategy response
        response = self.context_strategy.process(request)
        
        # Update messages (may include markers)
        self.messages = response.messages
        
        # Prepare messages for LLM (convert context role to system)
        messages_for_llm = self._prepare_for_llm(response.messages_to_send)
        
        return messages_for_llm
    
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

    def close(self):
        """Close the provider."""
        if hasattr(self, 'provider'):
            self.provider.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
