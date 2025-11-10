from typing import List, Dict, Any, Iterator, Union, Literal, Optional

from .context.strategies import BaseContextStrategy, NoopStrategy
from .context.strategies.base import StrategyRequest
from .message import Message, MessageChunk, HumanMessage, AIMessage, SystemMessage, ToolMessage, MarkerMessage
from .providers import create_provider
from .providers.types import ProviderCategory
from .utils.uri import parse as parse_uri


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
        last_chunk_was_final = False
        
        for provider_chunk in provider_chunks:
            chunk = self.provider.converter.from_provider_chunk(provider_chunk)
            complete_content += chunk.content
            
            # Check if this chunk is already marked as final
            if chunk.is_final:
                last_chunk_was_final = True
            
            yield chunk

        # Only send additional final chunk if provider didn't send one
        if complete_content and not last_chunk_was_final:
            final_message = AIMessage(
                content=complete_content
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
                content=complete_content
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
                if isinstance(usage, dict):
                    stats['total_tokens'] += usage.get('total_tokens', 0)
                    stats['input_tokens'] += usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
                    stats['output_tokens'] += usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
        
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

    def close(self):
        """Close the provider."""
        if hasattr(self, 'provider'):
            self.provider.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
