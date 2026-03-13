# src/chak/__init__.py
"""
chak: A simple, yet elegant, LLM API routing library.

Supports two URI formats:
1. Simple: provider/model (e.g., 'deepseek/deepseek-chat')
2. Full: provider@base_url:model (e.g., 'deepseek@https://api.deepseek.com:deepseek-chat')

Example:
>>> import chak
>>> # Simple format (recommended)
>>> conv = chak.Conversation('deepseek/deepseek-chat', api_key='xxx')
>>> response = conv.send('Hello!')
>>> 
>>> # Full format with custom base_url
>>> conv = chak.Conversation('deepseek@https://custom.api.com:deepseek-chat', api_key='xxx')
"""

__version__ = "0.3.0"

# Context handlers
from .context.handlers import (
    BaseContextHandler,
    NoopContextHandler,
    FIFOContextHandler,
    LRUContextHandler,
    SummarizationContextHandler
)
# Core API - import the main classes
from .conversation import Conversation, ToolExecutor
from .exceptions import (
    ChakError, ProviderError, ConfigError,
    ConversationNotFoundError, ContextError
)
from .message import (
    Message, MessageChunk, ReasoningChunk,
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage,
    ToolCallStartEvent, ToolCallSuccessEvent, ToolCallErrorEvent
)
# Attachment classes for multimodal support
from .attachment import Image, Audio, Video, MimeType, PDF, DOC, Excel, TXT, Link
# Types
from .schemas import Reasoning, Cache
# Utility functions
from .utils.uri import build, build_simple, parse
from .utils.logger import set_log_level

# MCP integration (optional)
try:
    from .tools.mcp import Server
    _mcp_available = True
except ImportError:
    _mcp_available = False
    
    # Provide helpful error message when MCP dependencies are missing
    class Server:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MCP dependencies not installed. Install with:\n"
                "  pip install mcp\n"
                "or:\n"
                "  pip install chakpy[mcp]"
            )

# Server (optional dependency)
try:
    from .server import serve
    _server_available = True
except ImportError:
    _server_available = False
    
    # Provide helpful error message when server dependencies are missing
    def serve(*args, **kwargs):
        raise ImportError(
            "Server dependencies not installed. Install with:\n"
            "  pip install chakpy[server]\n"
            "or:\n"
            "  pip install chakpy[all]"
        )


# Export the public API
__all__ = [
    # Core classes
    'Conversation',
    'ToolExecutor',
    'Message',
    'MessageChunk',
    'ReasoningChunk',
    'BaseMessage',
    'HumanMessage',
    'AIMessage',
    'SystemMessage',
    'ToolMessage',
    'ToolCallStartEvent',
    'ToolCallSuccessEvent',
    'ToolCallErrorEvent',
    
    # Multimodal attachments
    'Image',
    'Audio',
    'Video',
    'MimeType',
    'PDF',
    'DOC',
    'Excel',
    'TXT',
    'Link',
    
    # Types
    'Reasoning',
    'Cache',
    
    # Exceptions
    'ChakError',
    'ProviderError', 
    'ConfigError',
    'ConversationNotFoundError',
    'ContextError',
    
    # Context handlers
    'BaseContextHandler',
    'NoopContextHandler',
    'FIFOContextHandler',
    'LRUContextHandler',
    'SummarizationContextHandler',
    
    # Utilities
    'build',
    'build_simple',
    'parse',
    'set_log_level',
    
    # Server
    'serve',
    
    # Tools (MCP)
    'Server',

    # Version
    '__version__',
]