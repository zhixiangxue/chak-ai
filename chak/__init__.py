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

__version__ = "0.1.2"

# Strategy classes
from .context.strategies import FIFOStrategy, NoopStrategy, SummarizationStrategy, BaseContextStrategy
from .context.strategies.base import StrategyRequest, StrategyResponse
# Core API - import the main classes
from .conversation import Conversation
from .exceptions import (
    ChakError, ProviderError, ConfigError,
    ConversationNotFoundError, ContextError
)
from .message import (
    Message, MessageChunk,
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage, MarkerMessage
)
# Utility functions
from .utils.uri import build, build_simple, parse

# Server (optional dependency)
try:
    from .server import serve
    _server_available = True
except ImportError:
    _server_available = False
    serve = None  # type: ignore


# Export the public API
__all__ = [
    # Core classes
    'Conversation',
    'Message',
    'MessageChunk',
    'BaseMessage',
    'HumanMessage',
    'AIMessage',
    'SystemMessage',
    'ToolMessage',
    'MarkerMessage',
    
    # Exceptions
    'ChakError',
    'ProviderError', 
    'ConfigError',
    'ConversationNotFoundError',
    'ContextError',
    
    # Strategies
    'FIFOStrategy',
    'NoopStrategy',
    'SummarizationStrategy',
    'NoopStrategy',
    'BaseContextStrategy',
    'StrategyRequest',
    'StrategyResponse',
    
    # Utilities
    'build',
    'build_simple',
    'parse',
    
    # Server
    'serve',

    # Version
    '__version__',
]