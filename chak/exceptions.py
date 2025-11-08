# src/chak/exceptions.py
"""
Exception hierarchy for chak.
"""

class ChakError(Exception):
    """Base exception for all chak errors."""
    pass

class ProviderError(ChakError):
    """Errors related to LLM providers."""
    pass

class ConfigError(ChakError):
    """Configuration-related errors."""
    pass

class ConversationNotFoundError(ChakError):
    """Requested conversation not found."""
    pass

class ContextError(ChakError):
    """Context management errors."""
    pass

class URIError(ChakError):
    """URI parsing and validation errors."""
    pass