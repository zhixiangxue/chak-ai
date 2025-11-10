# chak/context/strategies/noop.py
"""Noop (No Operation) context strategy that passes through all messages."""

from .base import BaseContextStrategy, StrategyRequest, StrategyResponse


class NoopStrategy(BaseContextStrategy):
    """
    NoopStrategy (No Operation)
    
    Purpose:
    - Provide a pass-through context strategy that performs no filtering
      or transformation of the conversation messages.
    
    Semantics:
    - StrategyResponse.messages: returns the original messages unchanged.
    
    Notes:
    - Intended for debugging, baseline comparison, or scenarios where
      the caller wants full history sent to the LLM.
    - SystemMessage(s), MarkerMessage(s), and any other message types are
      passed through as-is.
    """
    
    def __init__(self):
        """Initialize the noop strategy."""
        super().__init__()
    
    def process(self, request: StrategyRequest) -> StrategyResponse:
        """
        Return all messages without any processing.
        
        Args:
            request: Strategy request containing messages
            
        Returns:
            Strategy response with all messages unchanged
        """
        return StrategyResponse(messages=request.messages)
