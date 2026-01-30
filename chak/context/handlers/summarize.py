# chak/context/handlers/summarize.py
"""Summarization context handler."""

from typing import List

from .base import BaseContextHandler
from ...message import Message, SystemMessage, HumanMessage
from ...providers import create_provider
from ...providers.types import ProviderCategory
from ...utils.uri import parse as parse_uri


class SummarizationContextHandler(BaseContextHandler):
    """
    SummarizationContextHandler - Compress old messages via LLM summarization.
    
    Behavior:
    - When history is too long, summarize old messages into a SystemMessage
    - Returns: [summary message] + [recent N messages]
    - Does NOT modify history_messages
    
    Parameters:
    - max_messages: Maximum number of messages in context (trigger threshold)
    - keep_recent: Number of recent messages to keep unsummarized
    - summarizer_model_uri: Model URI for summarization
    - summarizer_api_key: API key for summarization model
    """
    
    def __init__(
        self,
        max_messages: int,
        keep_recent: int = 5,
        summarizer_model_uri: str = "",
        summarizer_api_key: str = ""
    ):
        """
        Initialize summarization handler.
        
        Args:
            max_messages: Trigger summarization when history exceeds this count
            keep_recent: Number of recent messages to keep in full detail
            summarizer_model_uri: Model URI for generating summaries
            summarizer_api_key: API key for summarizer
        """
        super().__init__()
        if max_messages < keep_recent:
            raise ValueError("max_messages must be >= keep_recent")
        if not summarizer_model_uri:
            raise ValueError("summarizer_model_uri is required")
        if not summarizer_api_key:
            raise ValueError("summarizer_api_key is required")
        
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.summarizer_model_uri = summarizer_model_uri
        self.summarizer_api_key = summarizer_api_key
        
        # Initialize summarizer provider
        parsed = parse_uri(summarizer_model_uri)
        config = {
            'api_key': summarizer_api_key,
            'model': parsed['model']
        }
        if parsed['base_url']:
            config['base_url'] = parsed['base_url']
        config.update(parsed['params'])
        
        self.summarizer = create_provider(
            parsed['provider'],
            config,
            category=ProviderCategory.LLM
        )
    
    def handle(
        self,
        messages: List[Message],
        *,
        conversation_id: str,
    ) -> List[Message]:
        """
        Return summarized context if history is too long.
        
        Args:
            messages: Complete conversation history
            conversation_id: Unique ID for this conversation
            
        Returns:
            Context messages (summary + recent messages if needed)
        """
        if not messages:
            return []
        
        # Separate system messages and conversation messages
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        conversation_messages = [m for m in messages if not isinstance(m, SystemMessage)]
        
        # Check if summarization needed
        if len(conversation_messages) <= self.max_messages:
            # No summarization needed, return all
            return messages
        
        # Need summarization
        # Split: messages to summarize + messages to keep
        to_summarize = conversation_messages[:-self.keep_recent]
        to_keep = conversation_messages[-self.keep_recent:]
        
        # Generate summary
        summary_text = self._generate_summary(to_summarize)
        summary_message = SystemMessage(content=f"[Previous conversation summary]\n{summary_text}")
        
        # Return: system messages + summary + recent messages
        return system_messages + [summary_message] + to_keep
    
    def _generate_summary(self, messages: List[Message]) -> str:
        """
        Generate summary for given messages using LLM.
        
        Args:
            messages: Messages to summarize
            
        Returns:
            Summary text
        """
        # Build prompt
        conversation_text = self._format_messages_for_summary(messages)
        prompt = f"""Please provide a CONCISE summary of the following conversation. Focus on key topics, decisions, and important information.

Conversation:
{conversation_text}

Summary:"""
        
        # Call summarizer
        summary_msg = SystemMessage(content=prompt)
        response = self.summarizer.send(messages=[summary_msg], stream=False)
        
        return response.content or "(Summary generation failed)"
    
    def _format_messages_for_summary(self, messages: List[Message]) -> str:
        """
        Format messages into readable text for summarization.
        
        Args:
            messages: Messages to format
            
        Returns:
            Formatted conversation text
        """
        lines = []
        for msg in messages:
            role = msg.role
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
