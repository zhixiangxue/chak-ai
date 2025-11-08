# chak/context/strategies/summarize.py
"""Summarization context strategy that compresses history via LLM summarization."""

from typing import List, Optional, Callable
from .base import BaseContextStrategy, StrategyRequest, StrategyResponse
from ...message import Message, MarkerMessage, SystemMessage, HumanMessage, AIMessage
from ...exceptions import ContextError
from ...utils.uri import parse as parse_uri
from ...providers import create_provider
from ...providers.types import ProviderCategory


class SummarizationStrategy(BaseContextStrategy):
    """
    SummarizationStrategy
    
    Purpose:
    - Compress older conversation context via an LLM-generated summary when the
      current input tokens approach a threshold of the model's max_input_tokens.
    - Preserve recent turns verbatim for immediate context freshness.
    
    Semantics:
    - StrategyResponse.messages: returns the original messages plus a trailing
      summary MarkerMessage (role="context") that documents the summarization.
    - StrategyResponse.messages_to_send: computed view for the current round:
      SystemMessage(s) + the summary MarkerMessage + recent turns.
      The MarkerMessage will be converted to a SystemMessage before sending to the LLM
      by the conversation layer for compatibility, per project convention.
    
    Triggering:
    - Summarization triggers when token usage of [SystemMessage(s) + conversation messages]
      exceeds summarize_threshold * max_input_tokens.
    - summarize_threshold defaults to 0.75 (75%).
    
    Turn Definition:
    - A "turn" is the sequence from a HumanMessage up to (but not including)
      the next HumanMessage.
    - keep_recent_turns specifies how many of the most recent turns are preserved
      unchanged in messages_to_send.
    
    Token Control:
    - This strategy assumes that the summary reduces token pressure. If final tokens
      still exceed max_input_tokens, the caller should provide a smaller keep_recent_turns
      or adjust summarize_threshold.
    
    Parameters:
    - max_input_tokens (int): model context window upper bound.
    - summarize_threshold (float): fraction of max_input_tokens to trigger summarization.
    - keep_recent_turns (int): number of recent turns to keep verbatim.
    - token_counter (Optional[Callable[[str], int]]): custom token counter.
    - summarizer_model_uri (str): OpenAI-compatible model URI for summarization.
    - summarizer_api_key (str): API key for the summarizer model.
    
    Notes:
    - System messages are always included in messages_to_send.
    - The summary MarkerMessage records metadata: {type: "summary", summarized_count, summary}.
    - The strategy does not remove original messages; it appends a summary marker to preserve
      a complete processing record in the conversation.
    """
    
    def __init__(
        self,
        max_input_tokens: int,
        summarize_threshold: float = 0.75,
        keep_recent_turns: int = 2,
        token_counter: Optional[Callable[[str], int]] = None,
        summarizer_model_uri: str = "",
        summarizer_api_key: str = ""
    ):
        """
        Initialize SummarizationStrategy.
        
        Args:
            max_input_tokens: Maximum input token count (context window limit)
            summarize_threshold: Trigger summarization at this % of max_input_tokens
                                (default: 0.75 = 75%)
            keep_recent_turns: Number of recent conversation turns to preserve
                              (default: 2, i.e., keep last 2 user-assistant exchanges)
            token_counter: Custom token counting function
            summarizer_model_uri: OpenAI-compatible model URI for summarization (required)
            summarizer_api_key: API key for the summarizer model (required)
            
        Note:
            - A "turn" = 1 user message + 1 assistant response
            - Recent turns are always preserved for immediate context
            - Older messages are summarized when threshold is exceeded
            
        Raises:
            ValueError: If max_input_tokens is not positive or summarizer_model_uri/api_key is empty
        """
        super().__init__(token_counter=token_counter)
        
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if not summarizer_model_uri:
            raise ValueError("summarizer_model_uri is required")
        if not summarizer_api_key:
            raise ValueError("summarizer_api_key is required")
        
        self.max_input_tokens = max_input_tokens
        self.summarize_threshold = summarize_threshold
        self.keep_recent_turns = keep_recent_turns
        self.summarizer_model_uri = summarizer_model_uri
        self.summarizer_api_key = summarizer_api_key
        
        # Calculate trigger point
        self.trigger_tokens = int(max_input_tokens * summarize_threshold)
    
    def process(self, request: StrategyRequest) -> StrategyResponse:
        """
        Process messages according to Summarization strategy.
        
        Args:
            request: Strategy request containing messages
            
        Returns:
            Strategy response with processed messages
        """
        messages = request.messages
        
        if not messages:
            return StrategyResponse(messages=[], messages_to_send=[])
        
        # 1. Separate by type
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        marker_messages = [m for m in messages if isinstance(m, MarkerMessage)]
        conversation_messages = [
            m for m in messages 
            if not isinstance(m, (SystemMessage, MarkerMessage))
        ]
        
        # 2. Find last summary marker
        last_summary_idx = self._find_last_summary_marker(messages)
        
        # 3. Determine messages to analyze
        to_analyze = conversation_messages
        
        # 4. Calculate current token usage
        combined: List[Message] = list(system_messages) + list(to_analyze)  # type: ignore[assignment]
        total_tokens = self.count_messages_tokens(combined)
        
        # 5. Check if summarization needed
        if total_tokens <= self.trigger_tokens:
            # No summarization needed
            return StrategyResponse(
                messages=messages,
                messages_to_send=list(system_messages) + list(to_analyze)  # type: ignore[arg-type]
            )
        
        # 6. Perform summarization
        to_analyze_list: List[Message] = list(to_analyze)
        recent, to_summarize = self._split_recent_and_old(to_analyze_list)
        
        if not to_summarize:
            # Nothing to summarize (all messages are recent)
            return StrategyResponse(
                messages=messages,
                messages_to_send=list(system_messages) + list(to_analyze)  # type: ignore[arg-type]
            )
        
        # 7. Generate summary
        summary_text = self._llm_summarize(to_summarize)
        
        # 8. Create summary marker
        marker = MarkerMessage(
            content=f"[对话摘要] {summary_text}",
            metadata={
                "type": "summary",
                "summarized_count": len(to_summarize),
                "summary": summary_text
            }
        )
        
        # 9. Build response
        full_messages = messages + [marker]
        to_send_list: List[Message] = list(system_messages) + [marker] + recent  # type: ignore[assignment]
        
        return StrategyResponse(
            messages=full_messages,
            messages_to_send=to_send_list  # type: ignore[arg-type]
        )
    
    def _find_last_summary_marker(self, messages: List[Message]) -> Optional[int]:
        """找到最后一个摘要标记的索引。"""
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, MarkerMessage):
                if msg.metadata.get("type") == "summary":
                    return i
        return None
    
    def _split_recent_and_old(
        self, 
        messages: List[Message]
    ) -> tuple[List[Message], List[Message]]:
        """
        Split messages into recent turns and old messages to summarize.
        
        Returns:
            (recent_messages, old_messages)
        """
        if not messages:
            return [], []
        
        # Calculate how many messages to keep (turns = pairs)
        messages_to_keep = self.keep_recent_turns * 2
        
        if len(messages) <= messages_to_keep:
            # All messages are "recent"
            return messages, []
        
        # Split: old (to summarize) and recent (to keep)
        split_point = len(messages) - messages_to_keep
        old_messages = messages[:split_point]
        recent_messages = messages[split_point:]
        
        return recent_messages, old_messages
    
    def _llm_summarize(self, messages: List[Message]) -> str:
        """
        Summarize using the configured model via Chak's provider system.
        
        Args:
            messages: Messages to summarize
            
        Returns:
            Summary text generated by the LLM
        """
        # Build prompt from older messages
        segments: List[str] = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = "User"
            elif isinstance(m, AIMessage):
                role = "Assistant"
            else:
                role = "Message"
            text = (m.content or "").strip()
            if text:
                segments.append(f"{role}: {text}")
        
        prompt = "\n".join(segments)
        if not prompt:
            raise ContextError("No valid content to summarize")
        
        # Parse URI to get provider config
        parsed = parse_uri(self.summarizer_model_uri)
        
        # Build config dict
        config_dict = {
            'api_key': self.summarizer_api_key,
            'model': parsed['model']
        }
        if parsed['base_url']:
            config_dict['base_url'] = parsed['base_url']
        
        # Create provider using Chak's provider system
        provider = create_provider(
            parsed['provider'],
            config_dict,
            category=ProviderCategory.LLM
        )
        
        # Build messages for summarization
        system_inst = (
            "You are a concise conversation summarizer. Produce a short, "
            "human-readable summary in Chinese, highlighting key intents, "
            "decisions, and data."
        )
        summarize_messages = [
            SystemMessage(content=system_inst),
            HumanMessage(content=prompt)
        ]
        
        # Call provider to generate summary
        response = provider.send(summarize_messages, stream=False, temperature=0.2, max_tokens=256)
        
        # Extract content
        if isinstance(response, AIMessage):
            if response.content:
                return response.content.strip()
            raise ContextError("Summarizer model returned empty response")
        else:
            raise ContextError(f"Unexpected response type: {type(response)}")
