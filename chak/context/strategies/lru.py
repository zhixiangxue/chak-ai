# chak/context/strategies/lru.py
"""LRU (Least Recently Used) context strategy - Auto-prune cold topics."""

from typing import List, Optional, Callable

from .base import BaseContextStrategy, StrategyRequest, StrategyResponse
from .summarize import SummarizationStrategy
from ...exceptions import ContextError
from ...message import Message, MarkerMessage, HumanMessage, AIMessage, SystemMessage
from ...providers import create_provider
from ...providers.types import ProviderCategory
from ...utils.logger import logger
from ...utils.uri import parse as parse_uri


class LRUStrategy(BaseContextStrategy):
    """
    LRU (Least Recently Used) Strategy - Auto-prune cold topics
    
    Inspired by the classic LRU cache eviction algorithm, this strategy
    automatically prunes (forgets) topics that are least recently used.
    
    Design Philosophy:
    - Wrapper pattern: Wraps SummarizationStrategy internally
    - Transparent to developers: Same parameters as SummarizationStrategy
    - Automatic topic pruning: Analyzes recent markers, regenerates last marker
    
    Difference from SummarizationStrategy:
    - SummarizationStrategy: Summarizes ALL content (all topics)
    - LRUStrategy: Only keeps HOT topics, prunes COLD topics
    
    How it works:
    1. Internally calls SummarizationStrategy to generate summary markers
    2. Analyzes recent 5 markers to extract hot topics
    3. Regenerates the last marker, keeping only hot topic content
    4. Cold topics are automatically pruned (forgotten)
    
    Parameters (identical to SummarizationStrategy):
    - max_input_tokens (int): Maximum input token count
    - summarize_threshold (float): Threshold ratio for triggering (default 0.75)
    - prefer_recent_turns (int): Preserve recent turns (default 2)
    - token_counter (Optional[Callable[[str], int]]): Custom token counter
    - summarizer_model_uri (str): Summarizer model URI
    - summarizer_api_key (str): Summarizer model API key
    
    Internal Implementation (transparent to developers):
    - Automatically analyzes recent 5 markers to extract hot topics
    - Regenerates last marker with only hot topic content
    
    Example Usage:
    ```python
    # Identical to SummarizationStrategy usage
    strategy = LRUStrategy(
        max_input_tokens=128000,
        summarize_threshold=0.75,
        prefer_recent_turns=2,
        summarizer_model_uri="openai://gpt-4o-mini",
        summarizer_api_key="sk-xxx"
    )
    
    conversation = Conversation(
        provider_uri="openai://gpt-4o",
        api_key="sk-xxx",
        context_strategies=[strategy]
    )
    ```
    
    Note:
    - Zero learning curve: Same parameters as SummarizationStrategy
    - Easy to switch: Just change class name from SummarizationStrategy to LRUStrategy
    - Automatic pruning: No need to understand markers or internal logic
    """
    
    # Internal constant (not exposed to developers)
    _DEFAULT_RECENT_MARKERS_COUNT = 5  # Analyze recent 5 markers
    
    def __init__(
        self,
        max_input_tokens: int,
        summarize_threshold: float = 0.75,
        prefer_recent_turns: int = 2,
        token_counter: Optional[Callable[[str], int]] = None,
        summarizer_model_uri: str = "",
        summarizer_api_key: str = ""
    ):
        """Initialize LRU Strategy.
        
        Args:
            max_input_tokens: Maximum input token count (context window limit)
            summarize_threshold: Threshold ratio for triggering (default 0.75 = 75%)
            prefer_recent_turns: Preserve recent turns (default 2)
            token_counter: Custom token counting function
            summarizer_model_uri: Summarizer model URI (required)
            summarizer_api_key: Summarizer model API key (required)
            
        Note:
            - Parameters identical to SummarizationStrategy
            - Internally analyzes recent 5 markers to extract hot topics
        """
        super().__init__(token_counter=token_counter)
        
        # Save all parameters
        self.max_input_tokens = max_input_tokens
        self.summarize_threshold = summarize_threshold
        self.prefer_recent_turns = prefer_recent_turns
        self.summarizer_model_uri = summarizer_model_uri
        self.summarizer_api_key = summarizer_api_key
        
        # Internally create SummarizationStrategy instance (transparent to developers)
        self._summarization_strategy = SummarizationStrategy(
            max_input_tokens=max_input_tokens,
            summarize_threshold=summarize_threshold,
            prefer_recent_turns=prefer_recent_turns,
            token_counter=token_counter,
            summarizer_model_uri=summarizer_model_uri,
            summarizer_api_key=summarizer_api_key
        )
    
    def process(self, request: StrategyRequest) -> StrategyResponse:
        """
        Process messages with LRU topic pruning.
        
        Workflow:
        1. Call internal summarization_strategy.process() → get new messages (with new marker)
        2. Extract all summary markers
        3. If marker count <= _DEFAULT_RECENT_MARKERS_COUNT, return directly (no pruning needed)
        4. Analyze recent N markers to extract hot topics
        5. Find original messages corresponding to last marker
        6. Regenerate last marker (only keep hot topics)
        7. Replace last marker and return
        
        Args:
            request: Strategy request containing message list
            
        Returns:
            Strategy response with processed messages (cold topics pruned)
        """
        
        # Step 1: Let summarization do its job
        result = self._summarization_strategy.process(request)
        messages = result.messages
        
        # Step 2: Extract all summary markers
        marker_indices = []
        for i, msg in enumerate(messages):
            if isinstance(msg, MarkerMessage) and msg.metadata.get("type") == "summary":
                marker_indices.append(i)
        
        # Step 3: Check if topic pruning is needed
        if len(marker_indices) <= self._DEFAULT_RECENT_MARKERS_COUNT:
            logger.debug(f"[LRU] Skip: Only {len(marker_indices)} markers, no pruning needed")
            return result
        
        logger.debug(f"\n🗑️  [LRU] Starting topic pruning")
        logger.debug(f"   Total markers: {len(marker_indices)}")
        logger.debug(f"   Analyzing recent: {self._DEFAULT_RECENT_MARKERS_COUNT}")
        
        # Step 4: Extract recent N markers (for context)
        recent_marker_indices = marker_indices[-self._DEFAULT_RECENT_MARKERS_COUNT:]
        recent_markers = [messages[i] for i in recent_marker_indices]
        
        logger.debug(f"   Recent markers: {len(recent_markers)}")
        
        # Step 5: Find original messages corresponding to last marker
        last_marker_idx = marker_indices[-1]
        
        # Find position of second-to-last marker (if exists)
        if len(marker_indices) >= 2:
            prev_marker_idx = marker_indices[-2]
            summarize_start = prev_marker_idx
        else:
            summarize_start = 0
        
        # Messages to summarize: from summarize_start to last_marker_idx (exclusive)
        to_summarize = messages[summarize_start:last_marker_idx]
        
        logger.debug(f"   Re-summarizing message range: [{summarize_start}:{last_marker_idx}]")
        
        # Step 6: Regenerate last marker (only keep hot topics)
        new_summary = self._regenerate_summary_with_hot_topics(
            to_summarize, 
            recent_markers
        )
        
        logger.debug(f"   ✅ Hot-topic-focused summary generated")
        
        # Step 7: Create new LRU marker (insert after original marker, not replace)
        lru_marker = MarkerMessage(
            content=f"[LRU Pruned Summary] {new_summary}",
            metadata={
                "type": "lru",  # Different type to distinguish from regular summary
                "summarized_count": len(to_summarize),
                "summary": new_summary,
                "pruned_from_marker": last_marker_idx  # Track which marker was pruned
            }
        )
        
        # Step 8: Insert LRU marker AFTER the last summary marker (keep original)
        # This makes it easy to see LRU is working in tests
        new_messages = (
            messages[:last_marker_idx + 1] +  # Keep original summary marker
            [lru_marker] +                      # Add LRU marker
            messages[last_marker_idx + 1:]     # Keep subsequent messages
        )
        
        logger.debug(f"   Marker replacement complete\n")
        
        return StrategyResponse(messages=new_messages)
    
    def _regenerate_summary_with_hot_topics(
        self, 
        messages: List[Message], 
        recent_markers: List[MarkerMessage]
    ) -> str:
        """
        Regenerate summary, keeping only hot topic related content.
        
        Simplified approach:
        - Provide LLM with recent N marker summaries as context
        - Let LLM decide which topics are hot based on recent context
        - Automatically prune cold/unrelated topics
        
        Args:
            messages: Message list to summarize
            recent_markers: Recent N markers (for context)
            
        Returns:
            Summary text (only hot topics)
        """
        # Build content (same as SummarizationStrategy._llm_summarize)
        segments: List[str] = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = "User"
                text = (m.content or "").strip()
            elif isinstance(m, AIMessage):
                role = "Assistant"
                text = (m.content or "").strip()
            elif isinstance(m, MarkerMessage):
                role = "Previous Summary"
                text = m.metadata.get('summary', '').strip()
            else:
                role = "Message"
                text = (m.content or "").strip()
            
            if text:
                segments.append(f"{role}: {text}")
        
        prompt = "\n".join(segments)
        if not prompt:
            raise ContextError("No valid content to summarize")
        
        # Extract recent marker summaries for context
        recent_summaries = [
            marker.metadata.get('summary', '')
            for marker in recent_markers
            if marker.metadata.get('summary')
        ]
        recent_context = "\n\n---\n\n".join(recent_summaries) if recent_summaries else "No recent context"
        
        # Build topic-focused system prompt (in English)
        system_inst = (
            "You are a conversation summarizer. Your task is to create a CUMULATIVE summary, "
            "but ONLY keep content related to RECENT HOT TOPICS.\n\n"
            "## Recent Context (Last N Summaries)\n"
            f"{recent_context}\n\n"
            "## Output Structure (MANDATORY)\n\n"
            "[Summary]\n"
            "Each round should have:\n"
            "  - Topic: <what this round discussed>\n"
            "  - User Intent: <what user wanted in this round>\n"
            "  - Summary: <CONCISE summary - 3-5 bullet points max>\n\n"
            "---\n\n"
            "## CRITICAL RULES\n\n"
            "### Rule 1: Only Keep Hot-Topic-Related Rounds\n"
            "- Compare each round with RECENT CONTEXT above\n"
            "- If a round is UNRELATED to recent topics → **SKIP IT COMPLETELY**\n"
            "- Only summarize rounds related to recent hot topics\n"
            "- This keeps conversation focused, avoids context pollution\n\n"
            "### Rule 2: Handle Previous Summaries\n"
            "- If a round in 'Previous Summary' is UNRELATED to recent topics → Skip\n"
            "- If a round in 'Previous Summary' is RELATED to recent topics → Keep and refine\n\n"
            "### Rule 3: New Rounds Must Be CONCISE\n"
            "- Each bullet point: 1-2 sentences maximum\n"
            "- Remove examples, detailed explanations, tables, formulas\n"
            "- Focus on: core concepts, main conclusions, key differences\n\n"
            "---\n\n"
            "## Example\n\n"
            "Suppose recent topics are 'Machine Learning, Deep Learning', history has Python, Java, ML rounds:\n\n"
            "Wrong Output (includes unrelated topics):\n"
            "[Summary]\n"
            "ROUND 1: Python basics...\n"
            "ROUND 2: Java basics...\n"
            "ROUND 3: Machine Learning...\n\n"
            "Correct Output (only hot topics):\n"
            "[Summary]\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📌 ROUND 1\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Topic: Machine Learning Intro\n"
            "User Intent: Understand ML basics\n\n"
            "Summary:\n"
            "- ML enables computers to learn from data\n"
            "- Three main types: supervised, unsupervised, reinforcement learning\n"
            "- Common algorithms: linear regression, decision trees, neural networks\n"
        )
        
        # Call LLM
        parsed = parse_uri(self.summarizer_model_uri)
        config_dict = {
            'api_key': self.summarizer_api_key,
            'model': parsed['model']
        }
        if parsed['base_url']:
            config_dict['base_url'] = parsed['base_url']
        
        provider = create_provider(
            parsed['provider'],
            config_dict,
            category=ProviderCategory.LLM
        )
        
        summarize_messages = [
            SystemMessage(content=system_inst),
            HumanMessage(content=prompt)
        ]
        
        response = provider.send(
            summarize_messages, 
            stream=False, 
            temperature=0.2
        )
        
        if isinstance(response, AIMessage):
            if response.content:
                return response.content.strip()
            raise ContextError("Summarizer model returned empty response")
        else:
            raise ContextError(f"Unexpected response type: {type(response)}")
