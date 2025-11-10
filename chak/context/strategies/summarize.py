# chak/context/strategies/summarize.py
"""Summarization context strategy that compresses history via LLM summarization."""

from typing import List, Optional, Callable

from .base import BaseContextStrategy, StrategyRequest, StrategyResponse
from ...exceptions import ContextError
from ...message import Message, MarkerMessage, SystemMessage, HumanMessage, AIMessage
from ...providers import create_provider
from ...providers.types import ProviderCategory
from ...utils.logger import logger
from ...utils.uri import parse as parse_uri


class SummarizationStrategy(BaseContextStrategy):
    """
    SummarizationStrategy - Lossless compression summarization strategy
    
    Purpose:
    - Implements lossless compression by inserting marker messages, without deleting original messages
    - Each marker summarizes all messages from previous marker (inclusive) to current marker (exclusive)
    
    Semantics:
    - StrategyResponse.messages: Returns complete message list (including strategy-inserted markers)
    - Conversation automatically extracts: system messages + last marker (inclusive) → last message
    
    Triggering:
    - Triggers when [system messages + last marker (inclusive) → last message] token count
      exceeds max_input_tokens * summarize_threshold
    - summarize_threshold defaults to 0.75 (75%)
    
    Turn Definition:
    - "Turn" boundaries defined by HumanMessage
    - prefer_recent_turns specifies how many recent turns to preserve from summarization
    
    Simplified Logic:
    - No token budget calculation, directly preserve specified turns
    - Summary length unlimited, let LLM naturally compress (modern models have large context windows)
    - Use semantic words (CONCISE) in prompt for better compression
    
    Parameters:
    - max_input_tokens (int): Model context window upper limit
    - summarize_threshold (float): Threshold ratio for triggering summarization (default 0.75)
    - prefer_recent_turns (int): Preserve recent turns (default 2)
    - token_counter (Optional[Callable[[str], int]]): Custom token counter
    - summarizer_model_uri (str): Summarizer model URI
    - summarizer_api_key (str): Summarizer model API key
    
    Notes:
    - System messages always included in messages sent to LLM
    - Marker messages record metadata: {type: "summary", summarized_count, summary}
    - Original message flow completely preserved, only marker tags inserted
    - prefer_recent_turns is target value, auto-adjusted if total turns insufficient
    """
    
    def __init__(
        self,
        max_input_tokens: int,
        summarize_threshold: float = 0.75,
        prefer_recent_turns: int = 2,
        token_counter: Optional[Callable[[str], int]] = None,
        summarizer_model_uri: str = "",
        summarizer_api_key: str = ""
    ):
        """Initialize SummarizationStrategy.
        
        Args:
            max_input_tokens: Maximum input token count (context window limit)
            summarize_threshold: Threshold ratio for triggering summarization (default 0.75 = 75%)
            prefer_recent_turns: Preserve recent turns (default 2)
            token_counter: Custom token counting function
            summarizer_model_uri: Summarizer model URI (required)
            summarizer_api_key: Summarizer model API key (required)
            
        Note:
            - "Turn" = 1 HumanMessage + subsequent messages (until next HumanMessage)
            - prefer_recent_turns is target value, auto-adjusted if total turns insufficient
            - Summary length unlimited, let LLM naturally compress (modern models have large windows)
            - When threshold exceeded, summarize earlier messages and insert marker
            
        Raises:
            ValueError: If max_input_tokens is not positive or summarizer config missing
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
        self.prefer_recent_turns = prefer_recent_turns
        self.summarizer_model_uri = summarizer_model_uri
        self.summarizer_api_key = summarizer_api_key
        
        # Calculate trigger point
        self.trigger_tokens = int(max_input_tokens * summarize_threshold)
    
    def process(self, request: StrategyRequest) -> StrategyResponse:
        """
        Process messages according to summarization strategy, implementing lossless compression.
        
        Logic flow:
        1. Extract system messages, marker messages, and conversation messages
        2. Find last marker position
        3. Calculate token count for analyzed messages: system messages + last marker (inclusive) → last
        4. If below threshold, return original messages
        5. If above threshold:
           a. Find preserve interval start point (count back prefer_recent_turns + 1 HumanMessage)
           b. Summarize from last marker (inclusive) to preserve interval start (exclusive)
           c. Generate summary and create marker
           d. Insert marker before preserve interval start
        6. Return complete message list
        
        Args:
            request: Strategy request containing message list
            
        Returns:
            Strategy response containing processed complete message list
        """
        messages = request.messages
        
        if not messages:
            return StrategyResponse(messages=[])
        
        # 1. Classify by type
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        # conversation_messages contains all conversation messages (excluding system messages and markers)
        # Note: conversation_messages here is for finding preserve interval, only count messages after marker
        last_marker_idx = self._find_last_summary_marker(messages)
        
        if last_marker_idx is not None:
            # Has marker: only count conversation messages after marker
            conversation_messages = [
                m for m in messages[last_marker_idx + 1:]  # After marker
                if not isinstance(m, (SystemMessage, MarkerMessage))
            ]
        else:
            # No marker: count all conversation messages
            conversation_messages = [
                m for m in messages 
                if not isinstance(m, (SystemMessage, MarkerMessage))
            ]
        
        # 2. Find last summary marker position (already found above)
        
        # 3. Determine messages to analyze: simulate messages actually sent to LLM
        #    Actual sending rule: system messages + last marker (inclusive) → last
        if last_marker_idx is not None:
            # Has marker: system messages + marker and messages after
            messages_to_analyze = list(system_messages) + messages[last_marker_idx:]
        else:
            # No marker: system messages + all conversation messages
            messages_to_analyze = list(system_messages) + list(conversation_messages)
        
        # 4. Calculate current token usage (simulate actual sent tokens)
        total_tokens = self.count_messages_tokens(list(messages_to_analyze))
        
        logger.debug(f"\n📊 [Summarization] Token stats: {total_tokens}/{self.max_input_tokens} (trigger={self.trigger_tokens}, {total_tokens/self.max_input_tokens*100:.1f}%)")
        
        # 5. Check if summarization needed
        if total_tokens <= self.trigger_tokens:
            # Below threshold, no summarization needed
            logger.debug(f"✅ [Summarization] Skip: below threshold ({total_tokens} <= {self.trigger_tokens})")
            return StrategyResponse(messages=messages)
        
        logger.debug(f"⚠️  [Summarization] Triggered: above threshold ({total_tokens} > {self.trigger_tokens})")
        
        # 6. Need summarization: dynamically find preserve interval start
        preserve_start_idx, actual_turns = self._find_preserve_start_adaptive(
            list(conversation_messages),
            list(system_messages)
        )
        
        if preserve_start_idx is None or preserve_start_idx == 0:
            # Cannot find preserve interval or no summarizable messages
            logger.debug(f"❌ [Summarization] Cannot execute: no summarizable messages")
            logger.debug(f"   conversation_messages length: {len(conversation_messages)}")
            logger.debug(f"   preserve_start_idx: {preserve_start_idx}")
            return StrategyResponse(messages=messages)
        
        # 7. Determine summarization interval
        if last_marker_idx is not None:
            # Has marker: from last marker to preserve interval start
            # Need to find absolute position of preserve interval in original messages
            # preserve_start_idx in conversation_messages corresponds to position in original messages
            preserve_start_in_original = self._find_message_index_in_original(
                messages, conversation_messages[preserve_start_idx]
            )
            summarize_start = last_marker_idx
            summarize_end = preserve_start_in_original
        else:
            # No marker: from beginning to preserve interval start
            preserve_start_in_original = self._find_message_index_in_original(
                messages, conversation_messages[preserve_start_idx]
            )
            summarize_start = 0
            summarize_end = preserve_start_in_original
        
        to_summarize = messages[summarize_start:summarize_end]
        
        # Print summarization interval info
        logger.debug(f"\n📋 [Summarization] Executing:")
        logger.debug(f"   Total messages: {len(messages)}")
        logger.debug(f"   Summarize interval: [{summarize_start}:{summarize_end}] ({len(to_summarize)} messages)")
        logger.debug(f"   Preserve interval: [{preserve_start_in_original}:{len(messages)}] ({len(messages)-preserve_start_in_original} messages)")
        logger.debug(f"   Actual preserved turns: {actual_turns} (target:{self.prefer_recent_turns})")
        logger.debug(f"   Calling LLM to generate summary...")
        
        # 8. Generate summary (no max_tokens limit, let LLM freely generate)
        summary_text = self._llm_summarize(to_summarize)
        
        logger.debug(f"   ✅ Summary generated ({len(summary_text)} characters)")
        logger.debug(f"   Summary preview: {summary_text[:80]}...\n")
        
        # 10. Create marker
        marker = MarkerMessage(
            content=f"[Conversation Summary] {summary_text}",
            metadata={
                "type": "summary",
                "summarized_count": len(to_summarize),
                "summary": summary_text
            }
        )
        
        # 11. Build new message list: insert marker before preserve interval start
        new_messages = (
            messages[:preserve_start_in_original] + 
            [marker] + 
            messages[preserve_start_in_original:]
        )
        
        return StrategyResponse(messages=new_messages)
    
    def _find_last_summary_marker(self, messages: List[Message]) -> Optional[int]:
        """
        Find the index of the last summary marker.
        
        Args:
            messages: Message list
            
        Returns:
            Index of last summary marker, None if not found
        """
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, MarkerMessage):
                if msg.metadata.get("type") == "summary":
                    return i
        return None
    
    def _find_preserve_start(self, conversation_messages: List[Message]) -> Optional[int]:
        """
        Find preserve interval start point (count back prefer_recent_turns + 1 HumanMessage).
        
        Args:
            conversation_messages: Conversation message list (excluding system messages and markers)
            
        Returns:
            Start index of preserve interval (in conversation_messages), None if not found
        """
        if not conversation_messages:
            return None
        
        # Count back HumanMessage
        human_indices = []
        for i in range(len(conversation_messages) - 1, -1, -1):
            if isinstance(conversation_messages[i], HumanMessage):
                human_indices.append(i)
                # Found prefer_recent_turns + 1
                if len(human_indices) == self.prefer_recent_turns + 1:
                    return human_indices[-1]  # Return the first one
        
        # If message count less than prefer_recent_turns + 1, return None (nothing to summarize)
        return None
    
    def _find_preserve_start_adaptive(
        self, 
        conversation_messages: List[Message],
        system_messages: List[Message]
    ) -> tuple[Optional[int], int]:
        """
        Find preserve interval start point, preserving recent prefer_recent_turns turns.
        
        Simplified version: no token budget calculation, directly preserve specified turns.
        Modern model context windows are large enough (128k+), summary length is not a bottleneck.
        
        Args:
            conversation_messages: Conversation message list (excluding system messages and markers)
            system_messages: System message list (parameter kept for caller compatibility)
            
        Returns:
            (Preserve interval start index, actual preserved turns), (None, 0) if not found
        """
        if not conversation_messages:
            return None, 0
        
        # Count back all HumanMessage indices
        human_indices = []
        for i in range(len(conversation_messages) - 1, -1, -1):
            if isinstance(conversation_messages[i], HumanMessage):
                human_indices.append(i)
        
        if len(human_indices) <= 1:
            # Only 0 or 1 turns, cannot summarize
            return None, 0
        
        # Calculate actual preserved turns
        turns_to_keep = min(self.prefer_recent_turns, len(human_indices) - 1)
        
        # Preserve interval start is the (turns_to_keep + 1)th HumanMessage from back
        preserve_start_idx = human_indices[turns_to_keep]
        
        return preserve_start_idx, turns_to_keep
    
    def _find_message_index_in_original(
        self, 
        original_messages: List[Message], 
        target_message: Message
    ) -> int:
        """
        Find target message index in original message list.
        
        Args:
            original_messages: Original message list
            target_message: Target message
            
        Returns:
            Target message index in original list
            
        Raises:
            ContextError: If target message not found
        """
        for i, msg in enumerate(original_messages):
            if msg is target_message:
                return i
        raise ContextError(f"Target message not found in original message list")
    

    def _llm_summarize(self, messages: List[Message]) -> str:
        """
        Generate summary using configured model.
        
        Args:
            messages: Message list to summarize
            
        Returns:
            LLM-generated summary text
        """
        # Build prompt from old messages
        segments: List[str] = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = "User"
                text = (m.content or "").strip()
            elif isinstance(m, AIMessage):
                role = "Assistant"
                text = (m.content or "").strip()
            elif isinstance(m, MarkerMessage):
                # For marker, extract pure summary content from metadata, not content (content has "[Conversation Summary]" prefix)
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
        
        # Build summarization prompt with few-shot examples
        system_inst = (
            "You are a conversation summarizer. Your task is to create a CUMULATIVE summary "  
            "that preserves all previous rounds and adds new round information.\n\n"
            "## Output Structure (MANDATORY)\n\n"
            "[Summary]\n"
            "Each round should have:\n"
            "  - Topic: <what this round discussed>\n"
            "  - User Intent: <what user wanted in this round>\n"
            "  - Summary: <CONCISE summary of key points - 3-5 bullet points max>\n\n"
            "---\n\n"
            "## CRITICAL RULES\n\n"
            "### Rule 1: Previous Rounds Must Be Copied Exactly\n"
            "If there is 'Previous Summary' in the input:\n"
            "  1. Copy ALL previous rounds COMPLETELY (word-by-word)\n"
            "  2. Then APPEND new round information at the end\n"
            "  3. DO NOT shorten, compress, or rewrite previous rounds\n"
            "  4. Think: New Summary = All Previous Rounds (unchanged) + New Round\n\n"
            "### Rule 2: New Round Must Be CONCISE Summary\n"
            "For the NEW round you are creating:\n"
            "  - Extract ONLY the most important 3-5 key points\n"
            "  - Each bullet point: 1-2 sentences maximum\n"
            "  - Remove examples, detailed explanations, tables, formulas\n"
            "  - Focus on: core concepts, main conclusions, key differences\n"
            "  - DO NOT copy-paste full paragraphs from Assistant's response\n\n"
            "Think: If someone reads only your summary, they should understand the essence.\n\n"
            "---\n\n"
            "## Example 1: First Summary (No Previous Summary)\n\n"
            "Input:\n"
            "  User: Tell me about AI history\n"
            "  Assistant: [3000 words detailed response about AI history...]\n\n"
            "Output:\n"
            "[Summary]\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📌 ROUND 1\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Topic: Artificial Intelligence History\n"
            "User Intent: Learning AI development timeline\n\n"
            "Summary:\n"
            "- AI evolved through 6 stages: 1940s embryonic period, 1956 birth at Dartmouth, 1970s first winter, 1980s expert systems, 1990s second winter, 2000s+ modern rise\n"
            "- Key milestones: 1950 Turing Test, 2006 deep learning concept, 2012 ImageNet breakthrough, 2016 AlphaGo victory\n"
            "- Current era driven by big data, GPU computing, and breakthrough algorithms (GPT, BERT, etc.)\n\n"
            "---\n\n"
            "## Example 2: Second Summary (Has Previous Summary)\n\n"
            "Input:\n"
            "  Previous Summary: [Summary with Round 1...]\n"
            "  User: What's the difference between ML and DL?\n"
            "  Assistant: [2000 words with definitions, tables, examples...]\n\n"
            "Output:\n"
            "[Summary]\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📌 ROUND 1 (COPIED FROM PREVIOUS SUMMARY)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Topic: Artificial Intelligence History\n"
            "User Intent: Learning AI development timeline\n\n"
            "Summary:\n"
            "- AI evolved through 6 stages: 1940s embryonic period, 1956 birth at Dartmouth, 1970s first winter, 1980s expert systems, 1990s second winter, 2000s+ modern rise\n"
            "- Key milestones: 1950 Turing Test, 2006 deep learning concept, 2012 ImageNet breakthrough, 2016 AlphaGo victory\n"
            "- Current era driven by big data, GPU computing, and breakthrough algorithms (GPT, BERT, etc.)\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "📌 ROUND 2 (NEW CONTENT)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Topic: Machine Learning vs Deep Learning\n"
            "User Intent: Understanding core differences between ML and DL\n\n"
            "Summary:\n"
            "- Relationship: DL is a subset of ML\n"
            "- Key differences: ML needs manual feature engineering, DL learns features automatically; ML works with small data, DL needs large datasets and GPUs\n"
            "- ML is more interpretable (e.g., decision trees), DL is 'black box' but powerful for images/speech/text\n\n"
            "---\n\n"
            "## Key Points\n"
            "- Each round has its own Topic + User Intent + Summary\n"
            "- Previous rounds: copy exactly (word-by-word)\n"
            "- NEW round: extract 3-5 key points ONLY (concise summary, not full content)\n"
            "- Use the SAME LANGUAGE as input messages\n"
        )
        summarize_messages = [
            SystemMessage(content=system_inst),
            HumanMessage(content=prompt)
        ]
        
        # Call provider to generate summary (no max_tokens limit, let LLM naturally compress)
        response = provider.send(
            summarize_messages, 
            stream=False, 
            temperature=0.2
        )
        
        # Extract content
        if isinstance(response, AIMessage):
            if response.content:
                return response.content.strip()
            raise ContextError("Summarizer model returned empty response")
        else:
            raise ContextError(f"Unexpected response type: {type(response)}")
