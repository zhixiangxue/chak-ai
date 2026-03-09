# chak/context/handlers/summarize.py
"""Summarization context handler."""

from typing import Dict, List, Tuple

import tiktoken

from .base import BaseContextHandler
from ...message import AIMessage, HumanMessage, Message, SystemMessage, ToolMessage
from ...providers import create_provider
from ...providers.types import ProviderCategory
from ...utils.uri import parse as parse_uri

# Characters of tool-result content included in the summary prompt.
# Kept as an internal constant so users don't need to tune it.
_TOOL_SUMMARY_MAX_CHARS = 500

# Approximate per-message overhead added to every encoded content length
# to account for role labels, separators, and other chat-format tokens.
_TOKENS_PER_MESSAGE_OVERHEAD = 4


class SummarizationContextHandler(BaseContextHandler):
    """
    SummarizationContextHandler - Compress old messages via LLM summarization.

    Token-budget based: tracks actual token usage against the target model's
    context window and compresses old messages once the threshold is reached.

    Example::

        handler = SummarizationContextHandler(
            context_window=128_000,           # GPT-4o context limit
            summarizer_uri="openai/gpt-4o-mini",
            api_key="sk-...",
        )
        # Triggers automatically when context exceeds 80% of the window
        # and compresses down to ~20%.

    Behavior:
    - Returns: [system messages] + [summary SystemMessage] + [recent messages]
    - Does NOT modify history_messages
    - Summary is cached per conversation; LLM is only called when the set of
      messages to be summarized grows (new old messages accumulate).
    - The kept window always starts at a HumanMessage boundary so providers
      like Claude never receive a context whose first non-system message is a
      ToolMessage or a bare AIMessage.
    - Tool message content in the summary prompt is capped at
      ``_TOOL_SUMMARY_MAX_CHARS`` characters to keep the prompt size manageable.

    Parameters:
    - context_window: Target LLM's context window size in tokens
    - summarizer_uri: Model URI for the summarizer (e.g. "openai/gpt-4o-mini")
    - api_key: API key for the summarizer
    - summarize_threshold: Fraction of context_window that triggers summarization
                           (default 0.8 → triggers at 80%)
    - keep_ratio: Fraction of context_window to retain after summarization
                  (default 0.2 → keeps ~20% worth of recent messages)
    - tiktoken_encoding: Tiktoken encoding for token estimation
                         (default "o200k_base", a safe approximation for all models)
    """

    def __init__(
        self,
        context_window: int,
        summarizer_uri: str,
        api_key: str,
        summarize_threshold: float = 0.8,
        keep_ratio: float = 0.2,
        tiktoken_encoding: str = "o200k_base",
    ):
        """
        Initialize summarization handler.

        Args:
            context_window: Target LLM's context window in tokens (e.g. 128_000).
            summarizer_uri: Model URI for the summarizer (e.g. "openai/gpt-4o-mini").
            api_key: API key for the summarizer.
            summarize_threshold: Trigger fraction of context_window (default 0.8).
                                  Summarization fires when total tokens exceed
                                  ``context_window * summarize_threshold``.
            keep_ratio: Target fraction of context_window to keep after compression
                        (default 0.2).  The kept window may be slightly smaller
                        because the split boundary is snapped to a HumanMessage.
            tiktoken_encoding: Tiktoken encoding name used for token estimation
                               (default "o200k_base").  Works as a conservative
                               approximation for non-OpenAI models.
        """
        super().__init__()
        if not summarizer_uri:
            raise ValueError("summarizer_uri is required")
        if not api_key:
            raise ValueError("api_key is required")
        if not (0 < summarize_threshold <= 1.0):
            raise ValueError("summarize_threshold must be in (0, 1]")
        if not (0 < keep_ratio < summarize_threshold):
            raise ValueError("keep_ratio must be in (0, summarize_threshold)")

        self.context_window = context_window
        self.summarize_threshold = summarize_threshold
        self.keep_ratio = keep_ratio
        self.tiktoken_encoding = tiktoken_encoding

        # Pre-load the tokenizer once and reuse across all calls.
        self._enc = tiktoken.get_encoding(tiktoken_encoding)

        # Derived token budgets
        self._trigger_tokens: int = int(context_window * summarize_threshold)
        self._keep_tokens: int = int(context_window * keep_ratio)

        # Initialize summarizer provider
        parsed = parse_uri(summarizer_uri)
        config = {
            'api_key': api_key,
            'model': parsed['model'],
        }
        if parsed['base_url']:
            config['base_url'] = parsed['base_url']
        config.update(parsed['params'])

        self.summarizer = create_provider(
            parsed['provider'],
            config,
            category=ProviderCategory.LLM
        )

        # Cache: maps conversation_id -> (boundary_key, summary_text)
        # boundary_key is a tuple of message IDs that were summarized;
        # if the same IDs appear again we reuse the cached text.
        self._summary_cache: Dict[str, Tuple[tuple, str]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def handle(
        self,
        messages: List[Message],
        *,
        conversation_id: str,
    ) -> List[Message]:
        """
        Return summarized context when token usage exceeds the threshold.

        Args:
            messages: Complete conversation history.
            conversation_id: Unique ID for this conversation.

        Returns:
            Context messages: system messages + summary + recent messages.
        """
        if not messages:
            return []

        # Separate system messages and conversation messages
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        conversation_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        # Check if summarization is needed based on total token count
        total_tokens = self._count_tokens(messages)
        if total_tokens <= self._trigger_tokens:
            return messages

        # Find the split boundary based on the keep token budget.
        # The boundary is snapped forward to the nearest HumanMessage so that
        # to_keep always starts cleanly (Claude requires first non-system
        # message to be user/system).
        keep_start = self._find_keep_start(conversation_messages)

        to_summarize = conversation_messages[:keep_start]
        to_keep = conversation_messages[keep_start:]

        if not to_summarize:
            # Edge case: cannot find a valid split point, return as-is
            return messages

        # Retrieve or generate summary (cached by message IDs)
        boundary_key = tuple(m.id for m in to_summarize)
        cached = self._summary_cache.get(conversation_id)
        if cached and cached[0] == boundary_key:
            summary_text = cached[1]
        else:
            summary_text = self._generate_summary(to_summarize)
            self._summary_cache[conversation_id] = (boundary_key, summary_text)

        summary_message = SystemMessage(
            content=f"[Previous conversation summary]\n{summary_text}"
        )

        # Return: system messages + summary + recent messages
        return system_messages + [summary_message] + to_keep

    def reset(self) -> None:
        """Clear the summary cache (called automatically by conversation.clear())."""
        self._summary_cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_tokens(self, messages: List[Message]) -> int:
        """
        Estimate the total token count for a list of messages.

        Uses the pre-loaded tiktoken encoder.  For non-OpenAI models the count
        is approximate but conservative (tends to slightly overcount CJK text),
        which is safe for threshold-based triggering.

        Args:
            messages: Messages to count.

        Returns:
            Estimated total token count.
        """
        total = 0
        for msg in messages:
            content = (
                msg.content if isinstance(msg.content, str)
                else str(msg.content or "")
            )
            total += len(self._enc.encode(content)) + _TOKENS_PER_MESSAGE_OVERHEAD
        return total

    def _find_keep_start(self, conversation_messages: List[Message]) -> int:
        """
        Find the start index for the kept window based on the keep token budget.

        Scans messages from the end, accumulating token counts until the keep
        budget (``_keep_tokens``) would be exceeded.  The boundary is then
        snapped forward to the nearest HumanMessage so ``to_keep`` always
        starts cleanly (required by Claude and other strict providers).

        Args:
            conversation_messages: Non-system messages in conversation order.

        Returns:
            Index of the first message in the kept window (a HumanMessage),
            or 0 if no valid boundary is found (graceful fallback).
        """
        accumulated = 0
        keep_start = len(conversation_messages)  # default: keep nothing

        for i in range(len(conversation_messages) - 1, -1, -1):
            msg = conversation_messages[i]
            content = (
                msg.content if isinstance(msg.content, str)
                else str(msg.content or "")
            )
            msg_tokens = len(self._enc.encode(content)) + _TOKENS_PER_MESSAGE_OVERHEAD

            if accumulated + msg_tokens > self._keep_tokens:
                break  # adding this message would exceed the keep budget
            accumulated += msg_tokens
            keep_start = i

        # Snap to HumanMessage to ensure to_keep starts cleanly.
        #
        # Strategy:
        # 1. Snap FORWARD from keep_start — this is the ideal case: a HumanMessage
        #    exists just after the budget cutoff, so to_keep stays within budget.
        # 2. If no HumanMessage is found forward (e.g. a single tool message is
        #    larger than the entire keep budget, pushing keep_start past the last
        #    HumanMessage in the window), snap BACKWARD to the nearest HumanMessage.
        #    This guarantees at least one complete turn is kept, even if its token
        #    count slightly exceeds the budget.
        for i in range(keep_start, len(conversation_messages)):
            if isinstance(conversation_messages[i], HumanMessage):
                return i

        # Forward snap found nothing — fall back to backward snap
        for i in range(keep_start - 1, -1, -1):
            if isinstance(conversation_messages[i], HumanMessage):
                return i

        # No HumanMessage at all in the conversation
        return 0

    def _generate_summary(self, messages: List[Message]) -> str:
        """
        Generate a summary for the given messages using the summarizer LLM.

        Args:
            messages: Messages to summarize.

        Returns:
            Summary text.
        """
        conversation_text = self._format_messages_for_summary(messages)
        prompt = (
            "Please provide a CONCISE summary of the following conversation. "
            "Focus on key topics, decisions, and important information.\n\n"
            f"Conversation:\n{conversation_text}\n\nSummary:"
        )

        summary_msg = SystemMessage(content=prompt)
        response = self.summarizer.send(messages=[summary_msg], stream=False)

        return response.content or "(Summary generation failed)"

    def _format_messages_for_summary(self, messages: List[Message]) -> str:
        """
        Format messages into readable text for the summarization prompt.

        Tool messages are truncated to ``_TOOL_SUMMARY_MAX_CHARS`` characters
        to keep the summary prompt within a reasonable token budget.
        AIMessages that contain only tool_calls are rendered compactly.

        Args:
            messages: Messages to format.

        Returns:
            Formatted conversation text.
        """
        lines: List[str] = []
        for msg in messages:
            role = msg.role

            if isinstance(msg, ToolMessage):
                content = (
                    msg.content if isinstance(msg.content, str)
                    else str(msg.content)
                )
                if len(content) > _TOOL_SUMMARY_MAX_CHARS:
                    content = content[:_TOOL_SUMMARY_MAX_CHARS] + "...[truncated]"
                lines.append(f"{role}(tool_result): {content}")

            elif isinstance(msg, AIMessage) and msg.tool_calls:
                tool_names = [tc.function.name for tc in msg.tool_calls]
                text_content = (
                    msg.content if isinstance(msg.content, str) else ""
                ) or ""
                if text_content:
                    lines.append(
                        f"{role}: {text_content} "
                        f"[called tools: {', '.join(tool_names)}]"
                    )
                else:
                    lines.append(f"{role}: [called tools: {', '.join(tool_names)}]")

            else:
                content = (
                    msg.content if isinstance(msg.content, str)
                    else str(msg.content)
                )
                lines.append(f"{role}: {content}")

        return "\n".join(lines)
