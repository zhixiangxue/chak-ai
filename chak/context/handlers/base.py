# chak/context/handlers/base.py
"""Base class for context management handlers."""

import warnings
from abc import ABC
from typing import List, Set
from copy import deepcopy

from ...message import Message, AIMessage, ToolMessage


class BaseContextHandler(ABC):
    """
    Base class for context management handlers.

    Design principles:
    - Input: complete messages + conversation_id
    - Output: context_messages for this round of LLM call
    - Handler can freely add/delete/modify messages in the output
    - chak only validates message types, no correctness guarantee

    ------------------------------------------------------------------
    Conceptual model — turn vs round
    ------------------------------------------------------------------
    A **turn** starts with a HumanMessage and ends with a final AIMessage.
    Inside a turn, chak may perform one or more **rounds** with the LLM.
    Each round is one request-response exchange — chak sends the current
    messages to the LLM via ``provider.send()``, receives an AIMessage
    back, and (if the response contains tool_calls) executes the requested
    tools before starting the next round.  A turn with no tool calls has
    exactly one round; a turn that involves N tool cycles has N+1 rounds.

    Two hooks are available:

    * :meth:`handle_turn` — invoked **once per turn**, before the very first
      round is dispatched.  This is the classic ``asend()`` entry point
      compression: prune / summarize inter-turn history here.

    * :meth:`handle_round` — invoked **before every round** inside the
      current turn (including round 0).  This is where you compress
      in-flight tool loop history to avoid quadratic input-token growth
      when tool results (e.g. large PDFs) pile up during multi-step
      tool-calling.

    Both hooks default to no-op; subclasses opt in by overriding either
    or both.
    """

    def __init__(self):
        """Initialize handler."""
        self.input_messages = []
        self.output_messages = []

    # ------------------------------------------------------------------
    # Backwards-compatibility bridge
    # ------------------------------------------------------------------
    def __init_subclass__(cls, **kwargs):
        """Auto-bridge subclasses that still override the deprecated ``handle``.

        Historically ``BaseContextHandler`` exposed a single abstract method
        ``handle(messages, *, conversation_id)``.  The framework has since
        split responsibility into :meth:`handle_turn` (inter-turn) and
        :meth:`handle_round` (intra-turn).  Existing third-party handlers
        that only override the legacy ``handle`` must keep working with no
        code change on the user side.

        Strategy: when a subclass defines ``handle`` but not ``handle_turn``,
        wire the subclass's ``handle`` implementation into ``handle_turn`` at
        class-creation time.  Framework internals only ever call
        ``handle_turn``, so the user's compression logic still runs.
        A single ``DeprecationWarning`` is emitted so users know to rename.
        """
        super().__init_subclass__(**kwargs)
        if 'handle' in cls.__dict__ and 'handle_turn' not in cls.__dict__:
            warnings.warn(
                f"{cls.__name__} overrides BaseContextHandler.handle, which is "
                f"deprecated. Rename the method to handle_turn (same signature).",
                DeprecationWarning,
                stacklevel=2,
            )
            # Alias the legacy override onto handle_turn so framework calls hit it.
            cls.handle_turn = cls.__dict__['handle']

    # ------------------------------------------------------------------
    # Public entry points (called by the framework)
    # ------------------------------------------------------------------
    def __call__(
        self,
        messages: List[Message],
        *,
        conversation_id: str,
    ) -> List[Message]:
        """
        Turn-scoped entry point.  Invoked once per ``asend()``, before the
        first round is dispatched to the LLM.

        Args:
            messages: Complete conversation history (read-only snapshot)
            conversation_id: Unique ID for this conversation

        Returns:
            context_messages: Messages to send to LLM in this round
        """
        self.input_messages = deepcopy(messages)
        processed_messages = self.handle_turn(messages, conversation_id=conversation_id)
        processed_messages = self._ensure_tool_call_integrity(processed_messages)
        self.output_messages = deepcopy(processed_messages)
        return processed_messages

    def call_for_round(
        self,
        messages: List[Message],
        *,
        conversation_id: str,
        round_index: int,
    ) -> List[Message]:
        """
        Round-scoped entry point.  Invoked before every round inside a turn
        (including round 0), typically from the tool loop.

        Mirrors :meth:`__call__` but delegates to :meth:`handle_round` and
        passes the current ``round_index``.  Same deep-copy + integrity
        pipeline applies.

        Args:
            messages: In-flight message list for the current round
                      (already includes any AIMessage/ToolMessage produced by
                      preceding rounds in this turn).
            conversation_id: Unique ID for this conversation.
            round_index: 0-based index of the round about to be dispatched.

        Returns:
            context_messages: Messages to send to LLM in this round.
        """
        self.input_messages = deepcopy(messages)
        processed_messages = self.handle_round(
            messages,
            conversation_id=conversation_id,
            round_index=round_index,
        )
        processed_messages = self._ensure_tool_call_integrity(processed_messages)
        self.output_messages = deepcopy(processed_messages)
        return processed_messages

    # ------------------------------------------------------------------
    # Integrity guardrails
    # ------------------------------------------------------------------
    def _ensure_tool_call_integrity(self, messages: List[Message]) -> List[Message]:
        r"""Repair tool_call pairing after handler-driven truncation.

        Strict providers (OpenAI, Anthropic, ...) require every
        ``AIMessage.tool_calls`` entry to be immediately answered by one
        ``ToolMessage`` bearing the matching ``tool_call_id``.  Handlers that
        prune context can violate this invariant in two ways:

        1. **Orphan ``ToolMessage``** — the answering ``ToolMessage`` remains
           but its parent ``AIMessage(tool_calls=...)`` was dropped.  OpenAI
           rejects this with ``messages with role 'tool' must be a response
           to a preceding message with 'tool_calls'``.
        2. **Orphan ``AIMessage(tool_calls=...)``** — the tool-calling
           ``AIMessage`` remains but one or more of its answering
           ``ToolMessage``\s were dropped.  Anthropic rejects this with a 400
           because a ``tool_use`` block is missing its ``tool_result``.

        This method runs in two phases:

        * Phase 1 (reverse-orphan sweep): scan the messages once and record
          every ``tool_call_id`` that is *answered* somewhere later in the
          sequence.  Any ``AIMessage`` whose ``tool_calls`` contain at least
          one id without an answer is marked for removal along with any of
          its matching ``ToolMessage``\s (those ``ToolMessage``\s would
          otherwise become orphans of the second kind after we drop the
          ``AIMessage``).
        * Phase 2 (forward orphan sweep): reproduces the original
          left-to-right filter — a ``ToolMessage`` is kept only when a
          preceding ``AIMessage`` in the retained sequence expected it.

        Occasionally this drops an ``AIMessage`` that a naive reader would
        consider legitimate (e.g. handler kept the assistant but dropped its
        tool answers).  This is the safer default: providers returning 400
        is much worse than silently pruning one message.

        Args:
            messages: Context messages produced by ``handle_turn`` /
                      ``handle_round``.

        Returns:
            Messages with orphaned assistant / tool messages removed.
        """
        # ----- Phase 1: reverse-orphan sweep on AIMessage(tool_calls) -----
        # Collect every tool_call_id that is actually answered downstream.
        answered_ids: Set[str] = set()
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.tool_call_id:
                answered_ids.add(msg.tool_call_id)

        # Identify AIMessages missing at least one answer; drop them AND any
        # ToolMessage that answers a doomed AIMessage (would become orphan).
        dropped_ai_ids: Set[int] = set()  # object ids of AIMessages to drop
        answers_of_dropped: Set[str] = set()  # tool_call_ids attached to dropped AIs
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                expected = {tc.id for tc in msg.tool_calls}
                if not expected.issubset(answered_ids):
                    dropped_ai_ids.add(id(msg))
                    answers_of_dropped.update(expected)

        filtered: List[Message] = []
        for msg in messages:
            if isinstance(msg, AIMessage) and id(msg) in dropped_ai_ids:
                continue
            if (
                isinstance(msg, ToolMessage)
                and msg.tool_call_id in answers_of_dropped
            ):
                continue
            filtered.append(msg)

        # ----- Phase 2: forward orphan sweep on ToolMessage -----
        result: List[Message] = []
        pending_tool_call_ids: Set[str] = set()

        for msg in filtered:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    pending_tool_call_ids.add(tc.id)
                result.append(msg)
            elif isinstance(msg, ToolMessage):
                if msg.tool_call_id and msg.tool_call_id in pending_tool_call_ids:
                    pending_tool_call_ids.discard(msg.tool_call_id)
                    result.append(msg)
                # else: orphan tool message — drop silently
            else:
                result.append(msg)

        return result

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------
    def handle_turn(
        self,
        messages: List[Message],
        *,
        conversation_id: str,
    ) -> List[Message]:
        """
        Turn-scoped context handling.  Called once per turn (``asend()``),
        before the first LLM round of that turn is dispatched.

        Default implementation is a no-op (returns ``messages`` unchanged).
        Subclasses override this to implement inter-turn compression:
        prune / summarize old conversation turns to keep the prompt within
        budget.

        Different handlers may apply different logic:
        - Noop: Return all messages unchanged
        - FIFO: Keep recent turns, drop old ones
        - LRU: Keep frequently accessed messages
        - Summarize: Compress old messages into summaries
        - Offload: Move large content to external storage

        Args:
            messages: Complete conversation history (read-only snapshot)
            conversation_id: Unique ID for this conversation (for
                             persistence / offload)

        Returns:
            context_messages: Messages to send to LLM for the first round of
                              the current turn.
        """
        return messages

    def handle_round(
        self,
        messages: List[Message],
        *,
        conversation_id: str,
        round_index: int,
    ) -> List[Message]:
        """
        Round-scoped context handling.  Called before **every** round in the
        current turn, including round 0.

        Default implementation is a no-op (returns ``messages`` unchanged).
        Subclasses override this to compress in-flight tool loop history —
        for example, to offload large tool results (PDF pages, file
        listings) from earlier tool cycles so they aren't re-sent to the
        LLM on every subsequent round.

        Semantics:

        * ``round_index=0`` — first round of the turn.  ``messages`` is the
          list already produced by :meth:`handle_turn`, so any turn-level
          compression has already been applied.  Overriding here at round 0
          is usually unnecessary; most implementations only kick in once
          ``round_index >= 1``.
        * ``round_index >= 1`` — ``messages`` contains ``round_index``
          completed tool cycles (``AIMessage(tool_calls)`` +
          ``ToolMessage``s).  This is where cycle-level pruning belongs.

        Cycle boundary rule (important):
            A "tool cycle" is one ``AIMessage(tool_calls)`` followed by
            **all** of its matching ``ToolMessage``s.  When you offload,
            offload *whole cycles* — never split a tool-calling
            ``AIMessage`` from its answers.  The integrity guard rail in
            :meth:`_ensure_tool_call_integrity` will drop any pair that
            gets separated, but doing this cleanly in the handler avoids
            surprises.

        Args:
            messages: In-flight message list for the current round.
            conversation_id: Unique ID for this conversation.
            round_index: 0-based index of the round about to be dispatched.

        Returns:
            context_messages: Messages to send to LLM for this round.
        """
        return messages

    # ------------------------------------------------------------------
    # Deprecated shim — kept for one release cycle
    # ------------------------------------------------------------------
    def handle(
        self,
        messages: List[Message],
        *,
        conversation_id: str,
    ) -> List[Message]:
        """DEPRECATED — use :meth:`handle_turn` instead.

        This method exists solely to keep pre-``handle_turn`` user code
        working.  It emits a ``DeprecationWarning`` on direct invocation
        and delegates to :meth:`handle_turn`.

        Subclasses that still override ``handle`` are auto-bridged in
        :meth:`__init_subclass__` — their ``handle`` becomes ``handle_turn``
        for the class, so framework internals continue to work.
        """
        warnings.warn(
            "BaseContextHandler.handle is deprecated; call handle_turn instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.handle_turn(messages, conversation_id=conversation_id)
