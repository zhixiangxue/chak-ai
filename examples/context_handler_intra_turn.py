"""
Intra-turn Context Compression (handle_round)
=============================================

Compress the in-flight message list *inside* a single ``asend()`` — before
every LLM round in the tool loop — so that large tool results (PDF pages,
long file listings, dumps, etc.) don't get re-sent verbatim on every
subsequent round.

Why this matters
----------------
Without round-level compression, a research-style agent that reads N large
tool results across N rounds pays O(N²) input tokens: iteration K sends
tool_results[0..K-1] plus the current tool_result again.  With
``handle_round`` you can keep only the most recent tool cycles and replace
older ones with a compact placeholder.

Concept recap
-------------
* **turn**  — one HumanMessage → final AIMessage cycle (one ``asend()``).
* **round** — one chak↔LLM request/response inside a turn.  A turn with N
  tool cycles has N+1 rounds.

``BaseContextHandler`` offers two hooks:

* ``handle_turn(messages, *, conversation_id)`` — called once per turn
  (before the first round).  Use for **inter-turn** compression.
* ``handle_round(messages, *, conversation_id, round_index)`` — called
  before **every** round.  Use for **intra-turn** compression.

Both default to no-op; override whichever you need.

This example
------------
1. Defines two mock tools that return "big" strings (fake PDF pages).
2. Runs a turn that reads 4 pages, so the tool loop takes ~5 rounds.
3. Uses ``PruningResearchHandler`` — a custom handler that keeps the
   ``keep_recent_cycles`` most recent tool cycles and offloads older ones.
4. Prints per-round token counts so you can see the O(N²) → O(N) win.

Prerequisites:
    export BAILIAN_API_KEY=your_key_here
"""

import asyncio
import os
from typing import List, Optional

import dotenv

dotenv.load_dotenv()

import chak
from chak.context.handlers import NoopContextHandler
from chak.message import AIMessage, Message, SystemMessage, ToolMessage


# ---------------------------------------------------------------------------
# 1) Mock tools that return "big" content — stands in for real PDF pages
# ---------------------------------------------------------------------------

FAKE_PAGES = {
    1: "Executive summary: revenue up 12% YoY. " * 200,
    2: "Segment breakdown: NA 45%, EU 30%, APAC 25%. " * 200,
    3: "Risk factors: FX exposure, supply chain, regulatory. " * 200,
    4: "Outlook: guidance raised to $2.1B for FY25. " * 200,
}


def read_pdf_page(page: int) -> str:
    """Read a single page from the (fake) annual report PDF.

    Args:
        page: 1-based page number to read.
    """
    return FAKE_PAGES.get(page, f"(page {page} not found)")


def save_note(text: str) -> str:
    """Save a short note to the scratchpad.

    Args:
        text: The note text to persist.
    """
    return f"Saved: {text[:60]}..."


# ---------------------------------------------------------------------------
# 2) Custom handler: keep last N tool cycles, offload the rest
# ---------------------------------------------------------------------------

class PruningResearchHandler(NoopContextHandler):
    """Keep the most recent ``keep_recent_cycles`` tool cycles in the
    round-scoped context; replace older cycles with a compact placeholder.

    A "tool cycle" is one ``AIMessage(tool_calls=...)`` followed by all of
    its matching ``ToolMessage``\\ s.  Cycles are always removed as a whole
    — never split — so the provider's tool_use / tool_result pairing
    invariant is preserved.  The base class integrity guard drops any
    stragglers as a safety net.
    """

    def __init__(self, keep_recent_cycles: int = 2):
        super().__init__()
        if keep_recent_cycles < 0:
            raise ValueError("keep_recent_cycles must be >= 0")
        self.keep_recent_cycles = keep_recent_cycles

    # ------------------------------------------------------------------
    # Turn-level pass-through (leave inter-turn history alone here; a
    # real app might chain FIFO / summarize on top).
    # ------------------------------------------------------------------
    def handle_turn(self, messages, *, conversation_id):
        return messages

    # ------------------------------------------------------------------
    # Round-level pruning — the meat of this example.
    # ------------------------------------------------------------------
    def handle_round(self, messages, *, conversation_id, round_index):
        cycles = self._split_into_cycles(messages)

        if len(cycles) <= self.keep_recent_cycles + 1:
            # +1 because the first "cycle" is really the prefix (system +
            # human, no tool_calls).  Nothing to prune yet.
            return messages

        prefix = cycles[0]
        tool_cycles = cycles[1:]
        stale = tool_cycles[: -self.keep_recent_cycles] if self.keep_recent_cycles else tool_cycles
        fresh = tool_cycles[-self.keep_recent_cycles :] if self.keep_recent_cycles else []

        placeholder = SystemMessage(
            content=(
                f"[Context compressed] {len(stale)} earlier tool cycle(s) "
                "were offloaded to save tokens. Their results are no longer "
                "available in this context — rely on your scratchpad / notes "
                "for any information you need to keep."
            )
        )

        pruned: List[Message] = list(prefix) + [placeholder]
        for cycle in fresh:
            pruned.extend(cycle)
        return pruned

    # ------------------------------------------------------------------
    # Internal: split a message list into [prefix, cycle1, cycle2, ...]
    # ------------------------------------------------------------------
    @staticmethod
    def _split_into_cycles(messages: List[Message]) -> List[List[Message]]:
        """Group messages by tool cycle.

        The first group is the "prefix" (system messages + the human turn
        + any leading assistant text without tool_calls).  Every subsequent
        group starts with an ``AIMessage(tool_calls=...)`` and includes all
        directly-following ``ToolMessage``\\ s that answer it.
        """
        groups: List[List[Message]] = [[]]
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                # Start a new cycle.
                groups.append([msg])
            elif isinstance(msg, ToolMessage) and len(groups) > 1:
                groups[-1].append(msg)
            else:
                # System / Human / final AIMessage without tool_calls goes
                # into the current group (prefix or the trailing final).
                groups[-1].append(msg)
        return groups


# ---------------------------------------------------------------------------
# 3) Run the demo
# ---------------------------------------------------------------------------

async def main() -> None:
    api_key = os.getenv("BAILIAN_API_KEY", "")
    if not api_key:
        print("❌ Error: Please set BAILIAN_API_KEY environment variable")
        return

    handler = PruningResearchHandler(keep_recent_cycles=2)

    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        context_handler=handler,
        tools=[read_pdf_page, save_note],
    )

    # Instrument via the built-in hook: print token counts for every
    # provider.send() call so you can see round-level compression at work.
    from chak.utils.logger import logger

    original_send = conv.provider.send

    def _spy_send(*args, **kwargs):
        msgs = kwargs.get("messages") or (args[0] if args else [])
        approx_chars = sum(len(str(m.content or "")) for m in msgs)
        print(
            f"    ↳ provider.send(): {len(msgs):>2} messages, "
            f"~{approx_chars:>6} chars of content"
        )
        return original_send(*args, **kwargs)

    conv.provider.send = _spy_send  # type: ignore[assignment]

    prompt = (
        "Read pages 1, 2, 3, and 4 of the annual report one at a time. "
        "After reading each page, save a one-sentence note using save_note. "
        "Finally, tell me what the outlook is."
    )
    print("=" * 72)
    print("Prompt:")
    print(f"  {prompt}")
    print("=" * 72)

    reply = await conv.asend(prompt)

    print("=" * 72)
    print("Final assistant reply:")
    print(f"  {reply.content}")
    print()
    print(
        f"Total messages in history: {len(conv.messages)}  "
        f"(includes all intermediate AIMessage/ToolMessage from the tool loop)"
    )
    print(
        "Notice above how the per-round char count stays roughly flat once "
        "keep_recent_cycles is reached — that's intra-turn compression at work."
    )


if __name__ == "__main__":
    asyncio.run(main())
