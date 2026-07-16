"""
Inspector Demo — Multi-turn agent task with live browser observation

Demonstrates chak.inspector.watch(): open http://127.0.0.1:7878 to watch
messages arrive in real time (HumanMessage / AIMessage / tool_calls /
ToolMessage) without waiting for the agent to finish.

Task design:
  Three connected conversation turns progressively exploring Python's
  asyncio. Each turn triggers tool calls so the browser clearly shows
  the Turn 1 / Turn 2 / Turn 3 group structure.

Prerequisites:
    export DEEPSEEK_API_KEY=sk-xxx       # DeepSeek
    pip install 'chakpy[server]'         # inspector needs fastapi + uvicorn

Usage:
    python examples/inspector_demo.py
"""

import asyncio
import os
import re
from datetime import datetime

import dotenv
import httpx

dotenv.load_dotenv()

import chak
from chak.inspector import watch


# ---------------------------------------------------------------------------
# Native Python functions exposed as LLM tools
# ---------------------------------------------------------------------------
# We use the English docs.python.org because it is globally reachable,
# serves static HTML with no anti-scraping, and the topic ties neatly into
# a coherent "research asyncio" multi-turn task.

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


async def fetch_url(url: str, max_chars: int = 2500) -> str:
    """
    Fetch a web page and return its body text (HTML stripped, whitespace
    collapsed).

    Args:
        url: The URL to fetch (http/https).
        max_chars: Maximum characters to return (prevents context overflow,
                   default 2500).

    Returns:
        Extracted plain text, truncated to *max_chars* if necessary.
    """
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    # Strip HTML tags + collapse whitespace. Good enough for docs pages.
    text = _HTML_TAG_RE.sub(" ", r.text)
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + " …[truncated]"
    return text


def count_chars(text: str) -> int:
    """
    Count the number of characters in a piece of text (used for quantitative
    comparison by the LLM).

    Args:
        text: Any text.

    Returns:
        Total character count.
    """
    return len(text)


def get_current_time() -> str:
    """
    Get the current date and time, used for report timestamps.

    Returns:
        A string in the format ``'2026-07-14 16:30:00'``.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Every turn drills into one facet, so tool calls line up cleanly with turns.
TURNS = [
    {
        "label": "Turn 1 · asyncio module overview",
        "prompt": (
            "Round 1 of research: First call `get_current_time` to record the "
            "start time; then call `fetch_url` to fetch "
            "https://docs.python.org/3/library/asyncio.html ; use `count_chars` "
            "to count the body characters; finally summarize in English what "
            "the **asyncio module itself does** (3-5 sentences)."
        ),
    },
    {
        "label": "Turn 2 · Tasks & coroutines in depth",
        "prompt": (
            "Round 2: Building on the previous round, I want to dive deeper into "
            "Tasks and coroutines. Call `fetch_url` to fetch "
            "https://docs.python.org/3/library/asyncio-task.html and use "
            "`count_chars` to count the characters. Then output in Markdown:\n"
            "- The key differences between **coroutines vs Tasks** (use a table)\n"
            "- A minimal runnable code snippet (with async/await)\n"
            "- 3 common pitfalls for beginners"
        ),
    },
    {
        "label": "Turn 3 · Event loop & final summary",
        "prompt": (
            "Final round: Call `fetch_url` to fetch "
            "https://docs.python.org/3/library/asyncio-eventloop.html , use "
            "`count_chars` to count the characters, then call `get_current_time` "
            "to record the end time. Synthesizing all 3 rounds, output a "
            "**complete research summary** in Markdown including:\n"
            "- A character-count comparison table for the three pages\n"
            "- How **asyncio / Task / event loop** relate to each other "
            "(explained clearly in one paragraph)\n"
            "- A **recommended reading order** for beginners with rationale\n"
            "- Report footer (start time → end time)"
        ),
    },
]


async def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not found.")
        print("       Get a key at https://platform.deepseek.com/ and add it to .env")
        return

    conv = chak.Conversation(
        "deepseek/deepseek-v4-pro",
        api_key=api_key,
        system_prompt=(
            "You are a rigorous technical research assistant conducting a "
            "multi-turn deep dive into Python asyncio. Each round I will give "
            "you explicit steps — proactively call tools when external resources "
            "are needed. After all necessary tool calls are done, produce your "
            "answer in English."
        ),
        tools=[fetch_url, count_chars, get_current_time],
    )
    # Give the conv a human-readable title so the inspector sidebar
    # shows something meaningful instead of the default "Untitled".
    conv.title = "Asyncio deep-dive (3-turn research)"

    # Secondary conv: another conversation in the same process with a different
    # model, to demonstrate multi-conv support. The sidebar tab bar will show
    # it as a second tab, and the stats table will display two distinct
    # model_uri rows (qwen-plus vs qwen-turbo).
    conv2 = chak.Conversation(
        "deepseek/deepseek-v4-flash",
        api_key=api_key,
        system_prompt="You are a lightweight QA assistant. Answer concisely.",
        tools=[get_current_time],
    )
    conv2.title = "Quick QA (deepseek-v4-flash)"

    # Attach both convs to the same inspector (same port). The first watch()
    # starts the server and opens the browser; subsequent watch() calls simply
    # register the new conv onto the running server — a new tab appears
    # automatically in the sidebar.
    watch(conv, port=9797)
    watch(conv2, port=9797)

    print()
    print("=" * 70)
    print("  Inspector Demo — Multi-turn agent task with live observation")
    print("=" * 70)
    print()
    print("Browser opened at http://127.0.0.1:7878")
    print("Every new message the agent appends will refresh the page.")
    print(f"  Main conv: {len(TURNS)} turns of asyncio research (deepseek-v4-pro)")
    print("  Side conv: a quick question (deepseek-v4-flash) — switch via sidebar tabs")
    print("  Starting in 3 seconds — switch to the browser and watch.")
    print()
    await asyncio.sleep(3)

    # Send a warm-up message to the secondary conv first so both tabs have
    # content to show.
    print("[Side conv] Sending a warm-up message…")
    await conv2.asend("Tell me the current time in one sentence (use the tool).")
    print("   -> Side conv has 1 turn. Now starting the main conv.")
    await asyncio.sleep(1)

    for idx, turn in enumerate(TURNS, 1):
        print()
        print("-" * 70)
        print(f"[{turn['label']}] User prompt sent…")
        print("-" * 70)
        response = await conv.asend(turn["prompt"])
        print(f"Turn {idx} done — assistant output {len(response.content)} chars")
        # Small pause between turns so the browser can visibly separate them.
        await asyncio.sleep(1)

    print()
    print("=" * 70)
    stats = conv.stats()
    print(f"Total messages: {stats['total_messages']}   "
          f"Total tokens: {stats['total_tokens']}   "
          f"Turns: {len(TURNS)}")
    print("=" * 70)
    print()
    print("The browser page stays open — you can review the full message stream.")
    print("Data is cached client-side: even after this process exits, you can")
    print("still browse and export from the inspector page.")
    print("Press Ctrl+C to exit.")
    print()

    # Hold the process open so the user can inspect the final state in browser.
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print("\nBye")


if __name__ == "__main__":
    asyncio.run(main())
