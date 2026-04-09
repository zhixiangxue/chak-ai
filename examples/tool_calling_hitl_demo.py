"""Human-in-the-Loop (HITL) demo

Shows how to intercept every tool call before execution using
``hitl_handler``.  Three decision paths are demonstrated:

  allow  - let the call proceed as-is
  abort  - cancel the call; the LLM receives a cancellation notice
  allow with overrides - let the call proceed but silently rewrite one argument

Usage:
    python examples/tool_calling_hitl_demo.py
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

import dotenv

import chak
from chak.tools.manager import HITLDecision, HITLRequest

dotenv.load_dotenv(Path(__file__).resolve().parents[1] / ".env")


# ---------------------------------------------------------------------------
# Demo tools
# ---------------------------------------------------------------------------

def get_current_time() -> str:
    """Return the current date and time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def greet(name: str) -> str:
    """Return a greeting message for the given name."""
    return f"Hello, {name}!"


# ---------------------------------------------------------------------------
# HITL handler
# ---------------------------------------------------------------------------

async def hitl_handler(request: HITLRequest) -> HITLDecision:
    """Interactive console handler: ask the developer before each tool call.

    Demonstrates three outcomes:
    - 'y'  → allow the call unchanged
    - 'n'  → abort the call
    - 'e'  → allow but let developer edit one argument override
    """
    print(f"\n[HITL] Tool     : {request.tool_name}")
    print(f"[HITL] Arguments: {request.arguments}")
    answer = input("[HITL] Allow? (y=yes / n=abort / e=edit args): ").strip().lower()

    if answer == "n":
        print("[HITL] Aborted.")
        return HITLDecision.abort()

    if answer == "e":
        key = input("  Argument key to override: ").strip()
        value = input(f"  New value for '{key}': ").strip()
        print(f"[HITL] Allowed with override: {key}={value!r}")
        return HITLDecision.allow(overrides={key: value})

    print("[HITL] Allowed.")
    return HITLDecision.allow()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")

    conv = chak.Conversation(
        model_uri="openai/gpt-4o-mini",
        api_key=api_key,
        tools=[get_current_time, greet],
        hitl_handler=hitl_handler,
    )

    async for event in await conv.asend(
        "Please tell me the current time and greet Alice.",
        event=True,
    ):
        if isinstance(event, chak.MessageChunk):
            if event.content:
                print(event.content, end="", flush=True)
            if event.is_final:
                print()
        elif isinstance(event, chak.ToolCallStartEvent):
            print(f"\n[Tool call] {event.tool_name}  args={event.arguments}")
        elif isinstance(event, chak.ToolCallSuccessEvent):
            print(f"[Tool result] {event.tool_name} -> {event.result}")
        elif isinstance(event, chak.ToolCallCancelledEvent):
            print(f"[Tool cancelled] {event.tool_name}")


if __name__ == "__main__":
    asyncio.run(main())
