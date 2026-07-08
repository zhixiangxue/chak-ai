"""
Conversation Fluent Settings Demo

Demonstrates the fluent configuration API for conversation-level settings:

    conv.tool.verbose.on()                          # enable verbose tool logging
    conv.tool.verbose.off()                         # disable verbose tool logging
    conv.tool.loop.max(100)                         # set max tool-call iterations
    conv.tool.loop.unlimited()                      # remove iteration limit
    conv.tool.executor.use(ToolExecutor.THREAD)     # switch execution mode
    conv.fallback.on(FallbackOn.RETRYABLE_ERRORS)   # set fallback trigger

Deprecated (will be removed in v0.5):
    Conversation(tool_executor=..., fallback_on=...)
    conv.set_tool_executor(...)

Prerequisites:
    Set your LLM provider credentials. For example:

        export OPENAI_API_KEY=sk-...
        export CHAK_MODEL_URI="openai:gpt-4o-mini"

Usage:
    python examples/conv_setting.py
"""

import asyncio
import os
from datetime import datetime

import dotenv

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Config — use env vars or sensible defaults
# ---------------------------------------------------------------------------
MODEL_URI = "deepseek/deepseek-v4-pro"
API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Keep log level at INFO so the tree trace stands out (DEBUG would be noisy).
os.environ.setdefault("CHAK_LOG_LEVEL", "INFO")

import chak
from chak import Conversation, ToolExecutor, FallbackOn


# ============================================================================
# Native tool functions (no decorators needed)
# ============================================================================

def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculator(operation: str, a: float, b: float) -> float:
    """
    Perform a simple arithmetic calculation.

    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'.
        a: First operand.
        b: Second operand.
    """
    ops = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float("inf"),
    }
    if operation not in ops:
        return f"Unknown operation: {operation}"
    return ops[operation](a, b)


def echo(message: str) -> str:
    """Echo the given message back."""
    return f"Echo: {message}"


# ============================================================================
# Demo
# ============================================================================

async def main():
    print("=" * 60)
    print("  Conversation Fluent Settings Demo")
    print("=" * 60)

    conv = Conversation(
        model_uri=MODEL_URI,
        api_key=API_KEY,
        tools=[get_current_time, calculator, echo],
        system_prompt="You are a helpful assistant. Use tools when needed.",
    )

    # ------------------------------------------------------------------
    # 1. conv.tool.verbose — toggle tree-style tool-call trace logging
    # ------------------------------------------------------------------
    print("\n--- 1. verbose OFF (default) ---\n")
    response = await conv.asend(
        "What time is it now? Also calculate 123 * 456, and echo 'hello'."
    )
    print(f"\n[Assistant]: {response.content}\n")

    conv.tool.verbose.on()
    print("\n--- 1. verbose ON ---\n")
    response = await conv.asend(
        "Now: add 99 and 1, also echo 'world', and tell me the time again."
    )
    print(f"\n[Assistant]: {response.content}\n")
    conv.tool.verbose.off()

    # ------------------------------------------------------------------
    # 2. conv.tool.executor — switch execution mode at runtime
    # ------------------------------------------------------------------
    print("\n--- 2. executor: ASYNCIO -> THREAD ---\n")
    print(f"Current executor: {conv.tool.executor.mode}")
    conv.tool.executor.use(ToolExecutor.THREAD)
    print(f"New executor:     {conv.tool.executor.mode}")

    # ------------------------------------------------------------------
    # 3. conv.tool.loop — control iteration limits
    # ------------------------------------------------------------------
    print("\n--- 3. loop config ---\n")
    print(f"Current max iterations: {conv.tool.loop.max_iterations}")
    conv.tool.loop.max(100)
    print(f"New max iterations:     {conv.tool.loop.max_iterations}")

    # ------------------------------------------------------------------
    # 4. conv.fallback — set fallback trigger condition
    #    (only meaningful when fallbacks are configured in __init__)
    # ------------------------------------------------------------------
    print("\n--- 4. fallback config ---\n")
    print(f"Current fallback mode: {conv.fallback.mode}")
    conv.fallback.on(FallbackOn.RETRYABLE_ERRORS)
    print(f"New fallback mode:     {conv.fallback.mode}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
