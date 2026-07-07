"""
Tool Verbose Logging Demo

Demonstrates ``conv.verbose.on()`` / ``conv.verbose.off()`` to toggle
tree-style tool-call trace logging at runtime.

When verbose is ON, each tool-execution round prints a structured tree block
showing tool name, call_id, arguments, result/error, and wall-clock timing.
The trace is purely additive — existing log lines and streaming output are
unaffected.

Prerequisites:
    Set your LLM provider credentials. For example:

        export OPENAI_API_KEY=sk-...
        export CHAK_MODEL_URI="openai:gpt-4o-mini"

    Or any other supported provider:

        export DASHSCOPE_API_KEY=sk-...
        export CHAK_MODEL_URI="bailian@https://dashscope.aliyuncs.com/compatible-mode/v1:qwen-plus"

Usage:
    python examples/tool_verbose_demo.py
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
from chak import Conversation


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
    print("  Tool Verbose Logging Demo")
    print("=" * 60)

    conv = Conversation(
        model_uri=MODEL_URI,
        api_key=API_KEY,
        tools=[get_current_time, calculator, echo],
        system_prompt="You are a helpful assistant. Use tools when needed.",
    )

    # ------------------------------------------------------------------
    # Round 1 — verbose OFF (default): only minimal tool-call log lines
    # ------------------------------------------------------------------
    print("\n--- Round 1: verbose OFF ---\n")
    response = await conv.asend(
        "What time is it now? Also calculate 123 * 456, and echo 'hello'."
    )
    print(f"\n[Assistant]: {response.content}\n")

    # ------------------------------------------------------------------
    # Round 2 — verbose ON: tree-style trace after each tool round
    # ------------------------------------------------------------------
    conv.verbose.on()
    print("\n--- Round 2: verbose ON ---\n")
    response = await conv.asend(
        "Now: add 99 and 1, also echo 'world', and tell me the time again."
    )
    print(f"\n[Assistant]: {response.content}\n")

    # ------------------------------------------------------------------
    # Round 3 — verbose OFF again (dynamic toggle)
    # ------------------------------------------------------------------
    conv.verbose.off()
    print("\n--- Round 3: verbose OFF again ---\n")
    response = await conv.asend("What was the last thing I asked?")
    print(f"\n[Assistant]: {response.content}\n")

    print("Done! 🎉")


if __name__ == "__main__":
    asyncio.run(main())
