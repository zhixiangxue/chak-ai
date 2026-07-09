"""
Hook Instrumentor — Lightweight Observability with Tool Calling

Demonstrates a zero-intrusion observability layer: before_send timestamps the
start, after_send computes latency + token usage — all transparent to the
caller, whether the LLM uses tools or not.

Architecture:
    conv.hook.before_send  →  mark wall-clock start
    conv.hook.after_send   →  compute latency, extract usage, print metrics

This is the second stage of the three-layer observability evolution:
    1. monkey-patch  →  2. hooks (this example)  →  3. native OpenTelemetry

Prerequisites:
    Set DEEPSEEK_API_KEY.

Usage:
    python examples/hook_instrumentor.py
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

import dotenv

dotenv.load_dotenv()

import chak
from chak.message import AIMessage

# ============================================================================
# Tools
# ============================================================================


def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(a: float, b: float, operation: str = "add") -> float:
    """Perform a calculation on two numbers.

    Args:
        a: First number.
        b: Second number.
        operation: One of 'add', 'subtract', 'multiply', 'divide'.
    """
    op = operation.lower()
    if op == "add":
        return a + b
    elif op == "subtract":
        return a - b
    elif op == "multiply":
        return a * b
    elif op == "divide":
        return a / b if b != 0 else 0.0
    return 0.0


# ============================================================================
# Instrumentor
# ============================================================================


@dataclass
class RequestMetrics:
    """Per-request observability snapshot."""

    turn: int = 0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    tool_calls: list[str] = field(default_factory=list)  # tool names used this turn


class Instrumentor:
    """Lightweight instrumentor that collects metrics via hooks.

    Uses before_send to mark the wall-clock start and after_send to compute
    latency + token usage — all from within the hooks, no manual timing needed.

    Register on any Conversation:
        inst = Instrumentor()
        conv.hook.before_send(inst.before_send)
        conv.hook.after_send(inst.after_send)
    """

    def __init__(self):
        self._records: list[RequestMetrics] = []
        self._turn = 0
        self._t0: float = 0.0

    # -- Hook callbacks --------------------------------------------------
    async def before_send(self, conv, request, **send_kwargs):
        self._t0 = time.perf_counter()

    async def after_send(self, conv, request, **send_kwargs):
        self._turn += 1
        latency_ms = (time.perf_counter() - self._t0) * 1000

        # Pull the entire last turn — user + tool calls + final response
        turn_msgs = conv.get_messages(turns=-1)

        # Aggregate tokens across all messages in this turn
        prompt_total = 0
        completion_total = 0
        model = send_kwargs.get("model", "unknown")
        tool_names: list[str] = []

        for msg in turn_msgs:
            usage = msg.metadata.usage
            if usage:
                prompt_total += usage.prompt_tokens
                completion_total += usage.completion_tokens
            if msg.metadata.model:
                model = msg.metadata.model
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_names.append(tc.function.name)

        record = RequestMetrics(
            turn=self._turn,
            latency_ms=latency_ms,
            prompt_tokens=prompt_total,
            completion_tokens=completion_total,
            model=model,
            tool_calls=tool_names,
        )

        self._records.append(record)
        self._print(record)

    def _print(self, r: RequestMetrics):
        pt, ct = r.prompt_tokens, r.completion_tokens
        total = pt + ct
        tokens_str = f"prompt={pt} completion={ct} total={total}" if total else "<no usage>"
        tools_str = f"| tools=[{', '.join(r.tool_calls)}]" if r.tool_calls else ""
        print(
            f"\n📊 [turn {r.turn}] model={r.model} | "
            f"{tokens_str} {tools_str} | latency={r.latency_ms:.0f}ms"
        )

    @property
    def records(self) -> list[RequestMetrics]:
        return list(self._records)


# ============================================================================
# Demo
# ============================================================================


async def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠️  DEEPSEEK_API_KEY not set.  Set it to run this example.")
        return

    conv = chak.Conversation(
        "deepseek/deepseek-v4-pro",
        api_key=api_key,
        tools=[get_current_time, calculate],
        system_prompt="You are a helpful assistant. Use tools when needed.",
    )

    # Attach the instrumentor on both hook points
    inst = Instrumentor()
    conv.hook.before_send(inst.before_send)
    conv.hook.after_send(inst.after_send)

    print("=" * 60)
    print("Hook Instrumentor Demo (with tools)")
    print("=" * 60)

    prompts = [
        "What is the capital of France?",
        "What time is it right now?",
        "Calculate 123 * 456 and tell me the result.",
    ]

    for prompt in prompts:
        print(f"\n> {prompt}")
        response = await conv.asend(prompt)
        print(f"  {response.content}")  # type: ignore

    # Streaming — hooks work exactly the same, no special handling needed
    print("\n--- streaming ---")
    print("> Write a haiku about coding.")
    async for chunk in await conv.asend("Write a haiku about coding.", stream=True):
        if chunk.content:
            print(chunk.content, end="", flush=True)  # type: ignore
    print()

    # Summary
    print("\n" + "=" * 60)
    total_prompt = sum(r.prompt_tokens for r in inst.records)
    total_completion = sum(r.completion_tokens for r in inst.records)
    total_tool_calls = sum(len(r.tool_calls) for r in inst.records)
    print(f"Summary: {len(inst.records)} requests, "
          f"total prompt={total_prompt}, total completion={total_completion}, "
          f"tool calls={total_tool_calls}")


if __name__ == "__main__":
    asyncio.run(main())
