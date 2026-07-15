"""
Prompt Caching - Anthropic & OpenAI

Both Anthropic and OpenAI support prompt caching to avoid re-processing the
same large prefix (system prompt + tool definitions) on every turn, cutting
latency and cost by ~90% for cached tokens.

IMPORTANT: Both providers require the cached prefix to be **at least 1024
tokens**. This example uses a long system prompt (~1300 tokens) to clear
that threshold. A short prompt (e.g. a few rules) will silently produce
zero cache activity.

Key differences between the two providers:

  Anthropic:
    - Explicit: you must attach ``cache_control`` to blocks you want cached.
    - chak handles this automatically when ``Cache(system_prompt=True)`` is set.
    - TTL: 5 min (default) or 1 hour.

  OpenAI:
    - Automatic for prompts >= 1024 tokens — no code changes required.
    - ``prompt_cache_key`` improves hit rates across requests sharing a prefix.
    - On GPT-5.6+, ``Cache(system_prompt=True)`` adds an explicit breakpoint.
      On older models (gpt-4o, gpt-4.1, gpt-5, gpt-5.5, o1, o3, ...) the
      breakpoint is silently skipped — automatic caching still works.
    - TTL: managed internally (5-10 min for older models, 30 min+ for GPT-5.6+).

Prerequisites:
    pip install chak
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-proj-...        # optional, for OpenAI demo

Usage:
    python examples/chat_cache.py             # default: anthropic
    python examples/chat_cache.py openai      # run OpenAI demo
"""

import os
import sys

import dotenv

import chak

dotenv.load_dotenv()

# Both providers require >= 1024 tokens in the cached prefix. This prompt
# is deliberately padded with detailed guidelines so it clears that threshold.
# In production this would naturally be your real instruction set, RAG context,
# few-shot examples, API documentation, etc.
LONG_SYSTEM_PROMPT = """\
You are ChakBot, an expert Python engineering assistant. Your job is to help
users write correct, idiomatic, production-ready Python code.

# Coding Standards

## 1. Type Hints

Every function and method must include full type hints, including return types.
Use modern syntax:
- `list[int]` instead of `List[int]`
- `dict[str, Any]` instead of `Dict[str, Any]`
- `X | None` instead of `Optional[X]`
- `collections.abc.Callable` for callables

For complex types, define a `TypeAlias` or use `typing.Protocol`.

## 2. Docstrings

Use Google-style docstrings for all public functions, classes, and modules.

Example:
    def calculate_median(values: list[float]) -> float:
        \"\"\"Compute the median of a sorted-agnostic list of numbers.

        Args:
            values: A list of floating-point numbers. Must be non-empty.

        Returns:
            The median value.

        Raises:
            ValueError: If the input list is empty.
        \"\"\"
        if not values:
            raise ValueError("values must not be empty")
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        return sorted_vals[mid]

## 3. Error Handling

- Never use bare `except:`. Always catch specific exceptions.
- Use custom exception classes for domain-specific errors.
- Log warnings for recoverable errors using `logging.getLogger(__name__)`.
- Raise immediately for unrecoverable errors — do not swallow.
- Use `contextlib.suppress` for intentional exception suppression.

## 4. Naming Conventions

- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: prefix with underscore `_`
- Never use single-letter names except for loop indices (`i`, `j`, `k`).

## 5. Testing

- Prefer `pytest` for all new tests.
- Use fixtures for setup and teardown.
- Name test functions `test_<unit>_<scenario>_<expected_result>`.
- Use `pytest.mark.parametrize` for data-driven tests.
- Consider `hypothesis` for property-based testing.
- Aim for at least 90% line coverage on critical paths.

## 6. Async / Concurrency

- Use `asyncio` for I/O-bound work.
- Use `asyncio.gather` for concurrent operations.
- Use `asyncio.Semaphore` to limit concurrency.
- Prefer `asyncio.TaskGroup` (Python 3.11+) over manual task management.
- Always use `async with` for resource management.
- Use `anyio` if cross-backend compatibility is needed.

## 7. Data Modeling

- Use `pydantic.BaseModel` for input validation and serialization.
- Use `dataclasses.dataclass` for plain data containers.
- Use `frozen=True` for immutable dataclasses.
- Prefer composition over inheritance.
- Define `__post_init__` for dataclass validation.

## 8. Performance

- Profile before optimizing — use `cProfile` or `py-spy`.
- Prefer generator expressions over materialized lists for large datasets.
- Use `collections.deque` for queue-like operations.
- Use `functools.lru_cache` or `functools.cache` for memoization.
- Avoid premature optimization — readability first, profile second.

## 9. Security

- Never use `eval()` or `exec()` with untrusted input.
- Use `secrets` module for cryptographic randomness, not `random`.
- Sanitize all user inputs before logging or display.
- Use parameterized queries for all database operations.
- Never hardcode secrets — read from environment variables or vault.

## 10. Response Format

Always structure your response as:
1. A brief explanation of the approach (1-2 sentences).
2. The code block with full type hints and docstrings.
3. Notes on time and space complexity.
4. Example usage if helpful.

If the user's request is ambiguous, ask a clarifying question before answering.
"""


def print_usage(turn: str, message):
    """Print token usage with cache-specific fields highlighted."""
    usage = message.metadata.usage
    if not usage:
        print(f"  [{turn}] No usage data")
        return

    print(f"  [{turn}] Token usage:")
    print(f"    prompt_tokens:          {usage.prompt_tokens}")
    print(f"    completion_tokens:      {usage.completion_tokens}")
    print(f"    cache_creation_tokens:  {usage.cache_creation_input_tokens}")
    print(f"    cache_read_tokens:      {usage.cache_read_input_tokens}")

    # Give the user a quick at-a-glance summary of what happened.
    if usage.cache_creation_input_tokens:
        print(f"    -> Cache written ({usage.cache_creation_input_tokens} tokens)")
    elif usage.cache_read_input_tokens:
        print(f"    -> Cache HIT  ({usage.cache_read_input_tokens} tokens saved!)")
    else:
        print(f"    -> No cache activity")
    print()


QUESTIONS = [
    "Write a function that returns the n-th Fibonacci number.",
    "Now write one that returns the full sequence up to n.",
    "Add memoization to the first function.",
]


async def demo_anthropic():
    """Anthropic prompt caching demo.

    Anthropic requires explicit cache_control markers. chak injects them
    automatically when ``Cache(system_prompt=True)`` is set.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set, skipping.")
        return

    print("=" * 60)
    print("  Anthropic Prompt Caching Demo")
    print("=" * 60)

    # Anthropic: cache_control is injected on the system prompt block.
    # ttl defaults to 300 seconds (5 minutes); pass ttl=3600 for 1 hour.
    conv = chak.Conversation(
        "anthropic/claude-sonnet-4-6",
        api_key=api_key,
        system_prompt=LONG_SYSTEM_PROMPT,
        cache=chak.Cache(system_prompt=True, ttl=300),
    )

    for i, question in enumerate(QUESTIONS, 1):
        print(f"Turn {i}: {question}\n")
        response = await conv.asend(question)
        print(f"  Answer (truncated): {response.content[:120]}...\n")
        print_usage(f"Turn {i}", response)

    print_anthropic_notes()


async def demo_openai():
    """OpenAI prompt caching demo.

    OpenAI caching is automatic for prompts >= 1024 tokens. The ``key``
    parameter improves hit rates by routing similar requests to the same
    cache. On GPT-5.6+, ``system_prompt=True`` adds an explicit breakpoint.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping.")
        return

    print("=" * 60)
    print("  OpenAI Prompt Caching Demo")
    print("=" * 60)

    # OpenAI: prompt_cache_key improves hit rate; system_prompt adds an
    # explicit breakpoint on GPT-5.6+ only (silently skipped on older
    # models, since caching is automatic anyway). Even without any cache
    # config, OpenAI caches automatically — these just make it more reliable.
    conv = chak.Conversation(
        "openai/gpt-5",
        api_key=api_key,
        system_prompt=LONG_SYSTEM_PROMPT,
        cache=chak.Cache(
            system_prompt=True,  # No-op on gpt-4.1; would inject a breakpoint on GPT-5.6+
            key="chak-demo:python-assistant-v1",
        ),
    )

    for i, question in enumerate(QUESTIONS, 1):
        print(f"Turn {i}: {question}\n")
        response = await conv.asend(question)
        print(f"  Answer (truncated): {response.content[:120]}...\n")
        print_usage(f"Turn {i}", response)

    print_openai_notes()


def print_anthropic_notes():
    print("=" * 60)
    print("  How to read Anthropic cache results")
    print("=" * 60)
    print("  Turn 1: cache_creation_tokens > 0  -> prefix written to cache")
    print("  Turn 2+: cache_read_tokens > 0     -> prefix served from cache")
    print("  (cache_read tokens cost ~10x less than prompt_tokens)")
    print()


def print_openai_notes():
    print("=" * 60)
    print("  How to read OpenAI cache results")
    print("=" * 60)
    print("  OpenAI caching is automatic for prompts >= 1024 tokens.")
    print("  cache_read_tokens > 0 means the prefix was served from cache.")
    print("  The prompt_cache_key helps route requests to the same cache.")
    print()


async def main():
    provider = sys.argv[1] if len(sys.argv) > 1 else "anthropic"

    if provider == "anthropic":
        await demo_anthropic()
    elif provider == "openai":
        await demo_openai()
    elif provider == "all":
        await demo_anthropic()
        print("\n")
        await demo_openai()
    else:
        print(f"Unknown provider: {provider}")
        print("Usage: python examples/chat_cache.py [anthropic|openai|all]")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
