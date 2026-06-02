"""
Resilient Provider Failover Example

This example demonstrates request-level failover for temporary provider outages.
It intentionally points the primary Claude model and the first OpenAI fallback
model to unreachable local endpoints, then falls back to a working DeepSeek
model.

Important:
    Wrong API keys, invalid request parameters, and missing models are treated
    as configuration errors and should not trigger fallback by default.

Prerequisites:
    1. Set ANTHROPIC_API_KEY for the primary Claude provider.
    2. Set OPENAI_API_KEY for the first fallback provider.
    3. Set DEEPSEEK_API_KEY for the final fallback provider.

Usage:
    python examples/resilient_provider_failover.py
"""

import os

import dotenv

import chak


dotenv.load_dotenv()

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

missing_keys = [
    name
    for name, value in {
        "ANTHROPIC_API_KEY": anthropic_api_key,
        "OPENAI_API_KEY": openai_api_key,
        "DEEPSEEK_API_KEY": deepseek_api_key,
    }.items()
    if not value
]
if missing_keys:
    print(f"Error: Please set required environment variables: {', '.join(missing_keys)}")
    exit(1)

conv = chak.Conversation(
    "anthropic@http://127.0.0.1:9:claude-haiku-4-5",
    api_key=anthropic_api_key,
    timeout=2,
    fallbacks=[
        {
            "model_uri": "openai@http://127.0.0.1:9/v1:gpt-4o-mini",
            "api_key": openai_api_key,
            "timeout": 2,
        },
        {
            "model_uri": "deepseek/deepseek-v4-flash",
            "api_key": deepseek_api_key,
            "timeout": 30,
        },
    ],
)

print("=" * 70)
print("Non-streaming failover example")
print("=" * 70)

response = conv.send("Explain LLM provider failover in one sentence.")
print(f"Response: {response.content}")
print(f"Trace: {response.metadata.provider_trace}")

print("\n" + "=" * 70)
print("Streaming failover example")
print("=" * 70)

stream = conv.send("Write one short sentence about reliable AI services.", stream=True)
for chunk in stream:  # type: ignore
    if isinstance(chunk, chak.FailoverChunk):
        print(
            f"\n[failover] {chunk.failed_provider} failed, "
            f"retrying with {chunk.next_provider}: {chunk.error}\n"
        )
        continue

    if chunk.content:  # type: ignore
        print(chunk.content, end="", flush=True)  # type: ignore

print("\n")

print("=" * 70)
print("Provider trace summary (via Conversation.get_provider_traces)")
print("=" * 70)

traces = conv.get_provider_traces()
for i, trace in enumerate(traces):
    print(f"\nMessage {i + 1}:")
    print(f"  Primary:   {trace.primary_provider}/{trace.primary_model}")
    print(f"  Fallback:  {trace.fallback_used}")
    print(f"  Attempts:  {trace.failover_attempts}")
    print(f"  Resolved:  {trace.resolved_provider}/{trace.resolved_model}")
    if trace.failed_providers:
        print(f"  Failures:")
        for f in trace.failed_providers:
            print(f"    [{f.attempt_index}] {f.provider}/{f.model}: {f.error_type} ({f.error})")
