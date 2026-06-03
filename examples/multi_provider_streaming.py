"""
Multi-Provider Streaming Verification

Verify that all core providers work correctly with streaming output.
Providers tested: OpenAI, DeepSeek, Anthropic, MiniMax, Bailian (Qwen).

Prerequisites:
    Set the following environment variables (or place them in .env):
        OPENAI_API_KEY
        DEEPSEEK_API_KEY
        ANTHROPIC_API_KEY
        MINIMAX_API_KEY
        BAILIAN_API_KEY

Usage:
    python examples/multi_provider_streaming.py
    python examples/multi_provider_streaming.py --provider openai deepseek
"""

import argparse
import os
import sys
import time

import dotenv

dotenv.load_dotenv()

import chak

PROVIDERS = {
    "openai": {
        "model_uri": "openai/gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "model_uri": "deepseek/deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "anthropic": {
        "model_uri": "anthropic/claude-haiku-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "minimax": {
        "model_uri": "minimax@https://api.minimax.io/anthropic:MiniMax-M3",
        "api_key_env": "MINIMAX_API_KEY",
    },
    "bailian": {
        "model_uri": "bailian/qwen-plus",
        "api_key_env": "BAILIAN_API_KEY",
    },
}

PROMPT = "Write exactly one sentence explaining what streaming means in LLM APIs."


def run_provider(name: str, config: dict) -> bool:
    """Run streaming test for a single provider. Returns True if successful."""
    api_key = os.getenv(config["api_key_env"], "")
    if not api_key:
        print(f"  SKIP - {config['api_key_env']} not set")
        return False

    print(f"  Model: {config['model_uri']}")
    print(f"  Response: ", end="", flush=True)

    try:
        conv = chak.Conversation(config["model_uri"], api_key=api_key, timeout=60)
        start = time.time()
        chunks = []

        stream = conv.send(PROMPT, stream=True, timeout=60)
        for chunk in stream:
            if isinstance(chunk, chak.MessageChunk) and chunk.content:
                print(chunk.content, end="", flush=True)
                chunks.append(chunk.content)

        elapsed = time.time() - start
        content = "".join(chunks)
        print()

        # Validate
        if not content.strip():
            print(f"  FAIL - Empty response")
            return False

        # Show stats
        stats = conv.stats()
        tokens = stats.get("total_tokens", "N/A")
        print(f"  OK - {len(content)} chars, {tokens} tokens, {elapsed:.2f}s")
        return True

    except Exception as e:
        print()
        print(f"  FAIL - {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Multi-provider streaming verification")
    parser.add_argument(
        "--provider", "-p",
        nargs="+",
        choices=list(PROVIDERS.keys()),
        default=list(PROVIDERS.keys()),
        help="Providers to test (default: all)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Multi-Provider Streaming Verification")
    print("=" * 70)

    results = {}
    for name in args.provider:
        config = PROVIDERS[name]
        print(f"\n[{name.upper()}]")
        results[name] = run_provider(name, config)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    skipped = sum(1 for name in args.provider if not os.getenv(PROVIDERS[name]["api_key_env"]))
    failed = len(results) - passed - skipped

    for name, ok in results.items():
        status = "PASS" if ok else ("SKIP" if not os.getenv(PROVIDERS[name]["api_key_env"]) else "FAIL")
        print(f"  {name:<12} {status}")

    print(f"\n  {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
