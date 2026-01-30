"""
Reasoning Chat - Custom Providers

This example demonstrates how to use reasoning mode with different providers.
You can test with Bailian (Alibaba Cloud) or OpenAI models.

Prerequisites:
    1. Get your Bailian API key from: https://bailian.console.aliyun.com
    2. Set environment variable: export BAILIAN_API_KEY=your_key_here

Note:
    - Whether `reasoning_content` is actually returned depends on the
      underlying model and provider behavior.
    - For models that do not expose thinking content, `reasoning_content`
      will be None or an empty string.

Usage:
    # Test with Bailian
    export BAILIAN_API_KEY=your_key
    python examples/reasoning_chat_custom.py --provider bailian
    
    # Test with OpenAI (if you have o1 access)
    export OPENAI_API_KEY=your_key
    python examples/reasoning_chat_custom.py --provider openai
"""

import argparse
import os

import dotenv

import chak

dotenv.load_dotenv()


def example_bailian():
    """Example using Bailian (Alibaba Cloud) Qwen model."""
    api_key = os.getenv("BAILIAN_API_KEY", "")
    if not api_key:
        print("❌ Error: Please set BAILIAN_API_KEY environment variable")
        return

    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        system_message="你是一个会先认真思考再给出结论的助手。",
    )

    print("\n=== Testing Bailian (qwen-plus) ===")
    print("Sending message with reasoning mode...")
    response = conv.send(
        "请详细分析 1 到 10 的整数中，有多少个是质数，并给出结论。",
        reasoning={"effort": "medium"},
        timeout=120
    )

    print("\n=== Thinking (reasoning_content) ===")
    if getattr(response, "reasoning_content", None):
        print(response.reasoning_content)
    else:
        print("[No reasoning_content returned]")

    print("\n=== Final Answer (content) ===")
    print(response.content)


def example_openai():
    """Example using OpenAI reasoning model."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("❌ Error: Please set OPENAI_API_KEY environment variable")
        return

    conv = chak.Conversation(
        "openai/gpt-5.1",
        api_key=api_key,
        system_message="You are a helpful assistant that thinks carefully before answering.",
    )

    print("\n=== Testing OpenAI ===")
    print("Sending message with reasoning mode...")
    response = conv.send(
        "Analyze how many prime numbers are there between 1 and 10.",
        reasoning={"effort": "high", "summary": "auto"},
        timeout=120
    )

    print("\n=== Thinking (reasoning_content) ===")
    if getattr(response, "reasoning_content", None):
        print(response.reasoning_content)
    else:
        print("[No reasoning_content returned]")

    print("\n=== Final Answer (content) ===")
    print(response.content)

def main():
    print("\n=== Running all reasoning provider examples ===")

    print("\n--- Bailian example ---")
    example_bailian()

    print("\n--- OpenAI example ---")
    example_openai()


if __name__ == "__main__":
    main()
