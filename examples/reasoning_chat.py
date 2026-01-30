"""
Reasoning Chat - Providers with Reasoning Support

This example demonstrates how to use reasoning mode with different providers,
including both non-streaming and streaming modes.

Prerequisites:
    1. Get your Bailian API key from: https://bailian.console.aliyun.com
    2. Set environment variable: export BAILIAN_API_KEY=your_key_here

Note:
    - Whether `reasoning_content` is actually returned depends on the
      underlying model and provider behavior.
    - For models that do not expose thinking content, `reasoning_content`
      will be None or an empty string.

Usage:
    # Run all examples
    python examples/reasoning_chat.py
"""

import os

import dotenv

import chak

dotenv.load_dotenv()


def example_bailian():
    """Example using Bailian (Alibaba Cloud) Qwen model - Non-streaming."""
    api_key = os.getenv("BAILIAN_API_KEY", "")
    if not api_key:
        print("❌ Error: Please set BAILIAN_API_KEY environment variable")
        return

    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        system_message="你是一个会先认真思考再给出结论的助手。",
    )

    print("\n=== Testing Bailian (qwen-plus) - Non-streaming ===")
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


def example_bailian_stream():
    """Example using Bailian (Alibaba Cloud) Qwen model - Streaming."""
    api_key = os.getenv("BAILIAN_API_KEY", "")
    if not api_key:
        print("❌ Error: Please set BAILIAN_API_KEY environment variable")
        return

    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        system_message="你是一个会先认真思考再给出结论的助手。",
    )

    print("\n=== Testing Bailian (qwen-plus) - Streaming ===")
    print("Sending message with reasoning mode...\n")
    
    print("--- Answer Stream ---")
    for chunk in conv.send(
        "请分析 1 到 10 的整数中有多少个质数。",
        reasoning={"effort": "medium"},
        stream=True,
        timeout=120
    ):
        if isinstance(chunk, chak.MessageChunk):
            print(chunk.content, end="", flush=True)
        elif isinstance(chunk, chak.ReasoningChunk):
            print(f"\n[Thinking: {chunk.content}]", flush=True)
    
    print("\n")


def example_openai():
    """Example using OpenAI reasoning model - Non-streaming."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("❌ Error: Please set OPENAI_API_KEY environment variable")
        return

    conv = chak.Conversation(
        "openai/gpt-5",
        api_key=api_key,
        system_message="You are a helpful assistant that thinks carefully before answering.",
    )

    print("\n=== Testing OpenAI - Non-streaming ===")
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


def example_openai_stream():
    """Example using OpenAI reasoning model - Streaming."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("❌ Error: Please set OPENAI_API_KEY environment variable")
        return

    conv = chak.Conversation(
        "openai/gpt-5",
        api_key=api_key,
        system_message="You are a helpful assistant that thinks carefully before answering.",
    )

    print("\n=== Testing OpenAI - Streaming ===")
    print("Sending message with reasoning mode...\n")
    
    print("--- Answer Stream ---")
    for chunk in conv.send(
        "How many prime numbers are between 1 and 10?",
        reasoning={"effort": "high", "summary": "auto"},
        stream=True,
        timeout=120
    ):
        if isinstance(chunk, chak.MessageChunk):
            print(chunk.content, end="", flush=True)
        elif isinstance(chunk, chak.ReasoningChunk):
            print(f"\n[Thinking: {chunk.content}]", flush=True)
    
    print("\n")

def main():
    print("\n=== Running all reasoning provider examples ===")

    print("\n--- Bailian Non-streaming ---")
    # example_bailian()

    print("\n--- Bailian Streaming ---")
    example_bailian_stream()

    print("\n--- OpenAI Non-streaming ---")
    # example_openai()

    print("\n--- OpenAI Streaming ---")
    example_openai_stream()


if __name__ == "__main__":
    main()
