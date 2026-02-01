"""
Reasoning Chat - Bailian (Alibaba Cloud QwQ)

This example demonstrates reasoning mode with Bailian's QwQ model.
Reasoning mode allows the model to "think" before answering.

Prerequisites:
    1. Get your Bailian API key from: https://bailian.console.aliyun.com
    2. Set environment variable: export BAILIAN_API_KEY=your_key_here

Features:
    - Reasoning mode (reasoning=chak.Reasoning(...))
    - Both streaming and non-streaming
    - Both sync and async versions
    - ReasoningChunk vs MessageChunk

Usage:
    python examples/chat_reasoning_bailian.py
"""

import asyncio
import os

import dotenv

import chak

dotenv.load_dotenv()


def example_sync_nonstream():
    """Sync non-streaming reasoning example."""
    api_key = os.getenv("BAILIAN_API_KEY", "")
    if not api_key:
        print("❌ Error: Please set BAILIAN_API_KEY environment variable")
        return

    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        system_prompt="你是一个会先认真思考再给出结论的助手。",
    )

    print("\n=== Sync Non-Streaming Reasoning ===")
    print("Question: 请分析 1 到 10 的整数中有多少个质数。\n")
    
    response = conv.send(
        "请分析 1 到 10 的整数中有多少个质数。",
        reasoning=chak.Reasoning(effort="medium"),
        timeout=120
    )

    if hasattr(response, 'reasoning_content') and response.reasoning_content:
        print(f"[Reasoning Process]\n{response.reasoning_content}\n")
    
    print(f"[Final Answer]\n{response.content}\n")


def example_sync_stream():
    """Sync streaming reasoning example."""
    api_key = os.getenv("BAILIAN_API_KEY", "")
    if not api_key:
        print("❌ Error: Please set BAILIAN_API_KEY environment variable")
        return

    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        system_prompt="你是一个会先认真思考再给出结论的助手。",
    )

    print("\n=== Sync Streaming Reasoning ===")
    print("Question: 请分析 1 到 10 的整数中有多少个质数。\n")
    
    print("[Streaming Output]\n")
    for chunk in conv.send(
        "请分析 1 到 10 的整数中有多少个质数。",
        reasoning=chak.Reasoning(effort="medium"),
        stream=True,
        timeout=120
    ):
        if isinstance(chunk, chak.ReasoningChunk):
            print(f"[Thinking: {chunk.content}]", flush=True)
        elif isinstance(chunk, chak.MessageChunk):
            print(chunk.content, end="", flush=True)
    
    print("\n")


async def example_async_nonstream():
    """Async non-streaming reasoning example."""
    api_key = os.getenv("BAILIAN_API_KEY", "")
    if not api_key:
        print("❌ Error: Please set BAILIAN_API_KEY environment variable")
        return

    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        system_prompt="你是一个会先认真思考再给出结论的助手。",
    )

    print("\n=== Async Non-Streaming Reasoning ===")
    print("Question: 请分析 1 到 10 的整数中有多少个质数。\n")
    
    response = await conv.asend(
        "请分析 1 到 10 的整数中有多少个质数。",
        reasoning=chak.Reasoning(effort="medium"),
        timeout=120
    )

    if hasattr(response, 'reasoning_content') and response.reasoning_content:
        print(f"[Reasoning Process]\n{response.reasoning_content}\n")
    
    print(f"[Final Answer]\n{response.content}\n")


async def example_async_stream():
    """Async streaming reasoning example."""
    api_key = os.getenv("BAILIAN_API_KEY", "")
    if not api_key:
        print("❌ Error: Please set BAILIAN_API_KEY environment variable")
        return

    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        system_prompt="你是一个会先认真思考再给出结论的助手。",
    )

    print("\n=== Async Streaming Reasoning ===")
    print("Question: 请分析 1 到 10 的整数中有多少个质数。\n")
    
    print("[Streaming Output]\n")
    async for chunk in await conv.asend(
        "请分析 1 到 10 的整数中有多少个质数。",
        reasoning=chak.Reasoning(effort="medium"),
        stream=True,
        timeout=120
    ):
        if isinstance(chunk, chak.ReasoningChunk):
            print(f"[Thinking: {chunk.content}]", flush=True)
        elif isinstance(chunk, chak.MessageChunk):
            print(chunk.content, end="", flush=True)
    
    print("\n")


async def main_async():
    """Run async examples."""
    await example_async_nonstream()
    await example_async_stream()


def main():
    print("\n" + "="*60)
    print("Reasoning Chat Examples - Bailian (QwQ)")
    print("="*60)
    
    # Sync examples
    example_sync_nonstream()
    example_sync_stream()
    
    # Async examples
    asyncio.run(main_async())
    
    print("="*60)
    print("✓ All examples completed")
    print("="*60)


if __name__ == "__main__":
    main()
