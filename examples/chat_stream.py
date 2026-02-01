"""
Streaming Chat - Bailian (Alibaba Cloud Qwen)

This example demonstrates streaming responses in both sync and async modes.

Prerequisites:
    1. Get your Bailian API key from: https://bailian.console.aliyun.com
    2. Set environment variable: export BAILIAN_API_KEY=your_key_here

Features:
    - Streaming output (word-by-word)
    - Both sync and async versions
    - Real-time response display

Usage:
    python examples/chat_stream_bailian.py
"""

import asyncio
import os

import dotenv

import chak

dotenv.load_dotenv()


def example_sync_stream():
    """Sync streaming example."""
    api_key = os.getenv("BAILIAN_API_KEY", "")
    if not api_key:
        print("❌ Error: Please set BAILIAN_API_KEY environment variable")
        return

    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        system_prompt="你是一个友好的助手。",
    )

    print("\n=== Sync Streaming Example ===")
    print("Question: 用一句话介绍Python的主要特点。\n")
    print("Answer: ", end="", flush=True)
    
    for chunk in conv.send(
        "用一句话介绍Python的主要特点。",
        stream=True,
        timeout=60
    ):
        if isinstance(chunk, chak.MessageChunk):
            print(chunk.content, end="", flush=True)
    
    print("\n")


async def example_async_stream():
    """Async streaming example."""
    api_key = os.getenv("BAILIAN_API_KEY", "")
    if not api_key:
        print("❌ Error: Please set BAILIAN_API_KEY environment variable")
        return

    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        system_prompt="你是一个友好的助手。",
    )

    print("\n=== Async Streaming Example ===")
    print("Question: 用一句话介绍JavaScript的主要特点。\n")
    print("Answer: ", end="", flush=True)
    
    async for chunk in await conv.asend(
        "用一句话介绍JavaScript的主要特点。",
        stream=True,
        timeout=60
    ):
        if isinstance(chunk, chak.MessageChunk):
            print(chunk.content, end="", flush=True)
    
    print("\n")


async def main_async():
    """Run async examples."""
    await example_async_stream()


def main():
    print("\n" + "="*60)
    print("Streaming Chat Examples - Bailian (Qwen)")
    print("="*60)
    
    # Sync example
    example_sync_stream()
    
    # Async example
    asyncio.run(main_async())
    
    print("="*60)
    print("✓ All examples completed")
    print("="*60)


if __name__ == "__main__":
    main()
