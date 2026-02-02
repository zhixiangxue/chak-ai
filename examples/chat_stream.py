"""
Streaming Chat - Universal Provider Test

This example demonstrates streaming responses with any provider.

Prerequisites:
    Set environment variables based on provider:
    - BAILIAN_API_KEY for bailian/*
    - OPENAI_API_KEY for openai/*
    - etc.

Features:
    - Streaming output (word-by-word)
    - Both sync and async versions
    - Command-line provider selection

Usage:
    python examples/chat_stream.py bailian/qwen-plus
    python examples/chat_stream.py openai/gpt-4o-mini
"""

import argparse
import asyncio
import os

import dotenv

import chak

dotenv.load_dotenv()


def get_api_key(model_uri: str) -> str:
    """Get API key based on provider name."""
    provider = model_uri.split("/")[0].upper()
    key = os.getenv(f"{provider}_API_KEY", "")
    if not key:
        raise ValueError(f"Please set {provider}_API_KEY environment variable")
    return key


def example_sync_stream(model_uri: str):
    """Sync streaming example."""
    api_key = get_api_key(model_uri)

    conv = chak.Conversation(
        model_uri,
        api_key=api_key,
        system_prompt="你是一个友好的助手。",
    )

    print("\n=== Sync Streaming Example ===")
    print(f"Model: {model_uri}")
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


async def example_async_stream(model_uri: str):
    """Async streaming example."""
    api_key = get_api_key(model_uri)

    conv = chak.Conversation(
        model_uri,
        api_key=api_key,
        system_prompt="你是一个友好的助手。",
    )

    print("\n=== Async Streaming Example ===")
    print(f"Model: {model_uri}")
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


async def main_async(model_uri: str):
    """Run async examples."""
    await example_async_stream(model_uri)


def main():
    parser = argparse.ArgumentParser(description="Streaming chat example")
    parser.add_argument(
        "model_uri",
        nargs="?",
        default="bailian/qwen-plus",
        help="Model URI (e.g., bailian/qwen-plus, openai/gpt-4o-mini)",
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(f"Streaming Chat Example - {args.model_uri}")
    print("="*60)
    
    # Sync example
    example_sync_stream(args.model_uri)
    
    # Async example
    asyncio.run(main_async(args.model_uri))
    
    print("="*60)
    print("✓ All examples completed")
    print("="*60)


if __name__ == "__main__":
    main()
