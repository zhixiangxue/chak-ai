"""
Reasoning Chat - Universal Provider Test

This example demonstrates reasoning mode with any provider.
Reasoning mode allows the model to "think" before answering.

Prerequisites:
    Set environment variables based on provider:
    - BAILIAN_API_KEY for bailian/*
    - OPENAI_API_KEY for openai/*
    - etc.

Features:
    - Reasoning mode (reasoning=chak.Reasoning(...))
    - Both streaming and non-streaming
    - Both sync and async versions
    - Command-line provider selection

Usage:
    python examples/chat_reasoning.py openai/gpt-4o-mini
    python examples/chat_reasoning.py bailian/qwen-plus
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


def example_sync_nonstream(model_uri: str):
    """Sync non-streaming reasoning example."""
    api_key = get_api_key(model_uri)

    conv = chak.Conversation(
        model_uri,
        api_key=api_key,
        system_prompt="你是一个会先认真思考再给出结论的助手。",
    )

    print("\n=== Sync Non-Streaming Reasoning ===")
    print(f"Model: {model_uri}")
    print("Question: 请分析 1 到 10 的整数中有多少个质数。\n")
    
    response = conv.send(
        "请分析 1 到 10 的整数中有多少个质数。",
        reasoning=chak.Reasoning(effort="high", summary="auto"),
        timeout=120
    )

    if hasattr(response, 'reasoning_content') and response.reasoning_content:
        print(f"[Reasoning Process]\n{response.reasoning_content}\n")
    
    print(f"[Final Answer]\n{response.content}\n")


def example_sync_stream(model_uri: str):
    """Sync streaming reasoning example."""
    api_key = get_api_key(model_uri)

    conv = chak.Conversation(
        model_uri,
        api_key=api_key,
        system_prompt="你是一个会先认真思考再给出结论的助手。",
    )

    print("\n=== Sync Streaming Reasoning ===")
    print(f"Model: {model_uri}")
    print("Question: 请分析 1 到 10 的整数中有多少个质数。\n")
    
    print("[Streaming Output]\n")
    for chunk in conv.send(
        "请分析 1 到 10 的整数中有多少个质数。",
        reasoning=chak.Reasoning(effort="high", summary="auto"),
        stream=True,
        timeout=120
    ):
        if isinstance(chunk, chak.ReasoningChunk):
            print(f"[Thinking: {chunk.content}]", flush=True)
        elif isinstance(chunk, chak.MessageChunk):
            print(chunk.content, end="", flush=True)
    
    print("\n")


async def example_async_nonstream(model_uri: str):
    """Async non-streaming reasoning example."""
    api_key = get_api_key(model_uri)

    conv = chak.Conversation(
        model_uri,
        api_key=api_key,
        system_prompt="你是一个会先认真思考再给出结论的助手。",
    )

    print("\n=== Async Non-Streaming Reasoning ===")
    print(f"Model: {model_uri}")
    print("Question: 请分析 1 到 10 的整数中有多少个质数。\n")
    
    response = await conv.asend(
        "请分析 1 到 10 的整数中有多少个质数。",
        reasoning=chak.Reasoning(effort="high", summary="auto"),
        timeout=120
    )

    if hasattr(response, 'reasoning_content') and response.reasoning_content:
        print(f"[Reasoning Process]\n{response.reasoning_content}\n")
    
    print(f"[Final Answer]\n{response.content}\n")


async def example_async_stream(model_uri: str):
    """Async streaming reasoning example."""
    api_key = get_api_key(model_uri)

    conv = chak.Conversation(
        model_uri,
        api_key=api_key,
        system_prompt="你是一个会先认真思考再给出结论的助手。",
    )

    print("\n=== Async Streaming Reasoning ===")
    print(f"Model: {model_uri}")
    print("Question: 请分析 1 到 10 的整数中有多少个质数。\n")
    
    print("[Streaming Output]\n")
    async for chunk in await conv.asend(
        "请分析 1 到 10 的整数中有多少个质数。",
        reasoning=chak.Reasoning(effort="high", summary="auto"),
        stream=True,
        timeout=120
    ):
        if isinstance(chunk, chak.ReasoningChunk):
            print(f"[Thinking: {chunk.content}]", flush=True)
        elif isinstance(chunk, chak.MessageChunk):
            print(chunk.content, end="", flush=True)
    
    print("\n")


async def main_async(model_uri: str):
    """Run async examples."""
    await example_async_nonstream(model_uri)
    await example_async_stream(model_uri)


def main():
    parser = argparse.ArgumentParser(description="Reasoning chat example")
    parser.add_argument(
        "model_uri",
        nargs="?",
        default="openai/gpt-5",
        help="Model URI (e.g., bailian/qwen-plus, openai/gpt-5)",
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(f"Reasoning Chat Example - {args.model_uri}")
    print("="*60)
    
    # Sync examples
    example_sync_nonstream(args.model_uri)
    example_sync_stream(args.model_uri)
    
    # Async examples
    asyncio.run(main_async(args.model_uri))
    
    print("="*60)
    print("✓ All examples completed")
    print("="*60)


if __name__ == "__main__":
    main()
