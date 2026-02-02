"""
Event Stream Chat - Universal Provider Test

This example demonstrates event stream mode for full observability.
Event mode provides MessageChunk, ToolCall, and ToolResult events in real-time.

Prerequisites:
    Set environment variables based on provider:
    - BAILIAN_API_KEY for bailian/*
    - OPENAI_API_KEY for openai/*
    - etc.

Features:
    - Event stream mode (event=True)
    - Full observability: MessageChunk + ToolCall + ToolResult events
    - Both sync and async versions
    - Command-line provider selection

Usage:
    python examples/chat_event.py openai/gpt-4o
    python examples/chat_event.py bailian/qwen-plus
"""

import argparse
import asyncio
import os
from datetime import datetime

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


# Define tool functions
def get_current_time(timezone: str = "UTC") -> str:
    """Get current time in specified timezone.
    
    Args:
        timezone: Timezone name (e.g., 'UTC', 'Asia/Shanghai')
    
    Returns:
        Current time string
    """
    now = datetime.now()
    return f"Current time in {timezone}: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def get_weather(city: str) -> str:
    """Get weather information for a city.
    
    Args:
        city: City name
    
    Returns:
        Weather information
    """
    # Mock weather data
    weather_data = {
        "北京": "晴天，温度 15°C",
        "上海": "多云，温度 18°C",
        "深圳": "小雨，温度 22°C",
    }
    return weather_data.get(city, f"{city}的天气：晴天，温度 20°C")


def example_sync_event(model_uri: str):
    """Sync streaming example (event mode requires async - use stream=True instead)."""
    api_key = get_api_key(model_uri)

    conv = chak.Conversation(
        model_uri,
        api_key=api_key,
        system_prompt="你是一个友好的助手。",
    )

    print("\n=== Sync Streaming Example ===")
    print(f"Model: {model_uri}")
    print("Note: Event mode requires async. This shows regular streaming.\n")
    print("Question: 介绍一下北京的三个著名景点。\n")
    print("Answer: ", end="", flush=True)
    
    for chunk in conv.send(
        "介绍一下北京的三个著名景点。",
        stream=True,
        timeout=60
    ):
        if isinstance(chunk, chak.MessageChunk):
            if chunk.content:
                print(chunk.content, end="", flush=True)
    
    print("\n")


async def example_async_event(model_uri: str):
    """Async event stream example with tool calling."""
    api_key = get_api_key(model_uri)

    # First test: without event mode
    print("\n=== Test 1: Without event mode ===")
    print(f"Model: {model_uri}")
    conv1 = chak.Conversation(
        model_uri,
        api_key=api_key,
        system_prompt="You are a helpful assistant.",
        tools=[get_current_time, get_weather],
    )
    
    response = await conv1.asend(
        "What's the current time in UTC? Use the get_current_time function.",
        timeout=60
    )
    print(f"Response: {response.content}\n")
    
    # Second test: with event mode (new conversation)
    print("\n=== Test 2: With event mode (fresh conversation) ===")
    print(f"Model: {model_uri}")
    conv2 = chak.Conversation(
        model_uri,
        api_key=api_key,
        system_prompt="You are a helpful assistant.",
        tools=[get_current_time, get_weather],
    )
    
    print("Events:\n")
    async for event in await conv2.asend(
        "Call get_current_time to get UTC time, then call get_weather for Beijing. You MUST use both tools.",
        event=True,
        timeout=60
    ):
        if isinstance(event, chak.MessageChunk):
            if event.content:
                print(f"💬 [MessageChunk] {event.content}", end="", flush=True)
            if event.is_final:
                print("\n✅ [MessageChunk] Complete")
        elif isinstance(event, chak.ToolCallStartEvent):
            print(f"\n🔧 [ToolCallStart] {event.tool_name}(call_id={event.call_id})")
            print(f"   Arguments: {event.arguments}")
        elif isinstance(event, chak.ToolCallSuccessEvent):
            print(f"✅ [ToolCallSuccess] {event.tool_name}(call_id={event.call_id})")
            print(f"   Result: {event.result}")
        elif isinstance(event, chak.ToolCallErrorEvent):
            print(f"❌ [ToolCallError] {event.tool_name}(call_id={event.call_id})")
            print(f"   Error: {event.error}")
        else:
            print(f"\n📋 [Event] {type(event).__name__}: {event}")
    
    print("\n")


async def main_async(model_uri: str):
    """Run async examples."""
    await example_async_event(model_uri)


def main():
    parser = argparse.ArgumentParser(description="Event stream chat example")
    parser.add_argument(
        "model_uri",
        nargs="?",
        default="openai/gpt-4o",
        help="Model URI (e.g., bailian/qwen-plus, openai/gpt-4o)",
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(f"Event Stream Chat Example - {args.model_uri}")
    print("With Tool Calling for Full Observability")
    print("="*60)
    
    # Sync example
    example_sync_event(args.model_uri)
    
    # Async example
    asyncio.run(main_async(args.model_uri))
    
    print("="*60)
    print("✓ All examples completed")
    print("="*60)


if __name__ == "__main__":
    main()
