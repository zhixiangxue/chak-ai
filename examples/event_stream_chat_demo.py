"""
Event Stream Chat Demo

Demonstrates how to use event=True parameter to observe tool calls in real-time.
This is useful for building UIs that show tool execution progress.

Prerequisites:
    1. Set DASHSCOPE_API_KEY environment variable
    2. Python 3.10+ (for match-case syntax)
    
Usage:
    python examples/event_stream_chat_demo.py
"""

import asyncio
import os

import dotenv

dotenv.load_dotenv()

import chak
from chak.message import MessageChunk, ToolCallStartEvent, ToolCallSuccessEvent, ToolCallErrorEvent


# ============================================================================
# 🔑 IMPORTANT: Set your API key 🔑
# ============================================================================
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    print("❌ Error: Please set DASHSCOPE_API_KEY environment variable")
    print("   Example: export DASHSCOPE_API_KEY=sk-your-key-here")
    exit(1)


# ============================================================================
# Define tools
# ============================================================================
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def broken_tool(x: int) -> int:
    """A tool that always fails - demonstrates error handling."""
    raise ValueError("This tool is intentionally broken!")


# ============================================================================
# Main demo
# ============================================================================
async def main():
    # Create conversation with tools
    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        tools=[add, multiply, broken_tool],
        system_message="You are a helpful math assistant. Use tools to calculate."
    )
    
    print("=" * 70)
    print("Event Stream Demo - Tool Call Observability")
    print("=" * 70)
    print()
    
    # Ask a question that requires tool usage
    question = "What is (15 + 27) * 3? Also try calling broken_tool with x=5."
    print(f"👤 User: {question}")
    print()
    
    # Use event=True to get real-time tool call information
    tool_start_times = {}  # Track tool call start times
    async for event in await conv.asend(question, event=True):
        match event:
            case MessageChunk(content=text, is_final=final, metadata=meta, final_message=msg):
                # LLM content output
                # text = chunk.content (the generated text chunk)
                # final = chunk.is_final (whether this is the last chunk)
                # meta = chunk.metadata (dict with usage info, etc.)
                # msg = chunk.final_message (complete AIMessage when is_final=True)
                if text:
                    print(text, end="", flush=True)
                if final:
                    print("\n")
                    print(f"📊 Metadata: {meta}")
                    print(f"📝 Final message type: {type(msg).__name__}")
            
            case ToolCallStartEvent(tool_name=name, call_id=call_id, arguments=args, timestamp=ts):
                # Tool call started
                # name = event.tool_name (e.g., "add", "multiply")
                # call_id = event.call_id (unique identifier for this tool call)
                # args = event.arguments (e.g., {"a": 15, "b": 27})
                # ts = event.timestamp (start time in seconds)
                tool_start_times[call_id] = ts
                print(f"\n🔧 Calling tool: {name} (call_id: {call_id})")
                print(f"   Arguments: {args}")
            
            case ToolCallSuccessEvent(tool_name=name, call_id=call_id, result=res, timestamp=ts):
                # Tool call succeeded
                # name = event.tool_name
                # call_id = event.call_id (matches the call_id from ToolCallStartEvent)
                # res = event.result (the tool's return value as string)
                # ts = event.timestamp (end time in seconds)
                duration = ts - tool_start_times.get(call_id, ts)
                print(f"✅ Tool result: {name} (call_id: {call_id}) -> {res}")
                print(f"   ⏱️  Duration: {duration:.3f}s")
                print()
            
            case ToolCallErrorEvent(tool_name=name, call_id=call_id, error=err, timestamp=ts):
                # Tool call failed
                # name = event.tool_name
                # call_id = event.call_id
                # err = event.error (error message)
                # ts = event.timestamp (error time in seconds)
                duration = ts - tool_start_times.get(call_id, ts)
                print(f"❌ Tool failed: {name} (call_id: {call_id})")
                print(f"   Error: {err}")
                print(f"   ⏱️  Duration: {duration:.3f}s")
                print()
    
    print("=" * 70)
    print("Conversation completed!")
    print(f"Total messages: {len(conv.messages)}")
    print(f"Stats: {conv.stats()}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
