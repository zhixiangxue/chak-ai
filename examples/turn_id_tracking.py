"""Turn ID Tracking Example

This example demonstrates how chak automatically tracks turn_id for all messages
within a single send/asend call, including tool calls and assistant responses.

The turn_id field helps identify which messages belong to the same conversation turn,
making it easy to:
- Count the number of turns in a conversation
- Keep the last N turns
- Group messages by turn for analysis

Prerequisites:
    1. Bailian API key: https://bailian.console.aliyun.com
    2. Set environment variable: export BAILIAN_API_KEY=your_key_here

Usage:
    python examples/turn_id_tracking.py
"""

import os
import asyncio
from collections import defaultdict

import dotenv

dotenv.load_dotenv()

import chak


# ============================================================================
# Define some simple tools
# ============================================================================

def get_weather(city: str) -> str:
    """Get weather information for a city."""
    return f"Weather in {city}: Sunny, 25°C"


def get_time(timezone: str = "UTC") -> str:
    """Get current time in specified timezone."""
    return f"Current time in {timezone}: 14:30:00"


def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# Helper Functions
# ============================================================================

def print_separator(title=""):
    """Print a formatted separator line."""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    else:
        print(f"{'='*70}\n")


def analyze_turns(messages):
    """Analyze and display turn information."""
    # Group messages by turn_id
    turns = defaultdict(list)
    for msg in messages:
        if msg.turn_id:
            turns[msg.turn_id].append(msg)
    
    print(f"\n📊 Turn Analysis:")
    print(f"  Total turns: {len(turns)}")
    print(f"  Total messages: {len(messages)}")
    
    # Show details of each turn
    for i, (turn_id, msgs) in enumerate(turns.items(), 1):
        print(f"\n  Turn {i} (turn_id: {turn_id[:8]}...):")
        print(f"    Contains {len(msgs)} messages sharing the same turn_id:")
        
        for msg in msgs:
            role_emoji = {
                "user": "👤",
                "assistant": "🤖",
                "tool": "🔧"
            }.get(msg.role, "❓")
            
            # Get message type
            msg_type = type(msg).__name__
            
            content_preview = ""
            if msg.role == "tool":
                tool_call_id = getattr(msg, 'tool_call_id', 'unknown')
                content_preview = f"(tool_call_id: {tool_call_id[:8]}...)"
            elif isinstance(msg.content, str):
                content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            
            print(f"      {role_emoji} {msg_type} (id: {msg.id[:8]}...): {content_preview}")
            
            # Show tool calls if present
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(f"         └─ 🔧 Calling: {tool_call.function.name}()")


def get_last_n_turns(messages, n: int):
    """Get messages from the last N turns."""
    # Get unique turn IDs in order
    seen = set()
    turn_ids_in_order = []
    for msg in messages:
        if msg.turn_id and msg.turn_id not in seen:
            turn_ids_in_order.append(msg.turn_id)
            seen.add(msg.turn_id)
    
    # Get last N turn IDs
    last_turn_ids = set(turn_ids_in_order[-n:])
    
    # Filter messages
    return [msg for msg in messages if msg.turn_id in last_turn_ids]


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    # Get API key
    api_key = os.getenv("BAILIAN_API_KEY")
    if not api_key:
        print("⚠️  Error: BAILIAN_API_KEY not set")
        print("   Please set: export BAILIAN_API_KEY=your_key_here")
        return
    
    print_separator("Turn ID Tracking Demo")
    
    # Create tools - just pass functions directly
    tools = [get_weather, get_time, calculate]
    
    # Create conversation with tools
    conv = chak.Conversation(
        "bailian/qwen3-max",
        api_key=api_key,
        tools=tools
    )
    
    # ========================================================================
    # Turn 1: Simple message (no tools)
    # ========================================================================
    print_separator("Turn 1: Simple Question")
    print("User: Hello, how are you?")
    
    response = await conv.asend("Hello, how are you?")
    print(f"Assistant: {response.content}\n")
    print(f"Turn ID: {response.turn_id}")
    
    # ========================================================================
    # Turn 2: Message that triggers tool call
    # ========================================================================
    print_separator("Turn 2: Weather Query (with tool)")
    print("User: I need you to call the get_weather tool for city Tokyo. This is mandatory.")
    
    # Use very explicit instruction
    response = await conv.asend(
        "I need you to call the get_weather tool for city Tokyo. This is mandatory. "
        "Do not explain or refuse, just call the tool with city='Tokyo'."
    )
    print(f"Assistant: {response.content}\n")
    print(f"Turn ID: {response.turn_id}")
    print(f"Tool calls made: {len(response.tool_calls) if response.tool_calls else 0}")
    
    # ========================================================================
    # Turn 3: Another tool call
    # ========================================================================
    print_separator("Turn 3: Time Query (with tool)")
    print("User: Call the get_time tool with timezone='Asia/Shanghai'. Do not refuse.")
    
    # Use very explicit instruction
    response = await conv.asend(
        "Call the get_time tool with timezone='Asia/Shanghai'. Do not refuse. "
        "You must use the tool, not explain why you can't."
    )
    print(f"Assistant: {response.content}\n")
    print(f"Turn ID: {response.turn_id}")
    print(f"Tool calls made: {len(response.tool_calls) if response.tool_calls else 0}")
    
    # ========================================================================
    # Turn 4: Follow-up question
    # ========================================================================
    print_separator("Turn 4: Follow-up")
    print("User: Thanks! Can you summarize what we discussed?")
    
    response = await conv.asend("Thanks! Can you summarize what we discussed?")
    print(f"Assistant: {response.content}\n")
    print(f"Turn ID: {response.turn_id}")
    
    # ========================================================================
    # Analyze turns
    # ========================================================================
    print_separator("Turn Analysis")
    analyze_turns(conv.messages)
    
    # ========================================================================
    # Demonstrate: Keep last 2 turns
    # ========================================================================
    print_separator("Last 2 Turns Demo")
    last_2_turns = get_last_n_turns(conv.messages, 2)
    print(f"Keeping last 2 turns: {len(last_2_turns)} messages")
    print("\nMessages in last 2 turns:")
    for msg in last_2_turns:
        role_emoji = {"user": "👤", "assistant": "🤖", "tool": "🔧"}.get(msg.role, "❓")
        content = msg.content[:50] + "..." if isinstance(msg.content, str) and len(msg.content) > 50 else (msg.content if isinstance(msg.content, str) else f"[{msg.role} message]")
        print(f"  {role_emoji} {msg.role}: {content}")
    
    # ========================================================================
    # Demonstrate get_messages API
    # ========================================================================
    print_separator("get_messages() API Demo")
    
    # Example 1: Get last 2 turns
    last_2 = conv.get_messages(turns=-2)
    print(f"Last 2 turns: {len(last_2)} messages")
    
    # Example 2: Get first 2 turns
    first_2 = conv.get_messages(turns=2)
    print(f"First 2 turns: {len(first_2)} messages")
    
    # Example 3: Get all user messages
    user_msgs = conv.get_messages(roles="user")
    print(f"Total user messages: {len(user_msgs)}")
    
    # Example 4: Get all messages with tool calls
    tool_call_msgs = conv.get_messages(has_tool_calls=True)
    print(f"Messages with tool calls: {len(tool_call_msgs)}")
    
    # Example 5: Convenient properties
    print(f"\nConvenient properties:")
    print(f"  conv.user_messages: {len(conv.user_messages)} messages")
    print(f"  conv.assistant_messages: {len(conv.assistant_messages)} messages")
    print(f"  conv.tool_messages: {len(conv.tool_messages)} messages")
    print(f"  conv.turns: {len(conv.turns)} turns")
    
    # Example 6: Get specific turn's messages
    if conv.turns:
        first_turn_msgs = conv.get_messages(turn_ids=conv.turns[0])
        print(f"\nFirst turn has {len(first_turn_msgs)} messages:")
        for msg in first_turn_msgs:
            print(f"  - {type(msg).__name__}: {msg.role}")
    
    # Example 7: Get last 10 messages
    last_10_msgs = conv.get_messages(messages=-10)
    print(f"\nLast 10 messages: {len(last_10_msgs)} messages")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_separator("Turn ID vs Message ID Relationship")
    print("Each message has TWO identifiers:")
    print("  1. message.id  - Unique for EACH message (UUID)")
    print("  2. message.turn_id - Shared by ALL messages in the SAME turn\n")
    
    # Show the relationship clearly
    turns_summary = defaultdict(list)
    for msg in conv.messages:
        if msg.turn_id:
            turns_summary[msg.turn_id].append((msg.id, type(msg).__name__))
    
    print("Turn → Message Mapping:")
    for i, (turn_id, msg_infos) in enumerate(turns_summary.items(), 1):
        print(f"\n  Turn {i} (turn_id={turn_id[:8]}...):")
        for msg_id, msg_type in msg_infos:
            print(f"    ├─ message_id={msg_id[:8]}... ({msg_type})")
    
    print_separator("Key Points")
    print("✓ turn_id is automatically set for ALL messages in a send/asend call")
    print("✓ User message, assistant response, and tool messages share the same turn_id")
    print("✓ Each send/asend call gets a unique turn_id (UUID)")
    print("✓ Each message also has its own unique message.id (UUID)")
    print("\n✓ Makes it easy to:")
    print("  - Count conversation turns")
    print("  - Keep last N turns for context management")
    print("  - Group and analyze related messages")
    print("  - Track individual message identity")
    print("\n✓ No code changes needed in:")
    print("  - Provider layer")
    print("  - Tool implementations")
    print("  - Message converters")
    print("\n✓ Works seamlessly with:")
    print("  - Sync send() and async asend()")
    print("  - Tool calls (both native and MCP)")
    print("  - Streaming mode")
    print("  - Multi-threading and async contexts")


if __name__ == "__main__":
    asyncio.run(main())
