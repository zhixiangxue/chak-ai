"""
Message Filtering API Demo

This example demonstrates the powerful get_messages() API for filtering and
retrieving messages from a conversation. This is particularly useful for:
- Building custom context handlers
- Implementing message retention policies
- Analyzing conversation patterns
- Managing memory and token usage

The API supports multiple filter dimensions:
- turns: Filter by turn index/range (positive, negative, or tuple)
- messages: Filter by message index/range
- roles: Filter by message role(s)
- has_tool_calls: Filter messages with/without tool calls
- has_attachments: Filter messages with/without attachments
- turn_ids: Filter by specific turn ID(s)
- message_ids: Filter by specific message ID(s)

Prerequisites:
    1. Bailian API key: https://bailian.console.aliyun.com
    2. Set environment variable: export BAILIAN_API_KEY=your_key_here

Usage:
    python examples/message_filtering_demo.py
"""

import os
import asyncio

import dotenv
dotenv.load_dotenv()

import chak


# ============================================================================
# Define some tools
# ============================================================================

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': Found 3 relevant articles about {query}."


def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"


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


def print_messages(messages, title="Messages"):
    """Print a list of messages with details."""
    print(f"{title} ({len(messages)} total):")
    for i, msg in enumerate(messages, 1):
        role_emoji = {
            "user": "👤",
            "assistant": "🤖",
            "tool": "🔧",
            "system": "⚙️"
        }.get(msg.role, "❓")
        
        msg_type = type(msg).__name__
        content_preview = ""
        
        if msg.role == "tool":
            content_preview = f"[tool result]"
        elif isinstance(msg.content, str):
            content_preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        
        has_tools = "🔧" if msg.tool_calls else ""
        print(f"  {i}. {role_emoji} {msg_type} {has_tools}")
        if content_preview:
            print(f"     {content_preview}")


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
    
    print_separator("Message Filtering API Demo")
    
    # Create conversation with tools
    tools = [search_web, calculate, get_weather]
    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        tools=tools
    )
    
    # ========================================================================
    # Build a conversation with multiple turns and tool calls
    # ========================================================================
    print("Building conversation with multiple turns...\n")
    
    # Turn 1: Simple question
    print("Turn 1: Hello")
    await conv.asend("Hello!")
    
    # Turn 2: Web search (with tool)
    print("Turn 2: Web search with tool")
    await conv.asend("Search for information about Python asyncio")
    
    # Turn 3: Calculation (with tool)
    print("Turn 3: Math calculation with tool")
    await conv.asend("Calculate 123 * 456")
    
    # Turn 4: Weather query (with tool)
    print("Turn 4: Weather query with tool")
    await conv.asend("What's the weather in Shanghai?")
    
    # Turn 5: Summary
    print("Turn 5: Summary request")
    await conv.asend("Thanks! Summarize what we discussed.")
    
    print(f"\n✅ Conversation built: {len(conv.messages)} messages across {len(conv.turns)} turns\n")
    
    # ========================================================================
    # Demo 1: Filter by Turn Index
    # ========================================================================
    print_separator("Demo 1: Filter by Turn Index")
    
    print("1.1 Get last 2 turns (context window simulation):")
    last_2_turns = conv.get_messages(turns=-2)
    print_messages(last_2_turns, "Last 2 turns")
    
    print("\n1.2 Get first 3 turns (conversation history):")
    first_3_turns = conv.get_messages(turns=3)
    print_messages(first_3_turns, "First 3 turns")
    
    print("\n1.3 Get turns 2-4 (specific range):")
    mid_turns = conv.get_messages(turns=(1, 4))
    print_messages(mid_turns, "Turns 2-4")
    
    # ========================================================================
    # Demo 2: Filter by Message Index
    # ========================================================================
    print_separator("Demo 2: Filter by Message Index")
    
    print("2.1 Get last 5 messages (sliding window):")
    last_5_msgs = conv.get_messages(messages=-5)
    print_messages(last_5_msgs, "Last 5 messages")
    
    print("\n2.2 Get first 10 messages (history limit):")
    first_10_msgs = conv.get_messages(messages=10)
    print_messages(first_10_msgs, "First 10 messages")
    
    # ========================================================================
    # Demo 3: Filter by Role
    # ========================================================================
    print_separator("Demo 3: Filter by Role")
    
    print("3.1 Get all user messages:")
    user_msgs = conv.get_messages(roles="user")
    print_messages(user_msgs, "User messages")
    
    print("\n3.2 Get all tool messages:")
    tool_msgs = conv.get_messages(roles="tool")
    print_messages(tool_msgs, "Tool messages")
    
    print("\n3.3 Get user + tool messages (exclude assistant):")
    user_tool_msgs = conv.get_messages(roles=["user", "tool"])
    print_messages(user_tool_msgs, "User + Tool messages")
    
    # ========================================================================
    # Demo 4: Filter by Tool Calls
    # ========================================================================
    print_separator("Demo 4: Filter by Tool Calls")
    
    print("4.1 Get messages with tool calls:")
    with_tools = conv.get_messages(has_tool_calls=True)
    print_messages(with_tools, "Messages with tool calls")
    
    print("\n4.2 Get assistant messages WITH tool calls:")
    assistant_tools = conv.get_messages(roles="assistant", has_tool_calls=True)
    print_messages(assistant_tools, "Assistant messages with tool calls")
    
    print("\n4.3 Get assistant messages WITHOUT tool calls:")
    assistant_no_tools = conv.get_messages(roles="assistant", has_tool_calls=False)
    print_messages(assistant_no_tools, "Assistant messages without tool calls")
    
    # ========================================================================
    # Demo 5: Combined Filters (Real-world scenarios)
    # ========================================================================
    print_separator("Demo 5: Combined Filters (Real-world)")
    
    print("5.1 Context Handler Scenario: Last 2 turns, exclude tools")
    print("    (Keep recent context, but strip tool details)")
    context_msgs = conv.get_messages(turns=-2, roles=["user", "assistant"])
    print_messages(context_msgs, "Context for next call")
    
    print("\n5.2 Token Optimization: Last 3 turns, only final responses")
    print("    (Compact history by removing intermediate tool calls)")
    compact_msgs = conv.get_messages(turns=-3, roles=["user", "assistant"], has_tool_calls=False)
    print_messages(compact_msgs, "Compact history")
    
    print("\n5.3 Tool Analysis: All tool interactions in last 10 messages")
    print("    (Analyze tool usage patterns)")
    tool_analysis = conv.get_messages(messages=-10, roles=["assistant", "tool"], has_tool_calls=True)
    print_messages(tool_analysis, "Recent tool interactions")
    
    # ========================================================================
    # Demo 6: Convenient Properties
    # ========================================================================
    print_separator("Demo 6: Convenient Properties")
    
    print(f"conv.user_messages: {len(conv.user_messages)} messages")
    print(f"conv.assistant_messages: {len(conv.assistant_messages)} messages")
    print(f"conv.tool_messages: {len(conv.tool_messages)} messages")
    print(f"conv.turns: {len(conv.turns)} turns")
    
    print("\nFirst turn messages:")
    first_turn = conv.get_messages(turn_ids=conv.turns[0])
    print_messages(first_turn, "Turn 1")
    
    # ========================================================================
    # Demo 7: Building a Simple Context Handler
    # ========================================================================
    print_separator("Demo 7: Context Handler Example")
    
    print("Simulating a context handler that keeps:")
    print("  - Last 2 turns")
    print("  - Only user and assistant final responses")
    print("  - Exclude intermediate tool calls\n")
    
    managed_context = conv.get_messages(
        turns=-2,
        roles=["user", "assistant"],
        has_tool_calls=False
    )
    
    print(f"Original: {len(conv.messages)} messages")
    print(f"Managed: {len(managed_context)} messages")
    print(f"Reduction: {len(conv.messages) - len(managed_context)} messages removed\n")
    
    print_messages(managed_context, "Managed Context")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_separator("Key Takeaways")
    print("✓ Flexible message filtering by turns, roles, tool calls, etc.")
    print("✓ Combine multiple filters with AND logic")
    print("✓ Use negative indices for recent messages/turns")
    print("✓ Use tuples for ranges (start, end)")
    print("✓ Convenient properties: user_messages, assistant_messages, tool_messages, turns")
    print("\n✓ Perfect for building:")
    print("  - Custom context handlers")
    print("  - Token management strategies")
    print("  - Conversation analysis tools")
    print("  - Message retention policies")
    print_separator()


if __name__ == "__main__":
    asyncio.run(main())
