"""
Simple Chat with Mixed Tools - Native Functions + MCP

This example demonstrates mixing native Python functions with MCP tools.

Features:
    - Native Python functions (no decorators needed)
    - MCP tools (Exa web search via Smithery)
    - Both types work together seamlessly

Tools in this demo:
    Native Functions:
    - get_current_time: Get current date and time
    - calculate: Simple calculator (add, subtract, multiply, divide)
    
    MCP Tools (from Smithery/Exa):
    - exa_search: Advanced web search
    - exa_find_similar: Find similar content
    - exa_get_contents: Get page contents

Prerequisites:
    1. Smithery API key: https://smithery.ai/account/api-keys
       Set: export SMITHERY_KEY=your-key-here
    
    2. Bailian API key (for LLM): https://dashscope.aliyuncs.com/
       Set: export DASHSCOPE_API_KEY=your-key-here

Usage:
    python examples/simple_chat_with_tools.py
"""

import asyncio
import os
from datetime import datetime
from urllib.parse import urlencode

import dotenv

dotenv.load_dotenv()

# Enable debug logging to see tool calls
os.environ["CHAK_LOG_LEVEL"] = "DEBUG"

import chak
from chak.tools.mcp import Server


# ============================================================================
# Native Python Functions (No decorators needed!)
# ============================================================================

def get_current_time() -> str:
    """
    Get current date and time
    
    Returns:
        Current date and time in readable format
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(a: int, b: int, operation: str = "add") -> int:
    """
    Perform calculation on two numbers
    
    Args:
        a: First number
        b: Second number
        operation: Operation type (add, subtract, multiply, divide)
    
    Returns:
        Calculation result
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a // b if b != 0 else 0
    else:
        return 0


async def main():
    # ========================================================================
    # Check API Keys
    # ========================================================================
    smithery_key = os.getenv("SMITHERY_KEY")
    bailian_key = os.getenv("DASHSCOPE_API_KEY")
    
    if not smithery_key:
        print("❌ Error: SMITHERY_KEY not set")
        print("Get your key at: https://smithery.ai/account/api-keys")
        return
    
    if not bailian_key:
        print("❌ Error: DASHSCOPE_API_KEY not set")
        print("Get your key at: https://dashscope.aliyuncs.com/")
        return
    
    print("=" * 70)
    print("🔧 Simple Chat with Mixed Tools")
    print("=" * 70)
    print()
    print("Tools available:")
    print("  📌 Native Functions:")
    print("     - get_current_time: Get current date/time")
    print("     - calculate: Simple calculator")
    print()
    print("  🌐 MCP Tools (Exa via Smithery):")
    print("     - exa_search: Web search")
    print("     - exa_find_similar: Find similar content")
    print("     - exa_get_contents: Get page contents")
    print()
    print("=" * 70)
    print()
    
    # ========================================================================
    # Setup MCP Server (Smithery Exa)
    # ========================================================================
    profile = "forthcoming-alpaca-796g47"
    base_url = "https://server.smithery.ai/exa/mcp"
    params = {"api_key": smithery_key, "profile": profile}
    url = f"{base_url}?{urlencode(params)}"
    
    server = Server(url=url)
    
    print("🔗 Loading MCP tools from Smithery...")
    mcp_tools = await server.tools()
    print(f"✅ Loaded {len(mcp_tools)} MCP tools")
    print()
    
    # ========================================================================
    # Create Conversation with Mixed Tools
    # ========================================================================
    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=bailian_key,
        tools=[
            # Native Python functions
            get_current_time,
            calculate,
            # MCP tools
            *mcp_tools
        ]
    )
    
    print("💡 Watch for debug logs - you'll see both native and MCP tools being called!")
    print()
    print("=" * 70)
    
    # ========================================================================
    # Demo Questions (designed to use both types of tools)
    # ========================================================================
    questions = [
        # This will use native function: get_current_time
        "What time is it now?",
        
        # This will use native function: calculate
        "Calculate 456 multiplied by 123",
        
        # This will use MCP tool: exa_search
        "Search for recent news about Python 3.13 release",
        
        # This will use BOTH: calculate (native) + exa_search (MCP)
        "First, calculate 100 plus 200. Then search for information about LLM function calling",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}/{len(questions)}]")
        print(f"👤 You: {question}")
        print()
        
        response = await conv.asend(question)
        
        print(f"🤖 AI: {response.content}")
        print("-" * 70)
    
    # ========================================================================
    # Show Statistics
    # ========================================================================
    print()
    print("=" * 70)
    print("📊 Conversation Statistics")
    print("=" * 70)
    stats = conv.stats()
    print(f"Total Messages: {stats['total_messages']}")
    print(f"Total Tokens: {stats['total_tokens']}")
    print(f"Input Tokens: {stats['input_tokens']}")
    print(f"Output Tokens: {stats['output_tokens']}")
    print()
    print("=" * 70)
    print("✅ Demo completed!")
    print()
    print("💡 Key Takeaways:")
    print("   - Native Python functions work seamlessly with MCP tools")
    print("   - No decorators or special wrappers needed")
    print("   - The LLM automatically chooses the right tool for each task")
    print("   - Both sync (calculate) and async (MCP) tools work together")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
