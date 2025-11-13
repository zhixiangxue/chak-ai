"""
MCP with HTTP Transport - Smithery Exa Example

This example shows how to use MCP server with HTTP (streamable-http) transport.

Use Case:
    - Connect to HTTP-based MCP services
    - Web search with Exa via Smithery
    - Advanced search and content discovery

How it works:
    1. Tool Loading: chak connects to your MCP server and loads available tools
    2. Automatic Detection: When you send a message, the model decides if tools are needed
    3. Multi-round Calling: chak automatically handles the tool calling loop:
       - Model requests tool → chak executes → returns result → model continues
       - Supports multiple rounds until final answer
    4. Streaming Support: Works with both streaming and non-streaming modes

Server initialization:
    Server(url="http://localhost:8080/mcp", transport="http")
    
    For services requiring authentication:
    Server(url="https://api.example.com/mcp", 
           transport="http",
           headers={"Authorization": "Bearer TOKEN"})

Streaming with tools:
    async for chunk in await conv.asend("question", stream=True):
        print(chunk.content, end="", flush=True)
    
    chak handles the complexity:
    - Accumulates tool call instructions from streaming chunks
    - Executes tools in parallel when possible
    - Continues streaming the final answer

Filter specific tools:
    tools = await Server(url="...").tools(["search", "summarize"])
    conv = Conversation("openai/gpt-4o", tools=tools)

Graceful degradation:
    If the model doesn't support function calling, chak automatically falls back 
    to normal conversation mode - no errors, no hassle.

Prerequisites:
    1. Register at: https://smithery.ai (free account)
    2. Get your API key at: https://smithery.ai/account/api-keys
    3. Set environment variable: export SMITHERY_KEY=your_key_here
    
Usage:
    python examples/mcp_chat_http.py
"""

import asyncio
import os
from urllib.parse import urlencode

import dotenv

dotenv.load_dotenv()

# ============================================================================
# 🔍 Enable debug logging to see MCP tool calls
# ============================================================================
os.environ["CHAK_LOG_LEVEL"] = "DEBUG"

import chak
from chak.mcp import Server

# ============================================================================
# 🔑🔑🔑 IMPORTANT: Set your API key here 🔑🔑🔑
# ============================================================================
# Get your Smithery API key at: https://smithery.ai/account/api-keys
api_key = os.getenv("SMITHERY_KEY", "Your API key here")
if not api_key:
    print("❌ Error: Please set SMITHERY_KEY environment variable")
    print()
    print("   How to get your key:")
    print("   1. Register at: https://smithery.ai")
    print("   2. Go to: https://smithery.ai/account/api-keys")
    print("   3. Copy your API key")
    print("   4. Set: export SMITHERY_KEY=your-key-here")
    print()
    exit(1)

print("="*70)
print("MCP with HTTP Transport - Exa Web Search")
print("="*70)
print()
print("Configuration:")
print("  Transport: HTTP (streamable-http)")
print("  Service: Exa via Smithery")
print("  Model: bailian/qwen-flash")
print("  🔍 Debug Logging: ENABLED (CHAK_LOG_LEVEL=DEBUG)")
print()
print("How to get Smithery API key:")
print("  1. Register at: https://smithery.ai (free)")
print("  2. Get key at: https://smithery.ai/account/api-keys")
print()
print("="*70)
print()


async def main():
    # ========================================================================
    # Create MCP Server with HTTP transport
    # ========================================================================
    # Smithery profile (you can use the default or create your own)
    profile = "forthcoming-alpaca-796g47"
    
    # Construct HTTP URL with query parameters
    base_url = "https://server.smithery.ai/exa/mcp"
    params = {"api_key": api_key, "profile": profile}
    url = f"{base_url}?{urlencode(params)}"
    
    server = Server(url=url)
    
    print(f"🔗 Connected to: {base_url}")
    print(f"   Profile: {profile}")
    print()
    
    # List available tools
    print("📋 Available MCP Tools:")
    tools = await server.tools()
    for tool in tools:
        print(f"   - {tool.name}: {tool.description[:60]}...")
    print()
    
    # ========================================================================
    # Create Conversation with MCP tools
    # ========================================================================
    # Note: Using Bailian API key for the LLM
    bailian_api_key = os.getenv("BAILIAN_API_KEY", "")
    if not bailian_api_key:
        print("❌ Error: Please also set BAILIAN_API_KEY for the LLM")
        print("   The Smithery key is only for MCP tools")
        exit(1)
    
    conv = chak.Conversation(
        model_uri="bailian/qwen-flash",
        api_key=bailian_api_key,
        tools=tools
    )
    
    # ========================================================================
    # Pre-defined questions for quick demo
    # ========================================================================
    questions = [
        "Search for recent information about Python async programming best practices",
        "Find articles about LLM gateway design patterns",
        "What are the latest developments in FastAPI framework?"
    ]
    
    print("🤖 Running demo with pre-defined questions...")
    print("   (You can modify the questions list in the code)")
    print()
    print("💡 Watch for debug logs below:")
    print("   🔧 = MCP tool being called")
    print("   📨 = Sending request to MCP server")
    print("   📦 = Receiving response from MCP server")
    print()
    print("="*70)
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}/{len(questions)}]")
        print(f"You: {question}")
        print()
        
        # Send message (non-streaming to avoid mixing with logs)
        response = await conv.asend(question, stream=False)  # type: ignore

        print()
        print(f"Assistant: {response.content}")  # type: ignore
        print("-"*70)
    
    print()
    print("="*70)
    print("✅ Demo completed!")
    print()
    print("💡 Tips:")
    print("   - Exa provides powerful web search capabilities")
    print("   - The LLM uses search results to give informed answers")
    print("   - Try asking about current events or technical topics!")
    print("   - 🔍 DEBUG logs show the complete MCP tool calling process")
    print()
    print("🔧 To customize:")
    print("   - Modify the 'questions' list in the code")
    print("   - Or make it interactive by replacing the loop with input()")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
