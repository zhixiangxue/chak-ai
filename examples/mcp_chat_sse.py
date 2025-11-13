"""
MCP with SSE Transport - Alibaba Cloud Bailian Example

This example shows how to use MCP server with SSE (Server-Sent Events) transport.

Use Case:
    - Connect to cloud-hosted MCP services
    - Alibaba Cloud Bailian Amap Maps service
    - Query geographic coordinates from addresses

How it works:
    1. Tool Loading: chak connects to your MCP server and loads available tools
    2. Automatic Detection: When you send a message, the model decides if tools are needed
    3. Multi-round Calling: chak automatically handles the tool calling loop:
       - Model requests tool → chak executes → returns result → model continues
       - Supports multiple rounds until final answer
    4. Streaming Support: Works with both streaming and non-streaming modes

Server initialization methods:
    1. Direct construction (used in this example):
       Server(url="...", headers={...})
    
    2. From JSON string:
       config = '{"url": "...", "headers": {...}}'
       server = Server.from_config(config)
    
    3. From Dict:
       config = {"url": "...", "headers": {...}}
       server = Server.from_config(config)

Streaming with tools:
    async for chunk in await conv.asend("question", stream=True):
        print(chunk.content, end="", flush=True)
    
    chak handles the complexity:
    - Accumulates tool call instructions from streaming chunks
    - Executes tools in parallel when possible
    - Continues streaming the final answer

Filter specific tools:
    tools = await Server(url="...").tools(["weather", "calculator"])
    conv = Conversation("openai/gpt-4o", tools=tools)

Graceful degradation:
    If the model doesn't support function calling, chak automatically falls back 
    to normal conversation mode - no errors, no hassle.

Prerequisites:
    1. Get your API key from: https://bailian.console.aliyun.com
    2. Set environment variable: export BAILIAN_API_KEY=your_key_here
    
Usage:
    python examples/mcp_chat_sse.py
"""

import asyncio
import os

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
# Get API key from environment variable
api_key = os.getenv("BAILIAN_API_KEY", "Your API key here")
if not api_key:
    print("❌ Error: Please set BAILIAN_API_KEY environment variable")
    print("   Example: export BAILIAN_API_KEY=sk-your-key-here")
    exit(1)

print("="*70)
print("MCP with SSE Transport - Amap Maps Service")
print("="*70)
print()
print("Configuration:")
print("  Transport: SSE (Server-Sent Events)")
print("  Service: Alibaba Cloud Bailian - Amap Maps")
print("  Model: bailian/qwen-flash")
print("  🔍 Debug Logging: ENABLED (CHAK_LOG_LEVEL=DEBUG)")
print()
print("Available initialization methods:")
print("  1. Direct construction (used below):")
print("     Server(url=..., headers=...)")
print()
print("  2. From JSON string:")
print("     config = '{\"url\": \"...\", \"headers\": {...}}'")
print("     server = Server.from_config(config)")
print()
print("  3. From Dict:")
print("     config = {\"url\": \"...\", \"headers\": {...}}")
print("     server = Server.from_config(config)")
print()
print("="*70)
print()


async def main():
    # ========================================================================
    # Create MCP Server with SSE transport
    # ========================================================================
    server = Server(
        url="https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    # Alternative Method 2: From JSON config (commented out)
    # config_json = f'''{{
    #     "url": "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse",
    #     "headers": {{"Authorization": "Bearer {api_key}"}}
    # }}'''
    # server = Server.from_config(config_json)
    
    # Alternative Method 3: From Dict config (commented out)
    # config_dict = {
    #     "url": "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse",
    #     "headers": {"Authorization": f"Bearer {api_key}"}
    # }
    # server = Server.from_config(config_dict)
    
    # List available tools
    print("📋 Available MCP Tools:")
    tools = await server.tools()
    for tool in tools:
        print(f"   - {tool.name}: {tool.description[:60]}...")
    print()
    
    # ========================================================================
    # Create Conversation with MCP tools
    # ========================================================================
    conv = chak.Conversation(
        model_uri="bailian/qwen-flash",
        api_key=api_key,
        tools=tools
    )
    
    # ========================================================================
    # Pre-defined questions for quick demo
    # ========================================================================
    questions = [
        "What's the weather like in Beijing?",  # maps_weather
        "Find restaurants near Tiananmen Square",  # maps_geo + maps_around_search
        "How far is it from Shanghai Oriental Pearl Tower to the Bund?",  # maps_geo + maps_distance
        "Plan a walking route from West Lake to Lingyin Temple in Hangzhou"  # maps_geo + maps_direction_walking
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
    print("   - The MCP server (Amap Maps) provides geocoding tools")
    print("   - The LLM automatically calls tools when needed")
    print("   - Try asking about other famous landmarks!")
    print("   - 🔍 DEBUG logs show the complete MCP tool calling process")
    print()
    print("🔧 To customize:")
    print("   - Modify the 'questions' list in the code")
    print("   - Or make it interactive by replacing the loop with input()")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
