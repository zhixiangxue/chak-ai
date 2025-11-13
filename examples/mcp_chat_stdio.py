"""
MCP with stdio Transport - Local FastMCP Calculator Example

This example shows how to use MCP server with stdio (standard I/O) transport.

Use Case:
    - Run local MCP servers
    - Custom tools via FastMCP framework
    - Private/offline tool execution
    - Build your own MCP services

How it works:
    1. Tool Loading: chak connects to your MCP server and loads available tools
    2. Automatic Detection: When you send a message, the model decides if tools are needed
    3. Multi-round Calling: chak automatically handles the tool calling loop:
       - Model requests tool → chak executes → returns result → model continues
       - Supports multiple rounds until final answer
    4. Streaming Support: Works with both streaming and non-streaming modes

Server initialization:
    Server(command="mcp", args=["run", "your_server.py"])
    
    For npm-based servers:
    Server(command="npx", args=["-y", "@modelcontextprotocol/server-example"])

Streaming with tools:
    async for chunk in await conv.asend("question", stream=True):
        print(chunk.content, end="", flush=True)
    
    chak handles the complexity:
    - Accumulates tool call instructions from streaming chunks
    - Executes tools in parallel when possible
    - Continues streaming the final answer

Filter specific tools:
    tools = await Server(command="...").tools(["add", "subtract"])
    conv = Conversation("openai/gpt-4o", tools=tools)

Graceful degradation:
    If the model doesn't support function calling, chak automatically falls back 
    to normal conversation mode - no errors, no hassle.

Prerequisites:
    - FastMCP installed in your venv (should be already installed)
    - Local server file: examples/my_calculator_mcp_server.py
    - Alibaba Cloud API key for the LLM
    
Usage:
    python examples/mcp_chat_stdio.py
"""

import asyncio
import os
import platform
import sys
from pathlib import Path

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
# Get API key for the LLM (Bailian)
api_key = os.getenv("BAILIAN_API_KEY", "Your API key here")
if not api_key:
    print("❌ Error: Please set BAILIAN_API_KEY environment variable")
    print("   Example: export BAILIAN_API_KEY=sk-your-key-here")
    exit(1)

print("="*70)
print("MCP with stdio Transport - Local Calculator Server")
print("="*70)
print()
print("Configuration:")
print("  Transport: stdio (Standard I/O)")
print("  Service: Local FastMCP Calculator")
print("  Model: bailian/qwen-flash")
print("  🔍 Debug Logging: ENABLED (CHAK_LOG_LEVEL=DEBUG)")
print()
print("How it works:")
print("  1. Runs a local MCP server (mcp_calculator_server.py)")
print("  2. Communicates via stdin/stdout")
print("  3. Provides calculator tools: add, subtract, multiply, divide, power")
print()
print("💡 This demonstrates how to create your own MCP servers!")
print("   See: examples/my_calculator_mcp_server.py for server code")
print()
print("="*70)
print()


async def main():
    # ========================================================================
    # Create MCP Server with stdio transport
    # ========================================================================
    # Locate the calculator server script (in same directory as this example)
    example_dir = Path(__file__).parent
    calc_server_path = example_dir / "my_calculator_mcp_server.py"
    
    if not calc_server_path.exists():
        print(f"❌ Error: Calculator server not found at: {calc_server_path}")
        print("   Please ensure examples/my_calculator_mcp_server.py exists")
        exit(1)
    
    # Find mcp executable (automatically search in common locations)
    mcp_exe = None
    
    # Try current venv first
    if platform.system() == "Windows":
        venv_mcp = Path(sys.executable).parent / "mcp.exe"
    else:
        venv_mcp = Path(sys.executable).parent / "mcp"
    
    if venv_mcp.exists():
        mcp_exe = venv_mcp
    else:
        # Try to find in PATH
        import shutil
        mcp_in_path = shutil.which("mcp")
        if mcp_in_path:
            mcp_exe = Path(mcp_in_path)
    
    if not mcp_exe:
        print(f"❌ Error: 'mcp' command not found")
        print("   Please install FastMCP: pip install mcp")
        exit(1)
    
    print(f"🔧 Server script: {calc_server_path.name}")
    print(f"🔧 MCP executable: {mcp_exe}")
    print()
    
    # Create server with stdio transport
    server = Server(
        command=str(mcp_exe),
        args=["run", str(calc_server_path)]
    )
    
    # List available tools
    print("📋 Available MCP Tools:")
    tools = await server.tools()
    for tool in tools:
        print(f"   - {tool.name}: {tool.description}")
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
        "Calculate 15 plus 7",
        "What is 2 to the power of 8?",
        "Divide 100 by 4",
        "Multiply 6 by 8"
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
    print("   - The local MCP server provides calculator tools")
    print("   - The LLM automatically calls tools when it needs to calculate")
    print("   - This runs completely locally (except LLM API calls)")
    print("   - 🔍 DEBUG logs show the complete MCP tool calling process")
    print()
    print("🔧 Build your own MCP server:")
    print("   1. Check out: examples/my_calculator_mcp_server.py")
    print("   2. Use FastMCP framework: from mcp.server.fastmcp import FastMCP")
    print("   3. Define tools with @mcp.tool() decorator")
    print("   4. Run with: mcp run your_server.py")
    print()
    print("🔧 To customize:")
    print("   - Modify the 'questions' list in the code")
    print("   - Or make it interactive by replacing the loop with input()")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
