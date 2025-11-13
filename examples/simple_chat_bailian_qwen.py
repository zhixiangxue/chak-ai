"""
Bailian (Alibaba Cloud) Example

This example shows how to use chak with Alibaba Cloud's Bailian service.

Prerequisites:
    1. Get your API key from: https://bailian.console.aliyun.com
    2. Set environment variable: export BAILIAN_API_KEY=your_key_here
    
Usage:
    python examples/example_bailian.py
"""

import os

import dotenv

dotenv.load_dotenv()

import chak

# ============================================================================
# 🔑🔑🔑 IMPORTANT: Set your API key here 🔑🔑🔑
# ============================================================================
# Get API key from environment variable (recommended for security)
api_key = os.getenv("BAILIAN_API_KEY", "Your API key here")
if not api_key:
    print("❌ Error: Please set BAILIAN_API_KEY environment variable")
    print("   Example: export BAILIAN_API_KEY=sk-your-key-here")
    exit(1)

# Create conversation with simple URI format
# Format: provider/model
conv = chak.Conversation(
    "bailian/qwen-plus",  # Use qwen-plus model
    api_key=api_key
)

# Send a simple message
print("Sending message to Bailian...")
response = conv.send("用一句话解释什么是上下文管理")
print(f"\nResponse: {response.content}")

# Example with streaming
print("\n" + "="*70)
print("Streaming example:")
print("="*70 + "\n")

stream = conv.send("写一首关于人工智能的五言诗", stream=True)
for chunk in stream:  # type: ignore
    if chunk.content:  # type: ignore
        print(chunk.content, end="", flush=True)  # type: ignore

print("\n")

# View conversation stats
print("\n" + "="*70)
print("Conversation statistics:")
print("="*70)
stats = conv.stats()
print(f"Total messages: {stats['total_messages']}")
print(f"Total tokens: {stats.get('total_tokens', 'N/A')}")
