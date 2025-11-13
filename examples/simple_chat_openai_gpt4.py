"""
OpenAI GPT-4 Example

This example shows how to use chak with OpenAI's GPT-4 model.

Prerequisites:
    1. Get your API key from: https://platform.openai.com/api-keys
    2. Set environment variable: export OPENAI_API_KEY=your_key_here
    
Usage:
    python examples/simple_chat_openai_gpt4.py
"""

import os

import dotenv

dotenv.load_dotenv()

import chak

# ============================================================================
# 🔑🔑🔑 IMPORTANT: Set your API key here 🔑🔑🔑
# ============================================================================
# Get API key from environment variable (recommended for security)
api_key = os.getenv("OPENAI_API_KEY", "Your API key here")
if not api_key:
    print("❌ Error: Please set OPENAI_API_KEY environment variable")
    print("   Example: export OPENAI_API_KEY=sk-your-key-here")
    exit(1)

# Create conversation with simple URI format
# Format: provider/model
conv = chak.Conversation(
    "openai/gpt-4o-mini",  # Use gpt-4o-mini model (faster and cheaper than gpt-4)
    api_key=api_key
)

# Send a simple message
print("Sending message to OpenAI...")
response = conv.send("Explain what is context management in one sentence")
print(f"\nResponse: {response.content}")

# Example with streaming
print("\n" + "="*70)
print("Streaming example:")
print("="*70 + "\n")

stream = conv.send("Write a haiku about artificial intelligence", stream=True)
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
