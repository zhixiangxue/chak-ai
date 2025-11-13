"""
Ollama Local Deployment Example

This example shows how to use chak with locally deployed Ollama models.

Prerequisites:
    1. Install Ollama: https://ollama.com
    2. Pull a model: ollama pull qwen3:8b
    3. Start Ollama service (usually runs on http://localhost:11434)
    
Usage:
    python examples/example_ollama.py
"""

import chak

# Ollama runs locally, no API key needed (use 'ollama' as placeholder)
# Use complete URI format with custom base_url
# Format: provider@base_url:model
conv = chak.Conversation(
    "ollama@http://localhost:11434/v1:qwen3:8b",
    api_key="ollama"  # Ollama doesn't use API keys, but required by SDK
)

print("="*70)
print("Chatting with local Ollama model")
print("="*70 + "\n")

# Send a message
print("Sending message...")
response = conv.send("Explain what is context management in one sentence")
print(f"\nResponse: {response.content}")  # type: ignore

# Example with streaming
print("\n" + "="*70)
print("Streaming example:")
print("="*70 + "\n")

stream = conv.send("Write a haiku about AI", stream=True)
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
print(f"Message types: {stats['by_type']}")

print("\n💡 Tips:")
print("  - Ollama models run locally, completely free")
print("  - No internet connection needed")
print("  - Available models: qwen2.5, llama3.1, phi3, etc.")
print("  - Pull models: ollama pull qwen2.5")
print("  - List models: ollama list")
