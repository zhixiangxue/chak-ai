"""
LRU Context Handler Example

Simple example showing how to use LRUContextHandler.

Note: Current implementation is simplified - just keeps recent messages.
      Full LRU with topic detection will be implemented later.

Use Case:
    - Keep recently used messages
    - Simple message-based filtering

How it works:
    - Keeps last N messages
    - Older messages are filtered out
    - Full history is preserved

Prerequisites:
    Set environment variable: export BAILIAN_API_KEY=your_key_here
"""

import os
import dotenv

dotenv.load_dotenv()

import chak
from chak.context.handlers import LRUContextHandler

# Get API key
api_key = os.getenv("BAILIAN_API_KEY", "")
if not api_key:
    print("❌ Error: Please set BAILIAN_API_KEY environment variable")
    exit(1)

print("="*60)
print("LRU Handler - Keep Recently Used Messages")
print("="*60)
print()
print("Configuration: Keep last 10 messages")
print("="*60)
print()

# Create LRU handler (keep last 10 messages)
handler = LRUContextHandler(keep_recent=10)

# Create conversation
conv = chak.Conversation(
    "bailian/qwen-flash",
    api_key=api_key,
    context_handler=handler
)

# Auto test messages
test_messages = [
    "What's 1+1?",
    "What about 2+2?",
    "And 3+3?",
    "What's 4+4?",
    "How about 5+5?",
    "Tell me 6+6",
]

print("Running auto test...\n")

for i, msg in enumerate(test_messages, 1):
    print(f"[Turn {i}] You: {msg}")
    print("Assistant: ", end="", flush=True)
    
    response = conv.send(msg)
    print(response.content)
    
    # Show stats
    print(f"  📊 History: {len(conv.messages)} | Context: {len(handler.output_messages)} | Dropped: {len(handler.input_messages) - len(handler.output_messages)}")
    print()

print("="*60)
print(f"Completed {len(test_messages)} turns")
print(f"Final history: {len(conv.messages)} messages")
print("="*60)
