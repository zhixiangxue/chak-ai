"""
FIFO Context Handler Example

Simple example showing how to use FIFOContextHandler to keep only recent conversation turns.

Use Case:
    - Simple Q&A or customer service
    - Only need recent conversation history
    - No long-term memory required

How it works:
    - Keeps only the last N conversation turns
    - Older messages are filtered out
    - History is preserved but not sent to LLM

Prerequisites:
    Set environment variable: export BAILIAN_API_KEY=your_key_here
"""

import os
import dotenv

dotenv.load_dotenv()

import chak
from chak.context.handlers import FIFOContextHandler

# Get API key
api_key = os.getenv("BAILIAN_API_KEY", "")
if not api_key:
    print("❌ Error: Please set BAILIAN_API_KEY environment variable")
    exit(1)

print("="*60)
print("FIFO Handler - Keep Recent Turns Only")
print("="*60)
print()
print("Configuration: Keep last 3 turns")
print("="*60)
print()

# Create FIFO handler (keep last 3 turns)
handler = FIFOContextHandler(keep_recent_turns=3)

# Create conversation
conv = chak.Conversation(
    "bailian/qwen-flash",
    api_key=api_key,
    context_handler=handler
)

# Auto test messages
test_messages = [
    "Hello, what's 1+1?",
    "What about 2+2?",
    "And 3+3?",
    "What's 4+4?",
    "Finally, 5+5?",
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
