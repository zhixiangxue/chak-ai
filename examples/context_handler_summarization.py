"""
Summarization Context Handler Example

Simple example showing how to use SummarizationContextHandler to compress long conversations.

Use Case:
    - Long conversations that need context compression
    - Control token consumption
    - Maintain context while staying within limits

How it works:
    - When messages exceed threshold, old messages are summarized
    - Summary + recent messages are sent to LLM
    - Full history is preserved

Prerequisites:
    Set environment variable: export BAILIAN_API_KEY=your_key_here
"""

import os
import dotenv

dotenv.load_dotenv()

import chak
from chak.context.handlers import SummarizationContextHandler

# Get API key
api_key = os.getenv("BAILIAN_API_KEY", "")
if not api_key:
    print("❌ Error: Please set BAILIAN_API_KEY environment variable")
    exit(1)

print("="*60)
print("Summarization Handler - Auto Compress Context")
print("="*60)
print()
print("Configuration:")
print("  Max messages: 10 (trigger summarization)")
print("  Keep recent: 5 messages")
print()
print("="*60)
print()

# Create summarization handler
handler = SummarizationContextHandler(
    max_messages=10,              # Trigger when > 10 messages
    keep_recent=5,                # Keep last 5 messages
    summarizer_model_uri="bailian/qwen-flash",
    summarizer_api_key=api_key
)

# Create conversation
conv = chak.Conversation(
    "bailian/qwen-flash",
    api_key=api_key,
    context_handler=handler
)

# Auto test messages (more than 10 to trigger summarization)
test_messages = [
    "What's 1+1?",
    "What about 2+2?",
    "And 3+3?",
    "What's 4+4?",
    "How about 5+5?",
    "Tell me 6+6",
    "What's 7+7?",
    "And 8+8?",
]

print("Running auto test...\n")

for i, msg in enumerate(test_messages, 1):
    print(f"[Turn {i}] You: {msg}")
    print("Assistant: ", end="", flush=True)
    
    response = conv.send(msg)
    print(response.content)
    
    # Show stats
    compressed = len(handler.input_messages) - len(handler.output_messages)
    print(f"  📊 History: {len(conv.messages)} | Context: {len(handler.output_messages)} | Compressed: {compressed}")
    print()

print("="*60)
print(f"Completed {len(test_messages)} turns")
print(f"Final history: {len(conv.messages)} messages")
print("="*60)
