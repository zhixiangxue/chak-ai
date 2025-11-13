"""
FIFO Context Strategy Example

This example shows how to use FIFO (First-In-First-Out) strategy to limit context length.

Use Case: 
    - Simple Q&A or customer service
    - Only need recent conversation history
    - No need for long-term memory

Features:
    - Keep only the last N conversation turns
    - Automatically truncate older messages
    - Simple and predictable behavior

Parameters:
    - keep_recent_turns: How many recent turns to keep? A turn = all content from 
                        one user message to the next user message.
    - max_input_tokens: Set a "stomach capacity" limit for the strategy, ensuring 
                       it won't overflow the model's context window.

How it works:
    The strategy inserts a truncation marker before the retention interval, sending 
    only content after the marker. Original conversation? All preserved in 
    conversation.messages.

Prerequisites:
    1. Get your API key from: https://bailian.console.aliyun.com
    2. Set environment variable: export BAILIAN_API_KEY=your_key_here
    
Usage:
    python examples/strategy_chat_fifo.py
"""

import os

import dotenv

dotenv.load_dotenv()

import chak
from chak.context.strategies import FIFOStrategy

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
print("FIFO Strategy - Keep Recent Turns Only")
print("="*70)
print()
print("Configuration:")
print("  Model: bailian/qwen-flash")
print("  Strategy: FIFO")
print("  keep_recent_turns = 3 (only keep last 3 turns)")
print()
print("How it works:")
print("  - Conversation keeps accumulating messages")
print("  - When > 3 turns exist, older turns are automatically truncated")
print("  - LLM only sees the most recent 3 turns")
print()
print("Type 'quit' to exit, 'history' to see message count")
print("="*70)
print()

# Create FIFO strategy
# keep_recent_turns: Number of recent conversation turns to keep
strategy = FIFOStrategy(keep_recent_turns=3)

# Create conversation with FIFO strategy
conv = chak.Conversation(
    "bailian/qwen-flash",
    api_key=api_key,
    context_strategy=strategy
)

# Interactive chat loop
turn_count = 0

while True:
    try:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'history':
            print(f"\n📊 Total messages in history: {len(conv.messages)}")
            print(f"   Conversation turns: {turn_count}")
            print()
            continue
        
        turn_count += 1
        
        # Send message with streaming
        print("Assistant: ", end="", flush=True)
        
        stream = conv.send(user_input, stream=True)
        for chunk in stream:  # type: ignore
            if chunk.content:  # type: ignore
                print(chunk.content, end="", flush=True)  # type: ignore
        
        print("\n")
        
        # Show hint when FIFO truncation happens
        if turn_count > 3:
            print(f"💡 FIFO active: Keeping only last 3 turns (total history: {len(conv.messages)} messages)")
            print()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        break
    except Exception as e:
        print(f"\nError: {e}")
        break

# Show final statistics
print("="*70)
print("Session Summary:")
print(f"  Total turns: {turn_count}")
print(f"  Messages in history: {len(conv.messages)}")
print("="*70)
