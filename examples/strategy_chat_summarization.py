"""
Summarization Context Strategy Example

This example shows how to use Summarization strategy to compress context automatically.

Use Case:
    - Need long-term conversation memory
    - Want to control token consumption
    - Maintain context while staying within token limits

Features:
    - Automatically summarize old messages when token threshold is reached
    - Keep recent turns in full detail
    - Older messages are replaced with LLM-generated summaries
    - Save tokens while preserving context

Parameters:
    - max_input_tokens: How large is your model's context window? The strategy uses 
                       this to decide when to trigger.
    - summarize_threshold: At what percentage of the window should summarization 
                          trigger? 0.75 = 75%, leaving room for future conversation.
    - prefer_recent_turns: Keep the last few turns untouched to maintain the "live 
                          feel" of the conversation.
    - summarizer_model_uri / summarizer_api_key: Which model to use for summarization? 
                                                Can be the same as main conversation or 
                                                a cheaper one.

How it works:
    When conversations accumulate to a certain length, chak automatically triggers 
    summarization. It condenses early conversations into key points and inserts a 
    marker into the message chain. Subsequent sends only include this marker and 
    content after it. This preserves complete history while significantly reducing 
    actual tokens sent, allowing you to continue conversations without worrying about 
    context window size.
    
    Original conversations remain fully preserved in conversation.messages, ready for 
    viewing, export, or analysis anytime.

Prerequisites:
    1. Get your API key from: https://bailian.console.aliyun.com
    2. Set environment variable: export BAILIAN_API_KEY=your_key_here
    
Usage:
    python examples/strategy_chat_summarization.py
"""

import os

import dotenv

dotenv.load_dotenv()

import chak
from chak.context.strategies import SummarizationStrategy

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
print("Summarization Strategy - Auto Compress Context")
print("="*70)
print()
print("Configuration:")
print("  Model: bailian/qwen-flash")
print("  Strategy: Summarization")
print("  max_input_tokens = 2000")
print("  summarize_threshold = 0.7 (trigger at 70%)")
print("  prefer_recent_turns = 2 (keep last 2 turns in full)")
print()
print("How it works:")
print("  - When tokens reach 70% of 2000 (1400 tokens)")
print("  - Older messages are summarized by LLM")
print("  - Recent 2 turns are kept in full detail")
print("  - Summary replaces old messages, saving tokens")
print()
print("Type 'quit' to exit, 'history' to see message count")
print("="*70)
print()

# Create Summarization strategy
strategy = SummarizationStrategy(
    max_input_tokens=2000,              # Maximum tokens for input context
    summarize_threshold=0.7,            # Trigger summarization at 70% (1400 tokens)
    prefer_recent_turns=2,              # Keep last 2 turns in full detail
    summarizer_model_uri="bailian/qwen-flash",  # Model for generating summaries
    summarizer_api_key=api_key
)

# Create conversation with Summarization strategy
conv = chak.Conversation(
    "bailian/qwen-flash",
    api_key=api_key,
    context_strategy=strategy
)

# Interactive chat loop
turn_count = 0
summarization_triggered = False

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
            
            # Check if summarization has occurred
            from chak.message import MarkerMessage
            marker_count = sum(1 for m in conv.messages if isinstance(m, MarkerMessage))
            if marker_count > 0:
                print(f"   📝 Summaries created: {marker_count}")
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
        
        # Check if summarization was triggered
        from chak.message import MarkerMessage
        marker_count = sum(1 for m in conv.messages if isinstance(m, MarkerMessage))
        if marker_count > 0 and not summarization_triggered:
            summarization_triggered = True
            print(f"📝 Summarization triggered! Old messages compressed into summary.")
            print(f"   Recent {strategy.prefer_recent_turns} turns kept in full detail.")
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

from chak.message import MarkerMessage
marker_count = sum(1 for m in conv.messages if isinstance(m, MarkerMessage))
if marker_count > 0:
    print(f"  Summaries created: {marker_count}")
    print("  ✅ Context compressed while preserving meaning")
else:
    print("  ℹ️  No summarization occurred (didn't reach threshold)")

print("="*70)
