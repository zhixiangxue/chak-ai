"""
LRU Context Strategy Example

This example shows how to use LRU (Least Recently Used) strategy to auto-forget cold topics.

Use Case:
    - Multi-topic long conversations
    - Automatically focus on hot topics
    - Forget irrelevant old topics
    - Complex consulting or tutoring scenarios

Features:
    - Based on Summarization strategy
    - Automatically detects hot topics from recent context
    - Cold topics are forgotten (pruned)
    - Adaptive focus on what matters most

Parameters:
    - Same parameters as SummarizationStrategy, usage is identical:
      * max_input_tokens: Context window size
      * summarize_threshold: Trigger threshold (0.75 = 75%)
      * prefer_recent_turns: Keep recent turns in full detail
      * summarizer_model_uri / summarizer_api_key: Model for generating summaries
    - Internal enhancement: Based on Summarization strategy, additionally analyzes 
                          the last 5 summary markers
    - Smart forgetting: Detects which topics are no longer discussed, automatically 
                       fades cold topics, reinforces hot content

How it works:
    1. First works like SummarizationStrategy, generating summary markers
    2. When summary markers accumulate to a certain amount, LRU enhancement activates
    3. Analyzes the last 5 markers, identifying "hot topics" (continuously discussed) 
       and "cold topics" (no longer mentioned)
    4. Creates LRU markers, keeping only hot topic content, fading cold topics
    5. Original summary markers and complete history remain preserved for viewing anytime

Example use cases:
    - Conversations with frequent topic switches (e.g., Python → Java → Machine Learning)
    - Long conversations focusing only on current discussion topics
    - Want the model to "forget" early irrelevant topics, focusing on current task

Prerequisites:
    1. Get your API key from: https://bailian.console.aliyun.com
    2. Set environment variable: export BAILIAN_API_KEY=your_key_here
    
Usage:
    python examples/strategy_chat_lru.py
"""

import os

import dotenv

dotenv.load_dotenv()

import chak
from chak.context.strategies import LRUStrategy

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
print("LRU Strategy - Auto Forget Cold Topics")
print("="*70)
print()
print("Configuration:")
print("  Model: bailian/qwen-flash")
print("  Strategy: LRU (Least Recently Used)")
print("  max_input_tokens = 2000")
print("  summarize_threshold = 0.7 (trigger at 70%)")
print("  prefer_recent_turns = 2 (keep last 2 turns in full)")
print()
print("How it works:")
print("  1. Initially works like Summarization strategy")
print("  2. When enough summaries accumulate")
print("  3. LRU analyzes recent context to detect hot topics")
print("  4. Creates new summary with ONLY hot topics")
print("  5. Cold topics are forgotten (pruned)")
print()
print("Example scenario:")
print("  - Discuss Python → then Java → then Machine Learning")
print("  - As you continue ML discussion, Python/Java may be forgotten")
print("  - Context stays focused on current hot topic (ML)")
print()
print("Type 'quit' to exit, 'history' to see message count")
print("="*70)
print()

# Create LRU strategy
strategy = LRUStrategy(
    max_input_tokens=2000,              # Maximum tokens for input context
    summarize_threshold=0.7,            # Trigger summarization at 70%
    prefer_recent_turns=2,              # Keep last 2 turns in full detail
    summarizer_model_uri="bailian/qwen-flash",  # Model for generating summaries
    summarizer_api_key=api_key
)

# Create conversation with LRU strategy
conv = chak.Conversation(
    "bailian/qwen-flash",
    api_key=api_key,
    context_strategy=strategy
)

# Interactive chat loop
turn_count = 0
lru_triggered = False

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
            
            # Check marker types
            from chak.message import MarkerMessage
            summary_markers = sum(1 for m in conv.messages 
                                 if isinstance(m, MarkerMessage) 
                                 and m.metadata.get('type') == 'summary')
            lru_markers = sum(1 for m in conv.messages 
                            if isinstance(m, MarkerMessage) 
                            and m.metadata.get('type') == 'lru')
            
            if summary_markers > 0:
                print(f"   📝 Summary markers: {summary_markers}")
            if lru_markers > 0:
                print(f"   🗑️  LRU markers (topic pruning): {lru_markers}")
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
        
        # Check if LRU pruning occurred
        from chak.message import MarkerMessage
        lru_markers = [m for m in conv.messages 
                      if isinstance(m, MarkerMessage) 
                      and m.metadata.get('type') == 'lru']
        
        if lru_markers and not lru_triggered:
            lru_triggered = True
            print(f"🗑️  LRU topic pruning activated!")
            print(f"   Cold topics have been forgotten, focusing on hot topics.")
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
summary_markers = sum(1 for m in conv.messages 
                     if isinstance(m, MarkerMessage) 
                     and m.metadata.get('type') == 'summary')
lru_markers = sum(1 for m in conv.messages 
                 if isinstance(m, MarkerMessage) 
                 and m.metadata.get('type') == 'lru')

if lru_markers > 0:
    print(f"  📝 Summary markers: {summary_markers}")
    print(f"  🗑️  LRU markers: {lru_markers}")
    print("  ✅ Topic pruning occurred - context stayed focused")
elif summary_markers > 0:
    print(f"  📝 Summary markers: {summary_markers}")
    print("  ℹ️  Summarization occurred, but LRU not triggered yet")
else:
    print("  ℹ️  No summarization occurred (didn't reach threshold)")

print("="*70)
