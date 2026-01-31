"""
Token Usage Tracking Example

This example demonstrates how to track token usage across different providers.
It shows the unified Usage structure (prompt_tokens, completion_tokens, total_tokens)
working consistently across OpenAI and Bailian.

Prerequisites:
    1. OpenAI API key: https://platform.openai.com/api-keys
    2. Bailian API key: https://bailian.console.aliyun.com
    3. Set environment variables:
       - export OPENAI_API_KEY=your_openai_key
       - export BAILIAN_API_KEY=your_bailian_key

Usage:
    python examples/usage_token_tracking.py
"""

import os

import dotenv

dotenv.load_dotenv()

import chak

# ============================================================================
# Configuration
# ============================================================================

providers_config = [
    {
        "name": "OpenAI GPT-4o-mini",
        "uri": "openai/gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "prompt": "Explain quantum computing in one sentence",
    },
    {
        "name": "Bailian Qwen-Plus",
        "uri": "bailian/qwen-plus",
        "api_key_env": "BAILIAN_API_KEY",
        "prompt": "用一句话解释量子计算",
    },
]


# ============================================================================
# Helper Functions
# ============================================================================

def print_separator(title=""):
    """Print a formatted separator line."""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    else:
        print(f"{'='*70}\n")


def print_usage_details(message):
    """Print detailed token usage information from a message."""
    usage = message.metadata.usage
    
    if usage:
        print(f"  Provider: {message.metadata.provider}")
        print(f"  Model: {message.metadata.model}")
        print(f"  Prompt tokens: {usage.prompt_tokens}")
        print(f"  Completion tokens: {usage.completion_tokens}")
        print(f"  Total tokens: {usage.total_tokens}")
        
        # Calculate approximate cost (example rates, adjust as needed)
        if message.metadata.provider == "openai":
            # GPT-4o-mini pricing (example)
            input_cost = usage.prompt_tokens * 0.15 / 1_000_000  # $0.15 per 1M tokens
            output_cost = usage.completion_tokens * 0.6 / 1_000_000  # $0.60 per 1M tokens
            total_cost = input_cost + output_cost
            print(f"  Estimated cost: ${total_cost:.6f}")
    else:
        print("  No usage information available")


# ============================================================================
# Main Demo
# ============================================================================

def main():
    print_separator("Token Usage Tracking Demo")
    
    for config in providers_config:
        try:
            # Get API key
            api_key = os.getenv(config["api_key_env"])
            if not api_key:
                print(f"⚠️  Skipping {config['name']}: {config['api_key_env']} not set")
                continue
            
            print_separator(config["name"])
            
            # Create conversation
            conv = chak.Conversation(config["uri"], api_key=api_key)
            
            # Send message
            print(f"Prompt: {config['prompt']}\n")
            response = conv.send(config["prompt"])
            
            print(f"Response: {response.content}\n")
            
            # Display usage details
            print("Token Usage:")
            print_usage_details(response)
            
            # Send another message to show cumulative stats
            print(f"\nSending follow-up question...")
            follow_up = conv.send("Can you explain that in simpler terms?")
            print(f"Response: {follow_up.content}\n")
            
            print("Follow-up Token Usage:")
            print_usage_details(follow_up)
            
            # Show cumulative conversation stats
            print(f"\n📊 Conversation Statistics:")
            stats = conv.stats()
            print(f"  Total messages: {stats['total_messages']}")
            print(f"  Input tokens: {stats['input_tokens']}")
            print(f"  Output tokens: {stats['output_tokens']}")
            print(f"  Total tokens: {stats['total_tokens']}")
            
        except Exception as e:
            print(f"❌ Error with {config['name']}: {str(e)}")
            continue
    
    print_separator("Demo Complete")
    print("✓ All providers use the same Usage structure:")
    print("  - prompt_tokens: Input tokens")
    print("  - completion_tokens: Output tokens")
    print("  - total_tokens: Sum of both")
    print("\n✓ Access usage via: message.metadata.usage")
    print("✓ View stats via: conversation.stats()")


if __name__ == "__main__":
    main()
