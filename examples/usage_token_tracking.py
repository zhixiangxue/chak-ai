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
        # In chak's canonical Usage semantics all four buckets are disjoint:
        # prompt_tokens is fresh (non-cached) input only, and total_tokens is
        # the sum of the four disjoint buckets — the same shape on OpenAI and
        # Anthropic alike. See chak/metadata.py::Usage for details.
        print(f"  Prompt tokens (fresh input): {usage.prompt_tokens}")
        print(f"  Completion tokens:           {usage.completion_tokens}")
        print(f"  Cache write tokens:          {usage.cache_creation_input_tokens}")
        print(f"  Cache read tokens:           {usage.cache_read_input_tokens}")
        print(f"  Total tokens:                {usage.total_tokens}")
        
        # Uniform cost formula across providers: pass per-MTok unit prices
        # and Usage.estimate_cost multiplies each disjoint bucket for you.
        # Example rates below are GPT-4o-mini's; swap in real numbers per model.
        if message.metadata.provider == "openai":
            cost = usage.estimate_cost(
                input_price=0.15,
                output_price=0.60,
                # OpenAI's automatic prompt caching: cache_read is 50% off.
                cache_read_price=0.075,
                currency="USD",
            )
            # ``cost`` is a Money object — str(cost) prints as
            # "0.000123 USD"; cost.amount is the raw float; sum([...]) works
            # across a list of Money as long as currencies match.
            print(f"  Estimated cost: {cost}")
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
