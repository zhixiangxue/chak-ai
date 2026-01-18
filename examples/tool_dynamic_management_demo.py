"""
Dynamic Tool Management Demo

This example demonstrates how to dynamically manage tools in a conversation:
- Add tools during conversation
- Remove tools when not needed
- Clear all tools
- Get current tools list

Key scenarios:
1. Start with basic tools
2. Add specialized tools when needed
3. Remove tools to control LLM behavior
4. Clear tools for plain conversation

Use case: Different conversation phases need different tool sets
"""

import os
import asyncio
from pathlib import Path
import chak

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    print("⚠️  Warning: python-dotenv not installed, using system environment variables")
    pass


# ============================================================================
# Define some tools for demonstration
# ============================================================================

def calculate(expression: str) -> float:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return float(result)
    except Exception as e:
        return f"Error: {str(e)}"


def get_weather(city: str) -> str:
    """Get weather information for a city (mock data)."""
    weather_data = {
        "Beijing": "Sunny, 25°C",
        "Shanghai": "Cloudy, 22°C",
        "Guangzhou": "Rainy, 28°C",
    }
    return weather_data.get(city, f"Weather data not available for {city}")


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (mock)."""
    return f"Email sent to {to}: {subject}"


class ShoppingCart:
    """Shopping cart with state management."""
    
    def __init__(self):
        self.items = []
    
    def add_item(self, name: str, price: float, quantity: int = 1):
        """Add item to cart."""
        self.items.append({"name": name, "price": price, "quantity": quantity})
        return f"Added {quantity}x {name} to cart"
    
    def get_total(self) -> float:
        """Calculate total price."""
        return sum(item["price"] * item["quantity"] for item in self.items)


# ============================================================================
# Main demo
# ============================================================================
async def main():
    api_key = os.getenv("BAILIAN_API_KEY", "")
    if not api_key:
        print("❌ Error: Please set BAILIAN_API_KEY environment variable")
        return
    
    print("=" * 70)
    print("Dynamic Tool Management Demo")
    print("=" * 70)
    print()
    
    # ========================================================================
    # Phase 1: Start with basic calculation tool
    # ========================================================================
    print("📍 Phase 1: Basic Calculation")
    print("-" * 70)
    
    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        tools=[calculate],
        system_message="You are a helpful assistant."
    )
    
    print(f"Current tools: {[t.__name__ if callable(t) else t.__class__.__name__ for t in conv.get_tools()]}")
    print()
    
    response = await conv.asend("What is 123 * 456?")
    print(f"Assistant: {response.content}")
    print()
    
    # ========================================================================
    # Phase 2: Add weather tool for new task
    # ========================================================================
    print("📍 Phase 2: Add Weather Tool")
    print("-" * 70)
    
    conv.add_tools([get_weather])
    print(f"Current tools: {[t.__name__ if callable(t) else t.__class__.__name__ for t in conv.get_tools()]}")
    print()
    
    response = await conv.asend("What's the weather in Beijing?")
    print(f"Assistant: {response.content}")
    print()
    
    # ========================================================================
    # Phase 3: Add shopping cart for e-commerce scenario
    # ========================================================================
    print("📍 Phase 3: Add Shopping Cart")
    print("-" * 70)
    
    cart = ShoppingCart()
    conv.add_tools([cart])
    print(f"Current tools: {[t.__name__ if callable(t) else t.__class__.__name__ for t in conv.get_tools()]}")
    print()
    
    response = await conv.asend("Add 2 iPhones at $999 each to my cart, then tell me the total")
    print(f"Assistant: {response.content}")
    print(f"Cart state: {cart.items}, Total: ${cart.get_total()}")
    print()
    
    # ========================================================================
    # Phase 4: Remove weather tool (not needed anymore)
    # ========================================================================
    print("📍 Phase 4: Remove Weather Tool")
    print("-" * 70)
    
    conv.remove_tools([get_weather])
    print(f"Current tools: {[t.__name__ if callable(t) else t.__class__.__name__ for t in conv.get_tools()]}")
    print()
    
    response = await conv.asend("Calculate 100 + 200")
    print(f"Assistant: {response.content}")
    print()
    
    # ========================================================================
    # Phase 5: Add email tool temporarily
    # ========================================================================
    print("📍 Phase 5: Add Email Tool Temporarily")
    print("-" * 70)
    
    conv.add_tools([send_email])
    print(f"Current tools: {[t.__name__ if callable(t) else t.__class__.__name__ for t in conv.get_tools()]}")
    print()
    
    response = await conv.asend("Send an email to user@example.com with subject 'Order Confirmation'")
    print(f"Assistant: {response.content}")
    print()
    
    # ========================================================================
    # Phase 6: Clear all tools for plain conversation
    # ========================================================================
    print("📍 Phase 6: Clear All Tools (Plain Conversation)")
    print("-" * 70)
    
    conv.clear_tools()
    print(f"Current tools: {conv.get_tools()}")
    print()
    
    response = await conv.asend("What can you help me with now?")
    print(f"Assistant: {response.content}")
    print()
    
    # ========================================================================
    # Phase 7: Re-add tools for new task
    # ========================================================================
    print("📍 Phase 7: Re-add Tools for New Task")
    print("-" * 70)
    
    conv.add_tools([calculate, get_weather])
    print(f"Current tools: {[t.__name__ if callable(t) else t.__class__.__name__ for t in conv.get_tools()]}")
    print()
    
    response = await conv.asend("Calculate 50 * 20 and tell me the weather in Shanghai")
    print(f"Assistant: {response.content}")
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("✅ Demo Completed!")
    print()
    print("💡 Key Takeaways:")
    print("   1. get_tools() - Get current tools in original format")
    print("   2. add_tools() - Add new tools anytime during conversation")
    print("   3. remove_tools() - Remove tools by reference")
    print("   4. clear_tools() - Clear all tools for plain conversation")
    print()
    print("🎯 Use Cases:")
    print("   - Multi-phase workflows with different tool requirements")
    print("   - Dynamic tool loading based on user intent")
    print("   - Tool access control (add/remove based on permissions)")
    print("   - Testing different tool combinations")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
