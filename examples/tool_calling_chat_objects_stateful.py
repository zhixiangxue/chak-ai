"""
Object Tool Demo - Test NativeObjectTool with stateful objects

This demo shows how to use Python objects as tools, where:
- Object methods are automatically exposed as individual tools
- Object state is preserved across method calls
- LLM can interact with stateful business logic
"""

import asyncio
import os
from datetime import datetime

import dotenv
dotenv.load_dotenv()

os.environ["CHAK_LOG_LEVEL"] = "DEBUG"

import chak


# Example 1: Calculator with history
class Calculator:
    """Calculator that tracks calculation history"""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers"""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a"""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers"""
        result = a * b
        self.history.append(f"{a} × {b} = {result}")
        return result
    
    def divide(self, a: int, b: int) -> float:
        """Divide a by b"""
        if b == 0:
            return "Error: Division by zero"
        result = a / b
        self.history.append(f"{a} ÷ {b} = {result}")
        return result
    
    def get_history(self) -> list:
        """Get all calculation history"""
        return self.history
    
    def clear_history(self) -> str:
        """Clear calculation history"""
        count = len(self.history)
        self.history = []
        return f"Cleared {count} history records"


# Example 2: Shopping cart
class ShoppingCart:
    """Shopping cart with items and discount"""
    
    def __init__(self):
        self.items = []
        self.discount = 0
    
    def add_item(self, name: str, price: float, quantity: int = 1) -> str:
        """Add item to cart"""
        self.items.append({
            "name": name,
            "price": price,
            "quantity": quantity
        })
        return f"Added {quantity}x {name} (${price} each)"
    
    def remove_item(self, name: str) -> str:
        """Remove item from cart"""
        for i, item in enumerate(self.items):
            if item["name"] == name:
                removed = self.items.pop(i)
                return f"Removed {removed['quantity']}x {removed['name']}"
        return f"Item '{name}' not found in cart"
    
    def apply_discount(self, percent: float) -> str:
        """Apply discount percentage"""
        self.discount = percent
        return f"Applied {percent}% discount"
    
    def get_total(self) -> float:
        """Calculate total price"""
        subtotal = sum(item["price"] * item["quantity"] for item in self.items)
        total = subtotal * (1 - self.discount / 100)
        return round(total, 2)
    
    def get_cart_summary(self) -> dict:
        """Get cart summary"""
        subtotal = sum(item["price"] * item["quantity"] for item in self.items)
        return {
            "items": self.items,
            "item_count": len(self.items),
            "subtotal": subtotal,
            "discount": f"{self.discount}%",
            "total": self.get_total()
        }


async def demo_calculator():
    """Demo: Calculator with history"""
    print("\n" + "="*60)
    print("Demo 1: Calculator with History")
    print("="*60)
    
    bailian_key = os.getenv("BAILIAN_API_KEY")
    
    # Create calculator instance
    calc = Calculator()
    
    # 🔍 验证点 1: 对象初始状态
    print(f"\n🔍 BEFORE: calc.history = {calc.history}")
    print(f"🔍 BEFORE: len(calc.history) = {len(calc.history)}")
    print(f"🔍 BEFORE: id(calc) = {id(calc)}")
    
    # Create conversation with calculator as tool
    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=bailian_key,
        tools=[calc]  # ✨ Pass object directly!
    )
    
    # Test questions
    questions = [
        "Calculate 123 + 456",
        "Now multiply 100 by 25",
        "Then subtract 500 from 2500",
    ]
    
    for q in questions:
        print(f"\n👤 User: {q}")
        response = await conv.asend(q)
        print(f"🤖 AI: {response.content}")
    
    # 🔍 验证点 2: 对象是否被修改
    print("\n" + "="*60)
    print("🔍 VALIDATION: Did the object change?")
    print("="*60)
    print(f"🔍 AFTER: calc.history = {calc.history}")
    print(f"🔍 AFTER: len(calc.history) = {len(calc.history)}")
    print(f"🔍 AFTER: id(calc) = {id(calc)}")
    
    if len(calc.history) > 0:
        print("\n✅ SUCCESS: Object state was modified by LLM!")
        print(f"   - History entries: {len(calc.history)}")
        for i, entry in enumerate(calc.history, 1):
            print(f"   - Entry {i}: {entry}")
    else:
        print("\n❌ FAILED: Object state was NOT modified!")


async def demo_shopping_cart():
    """Demo: Shopping cart"""
    print("\n" + "="*60)
    print("Demo 2: Shopping Cart")
    print("="*60)
    
    bailian_key = os.getenv("BAILIAN_API_KEY")
    
    # Create shopping cart instance
    cart = ShoppingCart()
    
    # 🔍 验证点 1: 对象初始状态
    print(f"\n🔍 BEFORE: cart.items = {cart.items}")
    print(f"🔍 BEFORE: cart.discount = {cart.discount}%")
    print(f"🔍 BEFORE: cart.get_total() = ${cart.get_total()}")
    print(f"🔍 BEFORE: id(cart) = {id(cart)}")
    
    # Create conversation with cart as tool
    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=bailian_key,
        tools=[cart]  # ✨ Pass object directly!
    )
    
    # Test questions
    questions = [
        "Add 2 iPhones at $999 each to my cart",
        "Add 1 MacBook at $1999",
        "Apply a 10% discount",
    ]
    
    for q in questions:
        print(f"\n👤 User: {q}")
        response = await conv.asend(q)
        print(f"🤖 AI: {response.content}")
    
    # 🔍 验证点 2: 对象是否被修改
    print("\n" + "="*60)
    print("🔍 VALIDATION: Did the cart object change?")
    print("="*60)
    print(f"🔍 AFTER: cart.items = {cart.items}")
    print(f"🔍 AFTER: cart.discount = {cart.discount}%")
    print(f"🔍 AFTER: cart.get_total() = ${cart.get_total()}")
    print(f"🔍 AFTER: id(cart) = {id(cart)}")
    
    # 详细验证
    if len(cart.items) > 0:
        print("\n✅ SUCCESS: Cart object was modified by LLM!")
        print(f"   - Items count: {len(cart.items)}")
        for i, item in enumerate(cart.items, 1):
            print(f"   - Item {i}: {item['quantity']}x {item['name']} @ ${item['price']}")
        print(f"   - Discount applied: {cart.discount}%")
        subtotal = sum(item['price'] * item['quantity'] for item in cart.items)
        print(f"   - Subtotal: ${subtotal}")
        print(f"   - Total (after discount): ${cart.get_total()}")
    else:
        print("\n❌ FAILED: Cart object was NOT modified!")


async def demo_mixed_objects():
    """Demo: Multiple objects + regular functions"""
    print("\n" + "="*60)
    print("Demo 3: Mixed Tools (Objects + Functions)")
    print("="*60)
    
    bailian_key = os.getenv("BAILIAN_API_KEY")
    
    # Regular function
    def get_current_time() -> str:
        """Get current date and time"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create instances
    calc = Calculator()
    cart = ShoppingCart()
    
    # 🔍 验证点 1: 对象初始状态
    print(f"\n🔍 BEFORE:")
    print(f"  calc.history = {calc.history}")
    print(f"  calc id = {id(calc)}")
    print(f"  cart.items = {cart.items}")
    print(f"  cart.discount = {cart.discount}%")
    print(f"  cart id = {id(cart)}")
    
    # Create conversation with mixed tools
    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=bailian_key,
        tools=[
            get_current_time,  # Regular function
            calc,              # Object 1
            cart,              # Object 2
        ]
    )
    
    # Complex question
    question = (
        "First tell me what time it is. "
        "Then calculate 50 times 20, and add the result to my shopping cart as 'Gift Card' at that price. "
        "Finally apply a 15% discount and tell me the total."
    )
    
    print(f"\n👤 User: {question}")
    response = await conv.asend(question)
    print(f"🤖 AI: {response.content}")
    
    # 🔍 验证点 2: 对象是否被修改
    print("\n" + "="*60)
    print("🔍 VALIDATION: Did the objects change?")
    print("="*60)
    print(f"\n📊 Calculator Object:")
    print(f"  calc.history = {calc.history}")
    print(f"  calc id = {id(calc)}")
    
    print(f"\n📊 ShoppingCart Object:")
    print(f"  cart.items = {cart.items}")
    print(f"  cart.discount = {cart.discount}%")
    print(f"  cart.get_total() = ${cart.get_total()}")
    print(f"  cart id = {id(cart)}")
    
    # 验证
    calc_modified = len(calc.history) > 0
    cart_modified = len(cart.items) > 0 or cart.discount > 0
    
    if calc_modified and cart_modified:
        print("\n✅ SUCCESS: Both objects were modified by LLM!")
        print(f"\n  Calculator:")
        for i, entry in enumerate(calc.history, 1):
            print(f"    - Calculation {i}: {entry}")
        print(f"\n  Shopping Cart:")
        for i, item in enumerate(cart.items, 1):
            print(f"    - Item {i}: {item['quantity']}x {item['name']} @ ${item['price']}")
        print(f"    - Discount: {cart.discount}%")
        print(f"    - Total: ${cart.get_total()}")
    else:
        print("\n❌ FAILED: Objects were NOT fully modified!")
        print(f"  - Calculator modified: {calc_modified}")
        print(f"  - Cart modified: {cart_modified}")


async def main():
    """Run all demos"""
    # Run mixed objects demo
    await demo_mixed_objects()
    
    print("\n" + "="*60)
    print("✅ Demo completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
