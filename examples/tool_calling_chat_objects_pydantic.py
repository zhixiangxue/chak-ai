"""
Stateful Object Tool with Pydantic models

Demonstrates combining stateful objects with Pydantic type safety:
- Object methods maintain state across calls
- Parameters and return values use Pydantic models
- Automatic validation and serialization

This example shows an order management system where the object
maintains order state while using Pydantic for type-safe data exchange.
"""

import asyncio
import os
from typing import Optional, List

from pydantic import BaseModel, Field
import dotenv

dotenv.load_dotenv()

import chak


# Define Pydantic models
class Product(BaseModel):
    """Product information"""
    name: str = Field(description="Product name")
    price: float = Field(description="Product price")
    quantity: int = Field(default=1, description="Quantity")


class Order(BaseModel):
    """Order details"""
    order_id: int = Field(description="Order ID")
    products: List[Product] = Field(description="Products in order")
    total: float = Field(description="Total price")
    status: str = Field(default="pending", description="Order status")


class OrderStats(BaseModel):
    """Order statistics"""
    total_orders: int = Field(description="Total number of orders")
    total_revenue: float = Field(description="Total revenue")
    pending_orders: int = Field(description="Number of pending orders")
    completed_orders: int = Field(description="Number of completed orders")


# Stateful Order Management System
class OrderManager:
    """
    Stateful order management system with Pydantic type safety
    
    This class demonstrates:
    - State preservation across LLM calls
    - Pydantic models for parameters and return values
    - Business logic encapsulation
    """
    
    def __init__(self):
        self.orders = []
        self.next_order_id = 1
    
    def create_order(self, product: Product) -> Order:
        """
        Create a new order with a product
        
        Args:
            product: Product to order
        
        Returns:
            Created order details
        """
        order = Order(
            order_id=self.next_order_id,
            products=[product],
            total=product.price * product.quantity,
            status="pending"
        )
        
        self.orders.append(order)
        self.next_order_id += 1
        
        return order
    
    def add_product_to_order(self, order_id: int, product: Product) -> Order:
        """
        Add a product to existing order
        
        Args:
            order_id: Order ID
            product: Product to add
        
        Returns:
            Updated order
        """
        for order in self.orders:
            if order.order_id == order_id:
                order.products.append(product)
                order.total += product.price * product.quantity
                return order
        
        # Return empty order if not found
        return Order(order_id=order_id, products=[], total=0, status="not_found")
    
    def complete_order(self, order_id: int) -> Order:
        """
        Mark order as completed
        
        Args:
            order_id: Order ID to complete
        
        Returns:
            Updated order
        """
        for order in self.orders:
            if order.order_id == order_id:
                order.status = "completed"
                return order
        
        return Order(order_id=order_id, products=[], total=0, status="not_found")
    
    def get_order(self, order_id: int) -> Order:
        """
        Get order details
        
        Args:
            order_id: Order ID
        
        Returns:
            Order details
        """
        for order in self.orders:
            if order.order_id == order_id:
                return order
        
        return Order(order_id=order_id, products=[], total=0, status="not_found")
    
    def get_stats(self) -> OrderStats:
        """
        Get order statistics
        
        Returns:
            Statistics summary
        """
        total_revenue = sum(order.total for order in self.orders)
        pending = sum(1 for order in self.orders if order.status == "pending")
        completed = sum(1 for order in self.orders if order.status == "completed")
        
        return OrderStats(
            total_orders=len(self.orders),
            total_revenue=total_revenue,
            pending_orders=pending,
            completed_orders=completed
        )


async def main():
    """Main function demonstrating stateful object + Pydantic"""
    
    # Check API key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ Error: DASHSCOPE_API_KEY not set")
        print("Please set your Bailian API key in .env file or environment variable")
        print("Get your key at: https://dashscope.aliyuncs.com/")
        return
    
    # Create order manager instance
    order_manager = OrderManager()
    
    # Print initial object ID for verification
    print("=== Stateful Object + Pydantic Demo ===\n")
    print(f"📦 OrderManager object ID: {id(order_manager)}")
    print(f"📊 Initial state: {len(order_manager.orders)} orders\n")
    
    # Create conversation with the stateful object
    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        tools=[order_manager]  # Pass object directly!
    )
    
    # Test 1: Create orders
    print("📝 Creating orders...")
    response = await conv.asend(
        "Create three orders: "
        "1. Laptop at $1200 (quantity 1), "
        "2. Mouse at $25 (quantity 2), "
        "3. Keyboard at $80 (quantity 1)"
    )
    print(f"✅ {response}\n")
    
    # Verify state changed
    print(f"📊 After creation: {len(order_manager.orders)} orders")
    print(f"   Object ID still: {id(order_manager)}\n")
    
    # Test 2: Add product to existing order
    print("➕ Adding product to order...")
    response = await conv.asend(
        "Add a Monitor ($350, quantity 1) to order ID 1"
    )
    print(f"✅ {response}\n")
    
    # Test 3: Complete an order
    print("✔️ Completing order...")
    response = await conv.asend(
        "Complete order ID 1"
    )
    print(f"✅ {response}\n")
    
    # Test 4: Get statistics
    print("📊 Getting statistics...")
    response = await conv.asend(
        "Show me the order statistics"
    )
    print(f"✅ {response}\n")
    
    # Verify final state
    print("=" * 70)
    print("🔍 Final State Verification:")
    print(f"   Total orders: {len(order_manager.orders)}")
    print(f"   Object ID unchanged: {id(order_manager)}")
    print(f"\n   Order details:")
    for order in order_manager.orders:
        print(f"     - Order {order.order_id}: {len(order.products)} products, "
              f"${order.total:.2f}, status={order.status}")
    print("=" * 70)
    
    # Show conversation stats
    print("\n📊 Conversation Statistics:")
    stats = conv.stats()
    print(f"   Total messages: {stats['total_messages']}")
    print(f"   Total tokens: {stats['total_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())
