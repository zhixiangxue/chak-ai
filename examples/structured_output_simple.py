"""
Structured Output Example

Demonstrates using the 'returns' parameter to get structured data
from LLM responses via automatic function calling.

This example shows how to:
- Define Pydantic models for structured output
- Use the returns parameter to force structured responses
- Extract validated data from LLM
- Support for List[BaseModel] and Dict[str, BaseModel] (NEW!)
"""

import asyncio
import os
from typing import Optional, List, Dict

from pydantic import BaseModel, Field
import dotenv

dotenv.load_dotenv()

import chak


# Define output models
class User(BaseModel):
    """User information"""
    name: str = Field(description="User's full name")
    email: Optional[str] = Field(default=None, description="User's email address")
    age: Optional[int] = Field(default=None, description="User's age")


class Product(BaseModel):
    """Product information"""
    name: str = Field(description="Product name")
    price: float = Field(description="Product price in USD")
    category: str = Field(description="Product category")
    in_stock: bool = Field(default=True, description="Whether product is in stock")


class Sentiment(BaseModel):
    """Sentiment analysis result"""
    label: str = Field(description="Sentiment label: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the sentiment")


class Company(BaseModel):
    """Company information"""
    name: str = Field(description="Company name")
    industry: str = Field(description="Industry sector")
    employee_count: int = Field(description="Number of employees")


async def main():
    """Main function demonstrating structured output"""
    
    # Check API key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ Error: DASHSCOPE_API_KEY not set")
        print("Please set your Bailian API key in .env file or environment variable")
        print("Get your key at: https://dashscope.aliyuncs.com/")
        return
    
    # Create conversation
    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key
    )
    
    print("=== Structured Output Demo ===\n")
    
    # Example 1: Extract user information
    print("📝 Example 1: Extract Single User")
    print("-" * 50)
    user = await conv.asend(
        "Create a user profile for John Doe, email john@example.com, 30 years old",
        returns=User
    )
    if user:
        print(f"✅ Result type: {type(user).__name__}")
        print(f"   Name: {user.name}")
        print(f"   Email: {user.email}")
        print(f"   Age: {user.age}")
    else:
        print("❌ Failed to extract user information")
    print()
    
    # Example 2: Extract product information
    print("📝 Example 2: Extract Single Product")
    print("-" * 50)
    product = await conv.asend(
        "I'm looking at a MacBook Pro 16-inch, it costs $2499 and belongs to the laptops category",
        returns=Product
    )
    if product:
        print(f"✅ Result type: {type(product).__name__}")
        print(f"   Name: {product.name}")
        print(f"   Price: ${product.price}")
        print(f"   Category: {product.category}")
        print(f"   In Stock: {product.in_stock}")
    else:
        print("❌ Failed to extract product information")
    print()
    
    # Example 3: Extract List[BaseModel] - NEW!
    print("📝 Example 3: Extract List of Products (NEW!)")
    print("-" * 50)
    products = await conv.asend(
        "List 3 popular tech products: iPhone 15 Pro ($999), MacBook Air ($1199), AirPods Pro ($249)",
        returns=List[Product]
    )
    if products:
        print(f"✅ Result type: {type(products).__name__} with {len(products)} items")
        for i, prod in enumerate(products, 1):
            print(f"   {i}. {prod.name}: ${prod.price} ({prod.category})")
    else:
        print("❌ Failed to extract products")
    print()
    
    # Example 4: Extract Dict[str, BaseModel] - NEW!
    print("📝 Example 4: Extract Dict of Companies (NEW!)")
    print("-" * 50)
    companies = await conv.asend(
        "Give me info about 3 tech companies: Apple (technology, 150000 employees), Microsoft (software, 220000 employees), Google (internet, 190000 employees)",
        returns=Dict[str, Company]
    )
    if companies:
        print(f"✅ Result type: {type(companies).__name__} with {len(companies)} items")
        for key, company in companies.items():
            print(f"   {key}: {company.name} ({company.industry}, {company.employee_count:,} employees)")
    else:
        print("❌ Failed to extract companies")
    print()
    
    # Example 5: Extract List[User] - Multiple users
    print("📝 Example 5: Extract List of Users")
    print("-" * 50)
    users = await conv.asend(
        "Create profiles for 3 scientists: Albert Einstein (age 76), Marie Curie (age 66), Isaac Newton (age 84)",
        returns=List[User]
    )
    if users:
        print(f"✅ Result type: list with {len(users)} users")
        for i, user in enumerate(users, 1):
            print(f"   {i}. {user.name} (Age: {user.age})")
    else:
        print("❌ Failed to extract users")
    print()
    
    # Example 6: Sentiment analysis (single model)
    print("📝 Example 6: Sentiment Analysis")
    print("-" * 50)
    sentiment = await conv.asend(
        "This movie was absolutely amazing! The acting was superb and the plot kept me engaged throughout.",
        returns=Sentiment
    )
    if sentiment:
        print(f"✅ Result type: {type(sentiment).__name__}")
        print(f"   Label: {sentiment.label}")
        print(f"   Confidence: {sentiment.confidence:.2f}")
        print(f"   Reasoning: {sentiment.reasoning}")
    else:
        print("❌ Failed to extract sentiment")
    print()
    
    # Show conversation stats
    print("📊 Conversation Statistics:")
    stats = conv.stats()
    print(f"   Total messages: {stats['total_messages']}")
    print(f"   Total tokens: {stats['total_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())
