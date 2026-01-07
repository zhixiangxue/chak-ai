"""
Tool calling example with Pydantic models

Demonstrates using Pydantic BaseModel for:
- Type-safe function parameters
- Structured return values
- Automatic validation and serialization

This example shows a user management system where both input and output
use Pydantic models for better type safety and developer experience.
"""

import asyncio
import os
from typing import Optional

from pydantic import BaseModel, Field
import dotenv
from rich import print

dotenv.load_dotenv()

import chak


# Define Pydantic models for structured data
class UserInput(BaseModel):
    """User creation input"""
    name: str = Field(description="User's full name")
    email: str = Field(description="User's email address")
    age: Optional[int] = Field(default=None, description="User's age")


class UserOutput(BaseModel):
    """User data output"""
    id: int = Field(description="User's unique ID")
    name: str = Field(description="User's full name")
    email: str = Field(description="User's email address")
    age: Optional[int] = Field(default=None, description="User's age")
    status: str = Field(default="active", description="User account status")


class SearchCriteria(BaseModel):
    """Search criteria for finding users"""
    min_age: Optional[int] = Field(default=None, description="Minimum age")
    max_age: Optional[int] = Field(default=None, description="Maximum age")
    status: Optional[str] = Field(default=None, description="Account status filter")


class SearchResult(BaseModel):
    """Search result with user list"""
    count: int = Field(description="Number of users found")
    users: list[UserOutput] = Field(description="List of matching users")


# Simulated database
users_db = []
next_user_id = 1


def create_user(user: UserInput) -> UserOutput:
    """
    Create a new user in the system
    
    Args:
        user: User information to create
    
    Returns:
        Created user with assigned ID
    """
    global next_user_id
    
    new_user = UserOutput(
        id=next_user_id,
        name=user.name,
        email=user.email,
        age=user.age,
        status="active"
    )
    
    users_db.append(new_user)
    next_user_id += 1
    
    return new_user


def get_user(user_id: int) -> UserOutput:
    """
    Get user by ID
    
    Args:
        user_id: The user's unique ID
    
    Returns:
        User information
    """
    for user in users_db:
        if user.id == user_id:
            return user
    
    # Return a placeholder if not found
    return UserOutput(
        id=user_id,
        name="Unknown",
        email="unknown@example.com",
        status="not_found"
    )


def search_users(criteria: SearchCriteria) -> SearchResult:
    """
    Search for users matching criteria
    
    Args:
        criteria: Search filters to apply
    
    Returns:
        Search results with matching users
    """
    matching_users = []
    
    for user in users_db:
        # Apply age filters
        if criteria.min_age is not None and (user.age is None or user.age < criteria.min_age):
            continue
        if criteria.max_age is not None and (user.age is None or user.age > criteria.max_age):
            continue
        
        # Apply status filter
        if criteria.status is not None and user.status != criteria.status:
            continue
        
        matching_users.append(user)
    
    return SearchResult(
        count=len(matching_users),
        users=matching_users
    )


async def main():
    """Main function demonstrating Pydantic tool usage"""
    
    # Check API key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ Error: DASHSCOPE_API_KEY not set")
        print("Please set your Bailian API key in .env file or environment variable")
        print("Get your key at: https://dashscope.aliyuncs.com/")
        return
    
    # Create conversation with Pydantic-based tools
    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        tools=[create_user, get_user, search_users]
    )
    
    print("=== Pydantic Tool Calling Demo ===\n")
    
    # Test 1: Create users
    print("📝 Creating users...")
    response = await conv.asend(
        "Create three users: "
        "Alice (alice@example.com, 25 years old), "
        "Bob (bob@example.com, 30 years old), "
        "Charlie (charlie@example.com, 28 years old)"
    )
    print(f"✅ {response}\n")
    
    # Test 2: Search users by age range
    print("🔍 Searching users...")
    response = await conv.asend(
        "Find all users between 26 and 30 years old"
    )
    print(f"✅ {response}\n")
    
    # Test 3: Get specific user
    print("👤 Getting user details...")
    response = await conv.asend(
        "Get the details of user with ID 2"
    )
    print(f"✅ {response}\n")
    
    # Show conversation stats
    print("📊 Conversation Statistics:")
    stats = conv.stats()
    print(f"   Total messages: {stats['total_messages']}")
    print(f"   Total tokens: {stats['total_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())
