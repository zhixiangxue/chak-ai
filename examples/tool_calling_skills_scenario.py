"""Skill-based tool calling scenario: Testing all execution modes

Demonstrates how Skills work across different conversation modes:
- Non-streaming mode
- Streaming mode  
- Event stream mode (with tool observability)

Note: Skills use progressive disclosure - LLM may retry tool calls
as it learns the correct parameters. This is expected behavior.

Prerequisites:
    Set environment variables based on provider:
    - DASHSCOPE_API_KEY for bailian/*
    - OPENAI_API_KEY for openai/*
    - etc.

Usage:
    python examples/tool_calling_skills_scenario.py bailian/qwen-plus
    python examples/tool_calling_skills_scenario.py openai/gpt-4o
"""

import argparse
import asyncio
import os
from dotenv import load_dotenv

from chak import Conversation
from chak.tools import SkillBase, wrap_tools

# Load environment variables
load_dotenv()


def get_api_key(model_uri: str) -> str:
    """Get API key based on provider name."""
    provider = model_uri.split("/")[0].upper()
    key = os.getenv(f"{provider}_API_KEY", "")
    if not key:
        raise ValueError(f"Please set {provider}_API_KEY environment variable")
    return key


class CalculatorSkill(SkillBase):
    """Calculator skill for basic math operations"""
    
    name = "calculator"
    description = "Perform basic mathematical calculations"
    
    def power(self, base: int, exponent: int) -> int:
        """Calculate power of a number
        
        Args:
            base: Base number
            exponent: Exponent
            
        Returns:
            base raised to the power of exponent
        """
        result = base ** exponent
        print(f"\n[Tool Executed] calculator.power({base}, {exponent}) = {result}")
        return result


async def scenario_non_streaming(model_uri: str):
    """Scenario 1: Non-streaming mode (execute_loop)"""
    print("\n" + "="*70)
    print("Scenario 1: Non-streaming mode")
    print("="*70)
    
    api_key = get_api_key(model_uri)
    calc_skill = CalculatorSkill()
    tools = wrap_tools([calc_skill])
    
    conv = Conversation(
        model_uri=model_uri,
        api_key=api_key,
        tools=tools
    )
    
    try:
        print("User: Calculate 2^10 using calculator")
        response = await conv.asend("Calculate 2 to the power of 10, use the calculator tool")
        print(f"Assistant: {response.content}")
        print("Scenario 1 completed")
        return True
    except Exception as e:
        print(f"Scenario 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def scenario_streaming(model_uri: str):
    """Scenario 2: Streaming mode (execute_loop_stream)"""
    print("\n" + "="*70)
    print("Scenario 2: Streaming mode")
    print("="*70)
    
    api_key = get_api_key(model_uri)
    calc_skill = CalculatorSkill()
    tools = wrap_tools([calc_skill])
    
    conv = Conversation(
        model_uri=model_uri,
        api_key=api_key,
        tools=tools
    )
    
    try:
        print("User: Calculate 3^5 using calculator")
        print("Assistant: ", end="", flush=True)
        
        async for chunk in await conv.asend("Calculate 3 to the power of 5, use the calculator tool", stream=True):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        
        print("\nScenario 2 completed")
        return True
    except Exception as e:
        print(f"\nScenario 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def scenario_event_stream(model_uri: str):
    """Scenario 3: Event stream mode (execute_loop_with_events)"""
    print("\n" + "="*70)
    print("Scenario 3: Event stream mode (with tool observability)")
    print("="*70)
    
    api_key = get_api_key(model_uri)
    calc_skill = CalculatorSkill()
    tools = wrap_tools([calc_skill])
    
    conv = Conversation(
        model_uri=model_uri,
        api_key=api_key,
        tools=tools
    )
    
    try:
        from chak.message import MessageChunk, ToolCallStartEvent, ToolCallSuccessEvent, ToolCallErrorEvent
        
        print("User: Calculate 4^3 using calculator")
        print("Assistant: ", end="", flush=True)
        
        async for event in await conv.asend("Calculate 4 to the power of 3, use the calculator tool", event=True):
            if isinstance(event, MessageChunk):
                if event.content:
                    print(event.content, end="", flush=True)
            elif isinstance(event, ToolCallStartEvent):
                print(f"\n[Event] Tool call started: {event.tool_name}")
            elif isinstance(event, ToolCallSuccessEvent):
                print(f"[Event] Tool call succeeded: {event.result}")
            elif isinstance(event, ToolCallErrorEvent):
                print(f"[Event] Tool call error: {event.error}")
        
        print("\nScenario 3 completed")
        return True
    except Exception as e:
        print(f"\nScenario 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all skill-based tool calling scenarios"""
    parser = argparse.ArgumentParser(description="Skill-based tool calling scenarios")
    parser.add_argument(
        "model_uri",
        nargs="?",
        default="bailian/qwen-plus",
        help="Model URI (e.g., bailian/qwen-plus, openai/gpt-4o)",
    )
    args = parser.parse_args()
    
    print("\n" + "#"*70)
    print(f"# Skill-based Tool Calling Scenarios - {args.model_uri}")
    print("# Demonstrating Skills across different execution modes")
    print("#"*70)
    
    results = []
    
    # Run all scenarios
    results.append(await scenario_non_streaming(args.model_uri))
    results.append(await scenario_streaming(args.model_uri))
    results.append(await scenario_event_stream(args.model_uri))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Completed: {passed}/{total} scenarios")
    
    if passed == total:
        print("\nAll scenarios completed successfully!")
    else:
        print(f"\n{total - passed} scenario(s) failed. Check logs above.")


if __name__ == "__main__":
    asyncio.run(main())
