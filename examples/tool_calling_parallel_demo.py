"""
Parallel Tool Calling Demo

Demonstrates tool execution with different executor modes:
1. IO-bound tasks (async functions) - always concurrent
2. CPU-bound tasks (sync functions) - depends on executor choice
"""

import asyncio
import os
import time
from datetime import datetime

import dotenv

dotenv.load_dotenv()

import chak


# ============================================================
# IO-bound tasks (async functions)
# ============================================================

async def io_task_1(seconds: int = 2) -> str:
    """
    Simulates IO-bound task (e.g., API call, database query).
    Uses async sleep - always concurrent.
    """
    start = datetime.now()
    print(f"[IO Task 1] Started at {start.strftime('%H:%M:%S.%f')[:-3]}")
    await asyncio.sleep(seconds)
    end = datetime.now()
    print(f"[IO Task 1] Finished at {end.strftime('%H:%M:%S.%f')[:-3]}")
    return f"IO Task 1 completed at {end.strftime('%H:%M:%S.%f')[:-3]}"


async def io_task_2(seconds: int = 2) -> str:
    """Simulates IO-bound task."""
    start = datetime.now()
    print(f"[IO Task 2] Started at {start.strftime('%H:%M:%S.%f')[:-3]}")
    await asyncio.sleep(seconds)
    end = datetime.now()
    print(f"[IO Task 2] Finished at {end.strftime('%H:%M:%S.%f')[:-3]}")
    return f"IO Task 2 completed at {end.strftime('%H:%M:%S.%f')[:-3]}"


async def io_task_3(seconds: int = 2) -> str:
    """Simulates IO-bound task."""
    start = datetime.now()
    print(f"[IO Task 3] Started at {start.strftime('%H:%M:%S.%f')[:-3]}")
    await asyncio.sleep(seconds)
    end = datetime.now()
    print(f"[IO Task 3] Finished at {end.strftime('%H:%M:%S.%f')[:-3]}")
    return f"IO Task 3 completed at {end.strftime('%H:%M:%S.%f')[:-3]}"


# ============================================================
# CPU-bound tasks (sync functions)
# ============================================================

def cpu_task_1(n: int = 10000000) -> str:
    """
    Simulates CPU-bound task (e.g., heavy computation).
    Sync function - executor matters!
    """
    start = datetime.now()
    print(f"[CPU Task 1] Started at {start.strftime('%H:%M:%S.%f')[:-3]}")
    
    # Heavy computation
    result = sum(i * i for i in range(n))
    
    end = datetime.now()
    print(f"[CPU Task 1] Finished at {end.strftime('%H:%M:%S.%f')[:-3]}")
    return f"CPU Task 1 completed at {end.strftime('%H:%M:%S.%f')[:-3]}, result={result}"


def cpu_task_2(n: int = 10000000) -> str:
    """Simulates CPU-bound task."""
    start = datetime.now()
    print(f"[CPU Task 2] Started at {start.strftime('%H:%M:%S.%f')[:-3]}")
    
    result = sum(i * i for i in range(n))
    
    end = datetime.now()
    print(f"[CPU Task 2] Finished at {end.strftime('%H:%M:%S.%f')[:-3]}")
    return f"CPU Task 2 completed at {end.strftime('%H:%M:%S.%f')[:-3]}, result={result}"


def cpu_task_3(n: int = 10000000) -> str:
    """Simulates CPU-bound task."""
    start = datetime.now()
    print(f"[CPU Task 3] Started at {start.strftime('%H:%M:%S.%f')[:-3]}")
    
    result = sum(i * i for i in range(n))
    
    end = datetime.now()
    print(f"[CPU Task 3] Finished at {end.strftime('%H:%M:%S.%f')[:-3]}")
    return f"CPU Task 3 completed at {end.strftime('%H:%M:%S.%f')[:-3]}, result={result}"


async def test_io_bound():
    """Test IO-bound tasks (always concurrent)"""
    print("\n" + "=" * 70)
    print("TEST 1: IO-BOUND TASKS (Async Functions)")
    print("=" * 70)
    print("Expected: PARALLEL (~2s) - async functions are always concurrent")
    print("Note: Total time includes LLM processing, check tool timestamps")
    print("-" * 70)
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not set")
        return
    
    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        tools=[io_task_1, io_task_2, io_task_3]
    )
    
    start_time = time.time()
    response = await conv.asend(
        "Please call io_task_1, io_task_2, and io_task_3 with 2 seconds parameter."
    )
    elapsed = time.time() - start_time
    
    print()
    print("-" * 70)
    print(f"Total time: {elapsed:.2f}s (includes LLM calls)")
    print("Check timestamps above: tasks should start/finish together")
    print("Expected: ~2s for tool execution + LLM processing time")
    print("=" * 70)


async def test_cpu_bound_asyncio():
    """Test CPU-bound tasks with default asyncio executor"""
    print("\n" + "=" * 70)
    print("TEST 2: CPU-BOUND TASKS with ASYNCIO (default)")
    print("=" * 70)
    print("Expected: SERIAL-like (~3-4s) - GIL prevents true parallelism")
    print("-" * 70)
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not set")
        return
    
    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        tools=[cpu_task_1, cpu_task_2, cpu_task_3],
        tool_executor=chak.ToolExecutor.ASYNCIO  # Default
    )
    
    start_time = time.time()
    response = await conv.asend(
        "Please call cpu_task_1, cpu_task_2, and cpu_task_3 with n=10000000."
    )
    elapsed = time.time() - start_time
    
    print()
    print("-" * 70)
    print(f"Total time: {elapsed:.2f}s (includes LLM calls)")
    print("Check timestamps: CPU tasks should take ~1s each when serial")
    if elapsed > 6:
        print("Result: SERIAL-like - Expected (GIL limitation)")
    else:
        print("Result: Unexpectedly fast")
    print("=" * 70)


async def test_cpu_bound_thread():
    """Test CPU-bound tasks with ThreadPoolExecutor"""
    print("\n" + "=" * 70)
    print("TEST 3: CPU-BOUND TASKS with THREAD executor")
    print("=" * 70)
    print("Expected: SERIAL-like (~3-4s) - GIL still prevents parallelism")
    print("-" * 70)
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not set")
        return
    
    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        tools=[cpu_task_1, cpu_task_2, cpu_task_3],
        tool_executor=chak.ToolExecutor.THREAD
    )
    
    start_time = time.time()
    response = await conv.asend(
        "Please call cpu_task_1, cpu_task_2, and cpu_task_3 with n=10000000."
    )
    elapsed = time.time() - start_time
    
    print()
    print("-" * 70)
    print(f"Total time: {elapsed:.2f}s (includes LLM calls)")
    print("Check timestamps: should be similar to ASYNCIO test")
    if elapsed > 6:
        print("Result: SERIAL-like - Expected (GIL limitation)")
    else:
        print("Result: Unexpectedly fast")
    print("=" * 70)


async def test_cpu_bound_process():
    """Test CPU-bound tasks with ProcessPoolExecutor"""
    print("\n" + "=" * 70)
    print("TEST 4: CPU-BOUND TASKS with PROCESS executor")
    print("=" * 70)
    print("Expected: PARALLEL (~1-2s) - True parallelism without GIL")
    print("-" * 70)
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not set")
        return
    
    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        tools=[cpu_task_1, cpu_task_2, cpu_task_3],
        tool_executor=chak.ToolExecutor.PROCESS
    )
    
    start_time = time.time()
    response = await conv.asend(
        "Please call cpu_task_1, cpu_task_2, and cpu_task_3 with n=10000000."
    )
    elapsed = time.time() - start_time
    
    print()
    print("-" * 70)
    print(f"Total time: {elapsed:.2f}s (includes LLM calls)")
    print("Check timestamps: tasks should start/finish together")
    if elapsed < 6:
        print("Result: PARALLEL - Process pool works!")
    else:
        print("Result: SERIAL-like - Process pool overhead or issue")
    print("=" * 70)


async def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("TOOL EXECUTOR COMPARISON DEMO")
    print("=" * 70)
    
    # Test 1: IO-bound (async) - always works
    await test_io_bound()
    
    # Test 2: CPU-bound with asyncio - GIL limitation
    await test_cpu_bound_asyncio()
    
    # Test 3: CPU-bound with thread - GIL limitation
    await test_cpu_bound_thread()
    
    # Test 4: CPU-bound with process - true parallelism
    await test_cpu_bound_process()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("IO-bound (async):     Always concurrent")
    print("CPU-bound + ASYNCIO:  Serial-like (GIL)")
    print("CPU-bound + THREAD:   Serial-like (GIL)")
    print("CPU-bound + PROCESS:  True parallel (no GIL)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
