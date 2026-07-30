"""Unit tests for chak.utils.streaming.iter_in_thread.

The bridge must consume a blocking sync iterator in a worker thread so that:
- items arrive in order and errors propagate to the consumer,
- the event loop stays responsive while the iterator blocks between items,
- cancelling the consumer stops the producer and closes the iterator.
"""

import asyncio
import threading
import time

import pytest

from chak.utils.streaming import iter_in_thread


@pytest.mark.asyncio
async def test_yields_items_in_order():
    result = [x async for x in iter_in_thread(lambda: iter([1, 2, 3]))]
    assert result == [1, 2, 3]


@pytest.mark.asyncio
async def test_factory_error_propagates():
    def _factory():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        async for _ in iter_in_thread(_factory):
            pass


@pytest.mark.asyncio
async def test_mid_iteration_error_propagates_after_items():
    def _gen():
        yield 1
        raise RuntimeError("mid-stream")

    seen = []
    with pytest.raises(RuntimeError, match="mid-stream"):
        async for item in iter_in_thread(_gen):
            seen.append(item)
    assert seen == [1]


@pytest.mark.asyncio
async def test_event_loop_stays_responsive_while_iterator_blocks():
    # A sync iterator that blocks 0.15s per item — previously this froze the
    # loop; with the bridge, a heartbeat coroutine must keep ticking.
    def _slow_gen():
        for i in range(4):
            time.sleep(0.15)
            yield i

    ticks = 0

    async def _heartbeat(stop: asyncio.Event):
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.05)
            ticks += 1

    stop = asyncio.Event()
    hb = asyncio.create_task(_heartbeat(stop))
    result = [x async for x in iter_in_thread(_slow_gen)]
    stop.set()
    await hb

    assert result == [0, 1, 2, 3]
    # ~0.6s of blocking iteration should allow ~12 ticks; require a safe floor.
    assert ticks >= 5


@pytest.mark.asyncio
async def test_cancel_stops_producer_and_closes_iterator():
    closed = threading.Event()
    started = threading.Event()

    def _gen():
        try:
            for i in range(1000):
                started.set()
                time.sleep(0.05)
                yield i
        finally:
            # GeneratorExit lands here when the bridge closes the iterator.
            closed.set()

    async def _consume():
        async for _ in iter_in_thread(_gen):
            pass

    task = asyncio.create_task(_consume())
    await asyncio.to_thread(started.wait, 2.0)
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Producer stops at the next item boundary and closes the generator.
    assert await asyncio.to_thread(closed.wait, 2.0)
