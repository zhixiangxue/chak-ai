"""Bridge synchronous provider streams into asyncio without blocking the loop.

Provider SDKs expose streaming responses as *synchronous* iterators whose
``next()`` blocks on network I/O. Iterating one directly inside an async
generator freezes the entire event loop for the duration of the stream: no
other coroutine gets scheduled and ``task.cancel()`` cannot take effect
until the stream is fully consumed.

:func:`iter_in_thread` fixes this by running the factory *and* the whole
iteration in a dedicated worker thread, handing each item back to the event
loop through an ``asyncio.Queue``. The consuming coroutine awaits between
items, which keeps the loop responsive and makes cancellation effective at
any chunk boundary.
"""

import asyncio
import threading
from typing import Any, AsyncIterator, Callable, Iterable, Optional

# Message kinds flowing from the producer thread to the consumer coroutine.
_ITEM = 0
_ERROR = 1
_DONE = 2


async def iter_in_thread(factory: Callable[[], Iterable[Any]]) -> AsyncIterator[Any]:
    """Consume a sync-iterator factory in a worker thread, yielding items asynchronously.

    Args:
        factory: Zero-argument callable returning a synchronous iterable,
            e.g. ``lambda: provider.send(..., stream=True)``. It is invoked
            inside the worker thread so any network I/O it performs (including
            the initial HTTP request of a lazy generator) never touches the
            event loop thread.

    Yields:
        Items produced by the iterator, in arrival order.

    Notes:
        - Exceptions raised by the factory or during iteration are re-raised
          in the consuming coroutine.
        - When the consumer is cancelled or closes the generator early, the
          worker thread stops at the next item boundary and closes the
          underlying iterator, releasing the HTTP connection.
        - The queue is unbounded; LLM chunks are small and arrive at network
          speed, so buffering while the consumer is busy is acceptable.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stopped = threading.Event()

    def _put(kind: int, payload: Any) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))
        except RuntimeError:
            # Event loop already closed — nothing left to deliver to.
            pass

    def _producer() -> None:
        iterator: Optional[Any] = None
        try:
            iterator = iter(factory())
            for item in iterator:
                if stopped.is_set():
                    break
                _put(_ITEM, item)
        except BaseException as exc:  # forwarded to the consumer coroutine
            _put(_ERROR, exc)
        else:
            _put(_DONE, None)
        finally:
            # Close generators / SDK streams so the connection is released
            # even when the consumer abandoned the stream mid-way.
            close = getattr(iterator, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass

    thread = threading.Thread(
        target=_producer, name="chak-stream-bridge", daemon=True
    )
    thread.start()
    try:
        while True:
            kind, payload = await queue.get()
            if kind == _ITEM:
                yield payload
            elif kind == _DONE:
                return
            else:
                raise payload
    finally:
        # Reached on exhaustion, cancellation, or early close: tell the
        # producer to stop at the next chunk boundary.
        stopped.set()
