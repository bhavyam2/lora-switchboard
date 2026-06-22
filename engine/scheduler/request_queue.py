import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from engine.config import settings


@dataclass
class InferenceRequest:
    prompt: str
    adapter_id: str
    future: asyncio.Future = field(repr=False)


class RequestScheduler:
    """
    Decouples the FastAPI async event loop from the blocking PyTorch compute
    thread via an asyncio.Queue.

    A single background thread owns the GPU — it dequeues requests one at a
    time, swaps the adapter, runs inference, and resolves the caller's Future.
    This sidesteps the GIL: asyncio handles concurrency on the network side
    while the executor thread serialises all GPU work.
    """

    def __init__(self):
        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(
            maxsize=settings.request_queue_maxsize
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-worker")
        self._engine = None  # set after model is loaded via attach()
        self._running = False

    def attach(self, engine) -> None:
        """Bind the loaded engine (exposes .activate_adapter + .run_inference)."""
        self._engine = engine

    async def submit(self, prompt: str, adapter_id: str) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        request = InferenceRequest(prompt=prompt, adapter_id=adapter_id, future=future)
        await self._queue.put(request)
        return await future

    async def run(self) -> None:
        """Main dispatch loop — run as a background asyncio task."""
        self._running = True
        loop = asyncio.get_running_loop()
        print("[Scheduler] Dispatch loop started.")
        while self._running:
            req = await self._queue.get()
            try:
                result = await loop.run_in_executor(
                    self._executor,
                    self._process,
                    req.prompt,
                    req.adapter_id,
                )
                req.future.set_result(result)
            except Exception as exc:
                req.future.set_exception(exc)
            finally:
                self._queue.task_done()

    def _process(self, prompt: str, adapter_id: str) -> str:
        """Runs in the executor thread — all GPU ops happen here."""
        self._engine.weight_manager.activate(adapter_id)
        return self._engine.run_inference(prompt)

    async def shutdown(self) -> None:
        self._running = False
        self._executor.shutdown(wait=False)
