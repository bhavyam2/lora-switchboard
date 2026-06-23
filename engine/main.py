import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from engine.core.adapter_loader import AdapterLoader
from engine.core.hetero_batcher import HeterogeneousBatchRunner
from engine.core.model_loader import BaseModelLoader
from engine.core.weight_manager import WeightManager
from engine.scheduler.request_queue import RequestScheduler
from engine.api.routes import router


class InferenceEngine:
    """Thin wrapper that binds the loader, weight manager, and scheduler."""

    def __init__(self):
        self.loader = BaseModelLoader()
        self.weight_manager = WeightManager(
            lora_layers=self.loader.lora_layers,
            device=self.loader.device,
        )

    def run_inference(self, prompt: str) -> str:
        return self.loader.run_inference(prompt)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = InferenceEngine()
    scheduler = RequestScheduler()
    scheduler.attach(engine)

    app.state.engine = engine
    app.state.scheduler = scheduler
    app.state.adapter_loader = AdapterLoader()
    app.state.batcher = HeterogeneousBatchRunner(engine.loader, engine.weight_manager)

    dispatch_task = asyncio.create_task(scheduler.run())
    yield
    await scheduler.shutdown()
    dispatch_task.cancel()


app = FastAPI(title="lora-switchboard", lifespan=lifespan)
app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
