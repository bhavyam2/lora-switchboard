import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

    def run_inference(self, prompt: str, system_prompt: str | None = None) -> str:
        return self.loader.run_inference(prompt, system_prompt=system_prompt)


def _autoload_adapters(engine: "InferenceEngine", adapter_loader: AdapterLoader) -> None:
    """Load all adapters found in data/adapters/ that contain adapter_config.json."""
    adapters_dir = Path("data/adapters")
    if not adapters_dir.exists():
        return
    for adapter_dir in sorted(adapters_dir.iterdir()):
        if not (adapter_dir / "adapter_config.json").exists():
            continue
        adapter_id = adapter_dir.name
        try:
            weights, metadata = adapter_loader.load_from_dir(adapter_dir)
            engine.weight_manager.register(adapter_id, weights, metadata)
            print(f"[Engine] Auto-loaded adapter: {adapter_id}")
        except Exception as exc:
            print(f"[Engine] Skipping {adapter_id}: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = InferenceEngine()
    adapter_loader = AdapterLoader()

    _autoload_adapters(engine, adapter_loader)

    scheduler = RequestScheduler()
    scheduler.attach(engine)

    app.state.engine = engine
    app.state.scheduler = scheduler
    app.state.adapter_loader = adapter_loader
    app.state.batcher = HeterogeneousBatchRunner(engine.loader, engine.weight_manager)

    dispatch_task = asyncio.create_task(scheduler.run())
    yield
    await scheduler.shutdown()
    dispatch_task.cancel()


app = FastAPI(title="lora-switchboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
