import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class InferenceRequest(BaseModel):
    prompt: str
    adapter_id: str


class InferenceResponse(BaseModel):
    output: str
    adapter_id: str


@router.post("/infer", response_model=InferenceResponse)
async def infer(body: InferenceRequest, request: Request):
    scheduler = request.app.state.scheduler
    wm = request.app.state.engine.weight_manager
    system_prompt = wm.get_system_prompt(body.adapter_id)
    try:
        output = await scheduler.submit(body.prompt, body.adapter_id, system_prompt)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return InferenceResponse(output=output, adapter_id=body.adapter_id)


@router.get("/adapters/cached")
async def cached_adapters(request: Request):
    wm = request.app.state.engine.weight_manager
    return {"cached": wm.cached_ids}


@router.get("/adapters/list")
async def list_adapters(request: Request):
    wm = request.app.state.engine.weight_manager
    return {
        "adapters": [
            {"id": aid, **wm.get_metadata(aid)}
            for aid in wm.cached_ids
        ]
    }


@router.post("/adapters/register-random")
async def register_random(adapter_id: str, request: Request):
    wm = request.app.state.engine.weight_manager
    wm.register_random_adapter(adapter_id)
    return {"registered": adapter_id}


class HubLoadRequest(BaseModel):
    adapter_id: str
    hub_repo_id: str


class DirLoadRequest(BaseModel):
    adapter_id: str
    path: str


@router.post("/adapters/load-from-hub")
async def load_from_hub(body: HubLoadRequest, request: Request):
    loader = request.app.state.adapter_loader
    wm = request.app.state.engine.weight_manager
    try:
        weights, metadata = loader.load_from_hub(body.hub_repo_id)
        wm.register(body.adapter_id, weights, metadata)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"registered": body.adapter_id, "source": body.hub_repo_id}


@router.post("/adapters/load-from-dir")
async def load_from_dir(body: DirLoadRequest, request: Request):
    loader = request.app.state.adapter_loader
    wm = request.app.state.engine.weight_manager
    try:
        weights, metadata = loader.load_from_dir(Path(body.path))
        wm.register(body.adapter_id, weights, metadata)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"registered": body.adapter_id, "source": body.path}


class BatchInferRequest(BaseModel):
    requests: list[InferenceRequest]
    max_new_tokens: int = 50


class BatchInferResponse(BaseModel):
    outputs: list[InferenceResponse]
    batch_size: int


@router.post("/batch-infer", response_model=BatchInferResponse)
async def batch_infer(body: BatchInferRequest, request: Request):
    batcher = request.app.state.batcher
    scheduler = request.app.state.scheduler
    loop = asyncio.get_running_loop()

    pairs = [(r.prompt, r.adapter_id) for r in body.requests]
    try:
        outputs = await loop.run_in_executor(
            scheduler._executor,
            lambda: batcher.run_batch(pairs, body.max_new_tokens),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return BatchInferResponse(
        outputs=[
            InferenceResponse(output=out, adapter_id=req.adapter_id)
            for out, req in zip(outputs, body.requests)
        ],
        batch_size=len(body.requests),
    )
