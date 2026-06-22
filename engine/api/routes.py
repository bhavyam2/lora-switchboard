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
    try:
        output = await scheduler.submit(body.prompt, body.adapter_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return InferenceResponse(output=output, adapter_id=body.adapter_id)


@router.get("/adapters/cached")
async def cached_adapters(request: Request):
    wm = request.app.state.engine.weight_manager
    return {"cached": wm.cached_ids}


@router.post("/adapters/register-random")
async def register_random(adapter_id: str, request: Request):
    wm = request.app.state.engine.weight_manager
    wm.register_random_adapter(adapter_id)
    return {"registered": adapter_id}
