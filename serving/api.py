"""
FastAPI wrapper around a vLLM-served model.

Start the vLLM server first:
    python -m vllm.entrypoints.openai.api_server \
        --model outputs/finetuned_int4_awq \
        --host 0.0.0.0 --port 8001

Then start this API:
    uvicorn serving.api:app --host 0.0.0.0 --port 8000
"""

import time
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="ModelEdge Inference API", version="1.0.0")

VLLM_BASE_URL = "http://localhost:8001/v1"

SYSTEM_INSTRUCTION = (
    "You are a knowledgeable medical assistant. "
    "Answer accurately and state uncertainty when present."
)


class InferenceRequest(BaseModel):
    question: str = Field(..., min_length=5)
    choices: Optional[list[str]] = None
    max_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class InferenceResponse(BaseModel):
    answer: str
    latency_ms: float
    model: str


class HealthResponse(BaseModel):
    status: str
    vllm_reachable: bool


@app.get("/health", response_model=HealthResponse)
async def health():
    reachable = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{VLLM_BASE_URL}/models")
            reachable = r.status_code == 200
    except Exception:
        pass
    return HealthResponse(status="ok", vllm_reachable=reachable)


@app.post("/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest):
    prompt = _build_prompt(req.question, req.choices)

    payload = {
        "model": _get_model_name(),
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{VLLM_BASE_URL}/chat/completions", json=payload)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"vLLM error: {e}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"vLLM unreachable: {e}")

    elapsed_ms = (time.perf_counter() - start) * 1000
    data = resp.json()
    answer = data["choices"][0]["message"]["content"].strip()
    model_id = data.get("model", "unknown")

    return InferenceResponse(answer=answer, latency_ms=elapsed_ms, model=model_id)


def _build_prompt(question: str, choices: Optional[list[str]]) -> str:
    if choices:
        labels = "ABCDE"
        opts = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))
        return f"{question}\n\nOptions:\n{opts}"
    return question


def _get_model_name() -> str:
    try:
        import httpx as _httpx

        r = _httpx.get(f"{VLLM_BASE_URL}/models", timeout=3)
        if r.status_code == 200:
            models = r.json().get("data", [])
            if models:
                return models[0]["id"]
    except Exception:
        pass
    return "unknown"
