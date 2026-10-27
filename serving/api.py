"""
FastAPI wrapper around a vLLM-served model.

Merge the adapter first (vLLM cannot serve a bare adapter directory), then
start the vLLM server:
    python -m quantization.merge --model outputs/finetuned
    python -m vllm.entrypoints.openai.api_server \
        --model outputs/finetuned_merged --dtype half \
        --host 0.0.0.0 --port 8001

Then start this API:
    uvicorn serving.api:app --host 0.0.0.0 --port 8000
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")

# Must match the training format (data/dataset_utils.py ALPACA_TEMPLATE); the
# model was fine-tuned on this raw prompt, not on chat messages. A test in
# tests/test_api.py fails if the two drift apart.
PROMPT_TEMPLATE = (
    "Below is a medical question. Answer it accurately and concisely.\n\n"
    "### Instruction:\n{question}\n\n"
    "### Input:\n{choices}\n\n"
    "### Response:\n"
)

# Stop if the model starts another "### ..." section instead of ending.
STOP_SEQUENCES = ["\n###"]

# Model name cached at startup — avoids a blocking sync call on every request
_model_name: str = "unknown"


async def _refresh_model_name() -> str:
    """Fetch the served model id from vLLM. Safe to call repeatedly."""
    global _model_name
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{VLLM_BASE_URL}/models")
            if r.status_code == 200:
                models = r.json().get("data", [])
                if models:
                    _model_name = models[0]["id"]
    except Exception:
        pass
    return _model_name


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Best-effort at startup; if vLLM isn't up yet, /infer retries lazily.
    await _refresh_model_name()
    yield


app = FastAPI(title="ModelEdge Inference API", version="1.0.0", lifespan=lifespan)


class InferenceRequest(BaseModel):
    question: str = Field(..., min_length=5)
    choices: Optional[list[str]] = Field(default=None, max_length=10)
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

    # If vLLM was unreachable at startup, try once more now.
    model_name = _model_name
    if model_name == "unknown":
        model_name = await _refresh_model_name()

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stop": STOP_SEQUENCES,
    }

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{VLLM_BASE_URL}/completions", json=payload)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"vLLM error: {e}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"vLLM unreachable: {e}")

    elapsed_ms = (time.perf_counter() - start) * 1000
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="vLLM returned empty choices")
    answer = choices[0]["text"].strip()
    model_id = data.get("model", _model_name)

    return InferenceResponse(answer=answer, latency_ms=elapsed_ms, model=model_id)


def _build_prompt(question: str, choices: Optional[list[str]]) -> str:
    opts = ""
    if choices:
        labels = "ABCDEFGHIJ"
        opts = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))
    return PROMPT_TEMPLATE.format(question=question, choices=opts)
