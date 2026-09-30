"""FastAPI inference server: base model + LoRA adapter (or a merged model).

    ADE_BASE=Qwen/Qwen2.5-1.5B-Instruct ADE_ADAPTER=outputs/qwen2.5-1.5b-ade/adapter \
        uvicorn ade_ft.serve:app --port 8000
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .evaluate import generate
from .model import load_for_inference, load_tokenizer
from .schema import parse_output

app = FastAPI(title="Clinical ADE extractor", version="0.1.0")


class ExtractRequest(BaseModel):
    note: str = Field(min_length=10, max_length=8000)


class ExtractResponse(BaseModel):
    events: list[dict]
    valid_json: bool
    raw: str


@lru_cache(maxsize=1)
def _load():
    base = os.environ.get("ADE_BASE")
    if not base:
        raise RuntimeError("set ADE_BASE (and optionally ADE_ADAPTER)")
    adapter = os.environ.get("ADE_ADAPTER") or None
    tok = load_tokenizer(adapter or base)
    return load_for_inference(base, adapter), tok


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "base": os.environ.get("ADE_BASE"), "adapter": os.environ.get("ADE_ADAPTER")}


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    try:
        model, tok = _load()
    except Exception as err:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"model not loaded: {err}") from err
    raw = generate(model, tok, [req.note])[0]
    parsed = parse_output(raw)
    return ExtractResponse(
        events=[e.model_dump() for e in parsed.events] if parsed else [], valid_json=parsed is not None, raw=raw
    )
