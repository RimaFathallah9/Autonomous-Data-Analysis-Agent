"""FastAPI service wrapping the agent.

WHY an API? It is the boundary where the agent becomes a product: typed request/response
schemas (pydantic), automatic OpenAPI docs at /docs, and the thing Docker exposes.
Endpoints are plain `def` (not async) because pandas/sklearn are CPU-bound; FastAPI runs
sync endpoints in a thread pool so they don't block the event loop.
"""

from __future__ import annotations

import io
import json
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from ada.agent.graph import run_agent
from ada.agent.llm import make_client
from ada.agent.tools import Session
from ada.datasets import load_dataset, load_registry

MAX_CSV_BYTES = int(os.environ.get("ADA_MAX_CSV_BYTES", 5_000_000))
MAX_ROWS = int(os.environ.get("ADA_MAX_ROWS", 100_000))

app = FastAPI(title="Autonomous Data-Analysis Agent", version="0.1.0")


class AnalyzeRequest(BaseModel):
    task: str = Field("Clean this dataset: find and repair missing values, duplicate rows, outliers and typos.")
    dataset: str | None = Field(None, description="Name from the dataset registry.")
    csv: str | None = Field(None, description="Raw CSV text (alternative to `dataset`).")
    max_steps: int = Field(12, ge=1, le=30)


class AnalyzeResponse(BaseModel):
    final_message: str | None
    stop_reason: str
    steps: int
    tool_calls: list[dict]
    findings: dict
    cleaned_preview: list[dict]
    figures: dict[str, dict]


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_provider": os.environ.get("ADA_LLM_PROVIDER", "scripted")}


@app.get("/datasets")
def datasets() -> list[dict]:
    return [
        {"name": s.name, "kind": s.kind, "license": s.license, "task_tags": s.task_tags}
        for s in load_registry().values()
    ]


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if bool(req.dataset) == bool(req.csv):
        raise HTTPException(422, "Provide exactly one of `dataset` or `csv`.")
    if req.dataset:
        try:
            df = load_dataset(req.dataset).features if req.dataset != "synthetic_timeseries" else load_dataset(req.dataset).df
        except KeyError:
            raise HTTPException(404, f"Unknown dataset '{req.dataset}'")
    else:
        if len(req.csv.encode()) > MAX_CSV_BYTES:  # untrusted input: cap it
            raise HTTPException(413, "CSV too large")
        try:
            df = pd.read_csv(io.StringIO(req.csv))
        except Exception as e:
            raise HTTPException(422, f"Could not parse CSV: {e}")
    if len(df) > MAX_ROWS:
        raise HTTPException(413, f"Too many rows (max {MAX_ROWS})")

    session = Session(df=df.copy())
    result = run_agent(req.task, session, make_client(), req.max_steps)
    calls = [
        {"tool": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"])}
        for m in result.trace
        for tc in m.get("tool_calls", [])
    ]
    return AnalyzeResponse(
        final_message=result.final_message,
        stop_reason=result.stop_reason,
        steps=result.steps,
        tool_calls=calls,
        findings=json.loads(json.dumps(session.findings, default=str)),
        cleaned_preview=json.loads(session.df.head(5).to_json(orient="records")),
        figures={k: json.loads(f.to_json()) for k, f in session.figures.items()},
    )
