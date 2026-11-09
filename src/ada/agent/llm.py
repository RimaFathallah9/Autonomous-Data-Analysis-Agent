"""LLM abstraction.

The graph only depends on `LLMClient.chat(messages, tools) -> LLMResponse`. Two backends:

* OpenAICompatibleClient - anything speaking the OpenAI chat+tools protocol:
  OpenAI, Azure OpenAI, and LOCAL small models via Ollama / vLLM / LM Studio / llama.cpp
  (just change base_url + model). This is how you plug in an SLM.
* ScriptedClient - a deterministic "fake LLM" that follows a fixed plan. It lets us test
  the whole graph, the tools and the evaluation pipeline offline, for free, with zero
  randomness. It is ALSO the control group: if a real LLM can't beat the scripted policy,
  the LLM isn't adding value.

Selection via env vars: ADA_LLM_PROVIDER=scripted|openai, ADA_LLM_MODEL, ADA_LLM_BASE_URL,
OPENAI_API_KEY (use any string for local servers).
"""

from __future__ import annotations

import json
import os
from typing import Protocol

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class OpenAICompatibleClient:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, temperature: float = 0.0):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "none"))
        self.model = model
        self.temperature = temperature  # 0 => as deterministic as the provider allows

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools, temperature=self.temperature
        )
        msg = r.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}  # tools layer will report missing/invalid args back to the model
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return LLMResponse(content=msg.content, tool_calls=calls)


PLANS: dict[str, list[tuple[str, dict]]] = {
    "clean": [
        ("profile_dataframe", {}),
        ("drop_duplicates", {}),
        ("detect_outliers", {"method": "robust_z", "threshold": 5.0}),
        ("detect_category_typos", {}),
        ("impute_missing", {}),
        ("make_plot", {"kind": "missingness"}),
    ],
    "row_anomaly": [
        ("profile_dataframe", {}),
        ("detect_row_anomalies", {"contamination": 0.05}),
        ("make_plot", {"kind": "anomaly_scores"}),
    ],
    "timeseries": [
        ("detect_ts_anomalies", {"column": "value", "period": 24, "threshold": 4.0}),
        ("make_plot", {"kind": "timeseries", "column": "value"}),
    ],
}


def infer_task(text: str) -> str:
    t = text.lower()
    if "time series" in t or "timeseries" in t:
        return "timeseries"
    if "anomal" in t:
        return "row_anomaly"
    return "clean"


class ScriptedClient:
    """Deterministic policy: executes PLANS[task] one tool per turn, then summarises."""

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        user = next(m["content"] for m in messages if m["role"] == "user")
        plan = PLANS[infer_task(user)]
        done = sum(1 for m in messages if m["role"] == "assistant" and m.get("tool_calls"))
        if done < len(plan):
            name, args = plan[done]
            return LLMResponse(tool_calls=[ToolCall(id=f"call_{done}", name=name, arguments=args)])
        results = [m["content"] for m in messages if m["role"] == "tool"]
        return LLMResponse(content=f"Completed {len(plan)} steps. Last tool results: {results[-3:]}")


def make_client() -> LLMClient:
    provider = os.environ.get("ADA_LLM_PROVIDER", "scripted").lower()
    if provider == "scripted":
        return ScriptedClient()
    if provider == "openai":
        return OpenAICompatibleClient(
            model=os.environ.get("ADA_LLM_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get("ADA_LLM_BASE_URL") or None,
        )
    raise ValueError(f"Unknown ADA_LLM_PROVIDER '{provider}'")
