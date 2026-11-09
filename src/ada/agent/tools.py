"""Tools the LLM may call (function calling).

KEY DESIGN DECISION: the model never writes or executes arbitrary code. It can only pick
from this whitelist and fill in typed arguments. Reasons:
  * safety  - no sandbox escape, no file/network access from model output;
  * evaluability - a finite action space is loggable and scorable;
  * SLM-friendliness - small models are far better at "choose a tool + 2 args" than at
    writing correct pandas.

Each tool declares its arguments as a pydantic model. We (a) auto-generate the JSON
schema the LLM sees from it, and (b) validate whatever the LLM sends. If validation
fails we return the error TO the model so it can retry - instead of crashing the run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field, ValidationError

from ada import viz
from ada.agent import detectors
from ada.corruption import categorical_cols, numeric_cols


# ----------------------------------------------------------------- session state
@dataclass
class Session:
    """Mutable workspace one agent run operates on. Lives OUTSIDE the LLM message state
    because DataFrames are not JSON-serialisable and must not be pasted into prompts."""

    df: pd.DataFrame
    flagged: pd.DataFrame = None  # cells the agent considers wrong (bool)
    row_scores: pd.Series | None = None
    ts_scores: np.ndarray | None = None
    ts_flags: np.ndarray | None = None
    figures: dict[str, go.Figure] = field(default_factory=dict)
    findings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.flagged is None:
            self.flagged = pd.DataFrame(False, index=self.df.index, columns=self.df.columns)


# ----------------------------------------------------------------- registry
@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Session, Any], dict]

    def openai_schema(self) -> dict:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": schema},
        }


TOOLS: dict[str, ToolSpec] = {}


def tool(name: str, description: str, args_model: type[BaseModel]):
    def deco(fn):
        TOOLS[name] = ToolSpec(name, description, args_model, fn)
        return fn

    return deco


def openai_tool_schemas() -> list[dict]:
    return [t.openai_schema() for t in TOOLS.values()]


def call_tool(session: Session, name: str, arguments: str | dict | None) -> dict:
    """Validate + execute. Never raises: errors become data the model can read."""
    if name not in TOOLS:
        return {"error": f"unknown tool '{name}'. Available: {sorted(TOOLS)}"}
    try:
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments.strip() else {}
        args = TOOLS[name].args_model(**(arguments or {}))
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        return {"error": f"invalid arguments for {name}: {e}"}
    try:
        return TOOLS[name].fn(session, args)
    except Exception as e:  # tool bug or bad data: report, don't crash the graph
        return {"error": f"{type(e).__name__}: {e}"}


# ----------------------------------------------------------------- tools
class NoArgs(BaseModel):
    pass


@tool("profile_dataframe", "Summarise shape, dtypes, missing values and duplicate rows. Call this first.", NoArgs)
def profile_dataframe(s: Session, _: NoArgs) -> dict:
    df = s.df
    out = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": numeric_cols(df),
        "categorical_columns": categorical_cols(df),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v},
        "duplicate_rows": int(df.duplicated().sum()),
    }
    s.findings["profile"] = out
    return out


@tool("drop_duplicates", "Remove exact duplicate rows (keeps first occurrence).", NoArgs)
def drop_duplicates(s: Session, _: NoArgs) -> dict:
    before = len(s.df)
    s.df = s.df.drop_duplicates()
    s.flagged = s.flagged.loc[s.df.index]
    removed = before - len(s.df)
    s.findings["duplicates_removed"] = removed
    return {"removed": removed, "rows_now": len(s.df)}


class OutlierArgs(BaseModel):
    method: Literal["robust_z", "z"] = Field("robust_z", description="robust_z uses median/MAD (recommended).")
    threshold: float = Field(5.0, gt=0, le=50, description="Flag cells whose score exceeds this.")


@tool("detect_outliers", "Flag outlying numeric cells per column (does not modify data).", OutlierArgs)
def detect_outliers(s: Session, a: OutlierArgs) -> dict:
    flags = detectors.flag_outlier_cells(s.df, a.threshold, a.method)
    s.flagged = s.flagged | flags
    n = int(flags.to_numpy().sum())
    s.findings["outlier_cells"] = n
    return {"flagged_cells": n, "by_column": {k: int(v) for k, v in flags.sum().items() if v}}


class TypoArgs(BaseModel):
    min_freq: float = Field(0.01, gt=0, lt=0.5, description="Categories rarer than this fraction are suspicious.")


@tool("detect_category_typos", "Flag categorical cells whose value is suspiciously rare (likely typos).", TypoArgs)
def detect_category_typos(s: Session, a: TypoArgs) -> dict:
    flags = detectors.flag_rare_categories(s.df, a.min_freq)
    s.flagged = s.flagged | flags
    n = int(flags.to_numpy().sum())
    s.findings["typo_cells"] = n
    return {"flagged_cells": n}


@tool(
    "impute_missing",
    "Repair: replace missing and flagged cells with column median (numeric) or mode (categorical).",
    NoArgs,
)
def impute_missing(s: Session, _: NoArgs) -> dict:
    s.flagged = s.flagged | s.df.isna()
    fixed = int(s.flagged.to_numpy().sum())
    s.df = detectors.impute(s.df, s.flagged)
    s.findings["cells_repaired"] = fixed
    return {"cells_repaired": fixed, "missing_left": int(s.df.isna().sum().sum())}


class RowAnomalyArgs(BaseModel):
    contamination: float = Field(0.05, gt=0, lt=0.5, description="Expected fraction of anomalous rows.")


@tool("detect_row_anomalies", "Score every row with an Isolation Forest (higher = more anomalous).", RowAnomalyArgs)
def detect_row_anomalies(s: Session, a: RowAnomalyArgs) -> dict:
    s.row_scores = detectors.isolation_forest_scores(s.df)
    k = max(1, int(round(a.contamination * len(s.df))))
    top = s.row_scores.nlargest(k).index.tolist()
    s.findings["top_anomalous_rows"] = top[:10]
    return {"scored_rows": len(s.df), "top_rows": top[:10]}


class TSArgs(BaseModel):
    column: str = Field("value", description="Numeric column holding the series.")
    period: int = Field(24, ge=2, le=1000, description="Seasonal period in samples (24 for hourly/daily).")
    threshold: float = Field(4.0, gt=0, le=50)


@tool("detect_ts_anomalies", "Seasonal-residual robust z-score anomaly detection for a time series.", TSArgs)
def detect_ts_anomalies(s: Session, a: TSArgs) -> dict:
    if a.column not in s.df.columns:
        return {"error": f"column '{a.column}' not found; columns: {list(s.df.columns)}"}
    s.ts_scores = detectors.seasonal_residual_scores(s.df[a.column].to_numpy(), a.period)
    s.ts_flags = s.ts_scores > a.threshold
    s.findings["ts_flagged_points"] = int(s.ts_flags.sum())
    return {"flagged_points": int(s.ts_flags.sum())}


class PlotArgs(BaseModel):
    kind: Literal["missingness", "histogram", "anomaly_scores", "timeseries"]
    column: str | None = None


@tool("make_plot", "Create a Plotly figure and attach it to the report.", PlotArgs)
def make_plot(s: Session, a: PlotArgs) -> dict:
    if a.kind == "missingness":
        fig = viz.missingness_bar(s.df)
    elif a.kind == "histogram":
        if not a.column or a.column not in s.df.columns:
            return {"error": "histogram needs a valid 'column'"}
        fig = viz.histogram(s.df, a.column)
    elif a.kind == "anomaly_scores":
        if s.row_scores is None:
            return {"error": "run detect_row_anomalies first"}
        fig = viz.anomaly_scores(s.row_scores)
    else:
        if s.ts_flags is None:
            return {"error": "run detect_ts_anomalies first"}
        col = a.column or "value"
        fig = viz.timeseries_with_anomalies(s.df[col].to_numpy(), s.ts_flags)
    key = f"{a.kind}:{a.column or ''}"
    s.figures[key] = fig
    return {"figure": key}
