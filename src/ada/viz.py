"""Plotly figures. Each function returns a go.Figure so the same code feeds
notebooks, saved HTML reports and the FastAPI JSON response."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def missingness_bar(df: pd.DataFrame) -> go.Figure:
    pct = df.isna().mean().mul(100).sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=pct.index.astype(str), y=pct.values))
    fig.update_layout(title="Missing values per column", yaxis_title="% missing")
    return fig


def histogram(df: pd.DataFrame, column: str) -> go.Figure:
    fig = go.Figure(go.Histogram(x=df[column]))
    fig.update_layout(title=f"Distribution of {column}", xaxis_title=column)
    return fig


def anomaly_scores(scores: pd.Series, title: str = "Anomaly scores") -> go.Figure:
    fig = go.Figure(go.Scatter(x=list(range(len(scores))), y=scores.values, mode="markers"))
    fig.update_layout(title=title, xaxis_title="row", yaxis_title="score")
    return fig


def timeseries_with_anomalies(values: np.ndarray, flags: np.ndarray, x=None) -> go.Figure:
    x = np.arange(len(values)) if x is None else np.asarray(x)
    fig = go.Figure(go.Scatter(x=x, y=values, mode="lines", name="value"))
    fig.add_trace(
        go.Scatter(x=x[flags], y=values[flags], mode="markers", name="flagged", marker=dict(color="red", size=7))
    )
    fig.update_layout(title="Time series with flagged anomalies")
    return fig


def metric_bars(summary: pd.DataFrame, title: str) -> go.Figure:
    """summary columns: metric, method, mean, ci_low, ci_high -> grouped bars with 95% CI whiskers."""
    fig = go.Figure()
    for method, g in summary.groupby("method"):
        fig.add_trace(
            go.Bar(
                name=str(method),
                x=g["metric"],
                y=g["mean"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(g["ci_high"] - g["mean"]).to_numpy(),
                    arrayminus=(g["mean"] - g["ci_low"]).to_numpy(),
                ),
            )
        )
    fig.update_layout(title=title, barmode="group", yaxis_title="value (mean, 95% bootstrap CI)")
    return fig
