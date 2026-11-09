"""Tests double as executable documentation: each one states a property we rely on."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ada.agent.graph import run_agent
from ada.agent.llm import LLMResponse, ScriptedClient, ToolCall
from ada.agent.tools import Session, call_tool, openai_tool_schemas
from ada.api import app
from ada.corruption import CorruptionConfig, corrupt_tabular, inject_ts_anomalies
from ada.datasets import load_dataset, load_registry
from ada.eval import metrics as M
from ada.eval.runner import run_experiment
from ada.eval.stats import bootstrap_ci


# ---------------------------------------------------------------- data
def test_registry_has_licenses_and_unique_names():
    reg = load_registry()
    assert all(s.license for s in reg.values())
    assert "wine" in reg


def test_dataset_loading_is_deterministic():
    a, b = load_dataset("synthetic_tabular").df, load_dataset("synthetic_tabular").df
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------- corruption
CFG = CorruptionConfig(missing_rate=0.1, outlier_rate=0.05, typo_rate=0.05, duplicate_rate=0.05)


def test_same_seed_same_dirt_different_seed_different_dirt():
    X = load_dataset("synthetic_tabular").features
    r1, r2, r3 = (corrupt_tabular(X, CFG, s) for s in (1, 1, 2))
    pd.testing.assert_frame_equal(r1.dirty, r2.dirty)
    assert not r1.error_mask.equals(r3.error_mask)


def test_error_mask_is_exact():
    X = load_dataset("synthetic_tabular").features
    r = corrupt_tabular(X, CFG, 0)
    d = r.dirty.loc[X.index]
    changed = ~((d == X) | (d.isna() & X.isna()))  # cells that differ from truth
    assert (changed.to_numpy() == r.error_mask.to_numpy()).all()
    assert len(r.duplicate_rows) == round(0.05 * len(X))


def test_ts_labels_match_injection():
    base = np.sin(np.arange(2000) / 5)
    dirty, labels = inject_ts_anomalies(base, 5, seed=0)
    assert labels.any() and (dirty[~labels] == base[~labels]).all()


# ---------------------------------------------------------------- metrics (hand-computed)
def test_prf():
    m = M.prf([1, 1, 0, 0], [1, 0, 1, 0])
    assert m == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_event_f1_counts_events_not_points():
    truth = np.array([0, 1, 1, 1, 0, 0, 1, 0], bool)
    pred = np.array([0, 0, 1, 0, 0, 1, 0, 0], bool)  # hits event 1, misses event 2, one false alarm
    m = M.event_f1(truth, pred)
    assert m["event_recall"] == 0.5 and m["event_precision"] == 0.5


def test_auroc_perfect():
    assert M.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_claim_verification_catches_wrong_numbers():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    r = M.verify_numeric_claims(df, [{"stat": "mean", "column": "a", "value": 2.0},
                                     {"stat": "max", "column": "a", "value": 99.0}])
    assert r["faithfulness"] == 0.5


def test_bootstrap_ci_contains_mean():
    v = np.random.default_rng(0).normal(5, 1, 50)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi


# ---------------------------------------------------------------- tools
def test_tool_schemas_are_generated_from_pydantic():
    names = {t["function"]["name"] for t in openai_tool_schemas()}
    assert {"profile_dataframe", "detect_outliers", "impute_missing"} <= names


def test_bad_tool_args_return_error_instead_of_crashing():
    s = Session(df=pd.DataFrame({"a": [1.0, 2.0]}))
    assert "error" in call_tool(s, "detect_outliers", {"threshold": -3})
    assert "error" in call_tool(s, "no_such_tool", {})
    assert "error" in call_tool(s, "detect_outliers", "{not json")


# ---------------------------------------------------------------- agent graph
def test_agent_cleans_data_end_to_end():
    X = load_dataset("synthetic_tabular").features
    r = corrupt_tabular(X, CFG, 0)
    s = Session(df=r.dirty.copy())
    out = run_agent("Clean this dataset", s, ScriptedClient())
    assert out.stop_reason == "model_finished"
    assert s.df.isna().sum().sum() == 0
    assert len(s.df) == len(X)  # all duplicates removed


class LoopingClient:
    """A broken model that never stops calling tools."""

    def chat(self, messages, tools):
        return LLMResponse(tool_calls=[ToolCall(id="x", name="profile_dataframe", arguments={})])


def test_step_budget_stops_runaway_agent():
    s = Session(df=pd.DataFrame({"a": [1.0]}))
    out = run_agent("loop", s, LoopingClient(), max_steps=3)
    assert out.stop_reason == "max_steps" and out.steps == 3


# ---------------------------------------------------------------- experiment + API
def test_smoke_experiment_and_agent_beats_naive_on_timeseries(tmp_path):
    out = run_experiment("experiments/smoke.yaml", tmp_path)
    summary = pd.read_csv(out / "summary.csv")
    for f in ["results.csv", "paired.csv", "env.json", "report.html", "summary.md"]:
        assert (out / f).exists()
    ts = summary[(summary.suite == "timeseries") & (summary.metric == "auprc")].set_index("method")["mean"]
    assert ts["agent"] > ts["naive_zscore"]
    cl = summary[(summary.suite == "cleaning") & (summary.metric == "over_cleaning_rate")].set_index("method")["mean"]
    assert cl["oracle"] == 0 and cl["no_cleaning"] == 0


def test_api():
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert len(c.get("/datasets").json()) >= 4
    r = c.post("/analyze", json={"dataset": "wine"})
    assert r.status_code == 200 and r.json()["steps"] == 6 and r.json()["figures"]
    assert c.post("/analyze", json={}).status_code == 422
    assert c.post("/analyze", json={"csv": "a,b\n1,2\n3,4\n"}).status_code == 200
