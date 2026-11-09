"""Config-driven experiment runner.

    config.yaml --> for each suite x dataset x condition x seed x method --> long-format rows
    --> results.csv (raw) + summary.csv (mean, 95% CI) + paired.csv (tests) + report.html

Every run directory also stores the exact config, library versions, git commit and data
fingerprints so the numbers can be traced back to what produced them.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from ada import viz
from ada.agent import detectors
from ada.agent.graph import run_agent
from ada.agent.llm import LLMClient, make_client
from ada.agent.tools import Session
from ada.corruption import CorruptionConfig, CorruptionResult, corrupt_tabular, inject_ts_anomalies, numeric_cols
from ada.datasets import load_dataset
from ada.datasets.registry import REPO_ROOT
from ada.eval import metrics as M
from ada.eval.stats import paired_comparison, summarize

Rows = list[dict]


def _add(rows: Rows, base: dict, metrics: dict[str, float]) -> None:
    for k, v in metrics.items():
        rows.append({**base, "metric": k, "value": float(v)})


# ================================================================== cleaning suite
def _no_cleaning(res: CorruptionResult):
    return res.dirty, res.dirty.isna()


def _naive_cleaning(res: CorruptionResult):
    """Textbook baseline: mean/std z-score>3, rare categories, median/mode fill. No dedupe."""
    d = res.dirty
    flags = detectors.flag_outlier_cells(d, 3.0, "z") | detectors.flag_rare_categories(d, 0.01) | d.isna()
    return detectors.impute(d, flags), flags


def _oracle(res: CorruptionResult):
    return res.clean, res.error_mask


def _agent_cleaning(res: CorruptionResult, client: LLMClient):
    s = Session(df=res.dirty.copy())
    run_agent("Clean this dataset: find and repair missing values, duplicate rows, outliers and typos.", s, client)
    return s.df, s.flagged


def _score_cleaning(res: CorruptionResult, repaired, flagged, ytr, Xte, yte) -> dict[str, float]:
    idx = repaired.index.intersection(res.clean.index)
    truth = res.error_mask.loc[idx].to_numpy().ravel()
    pred = flagged.loc[idx].to_numpy().ravel()
    out = {f"detect_{k}": v for k, v in M.prf(truth, pred).items()}
    out["repair_nrmse"] = M.repair_nrmse(res.clean, repaired, res.error_mask)
    out["repair_cat_acc"] = M.repair_cat_acc(res.clean, repaired, res.error_mask)
    out["over_cleaning_rate"] = M.over_cleaning_rate(res.dirty, repaired, res.error_mask)
    if len(res.duplicate_rows):
        left = len(repaired.index.intersection(res.duplicate_rows))
        out["dup_removed_frac"] = 1 - left / len(res.duplicate_rows)
    if ytr is not None:
        y_aligned = ytr.loc[res.source_index.loc[repaired.index].to_numpy()].reset_index(drop=True)
        out["downstream_acc"] = M.downstream_accuracy(repaired.reset_index(drop=True), y_aligned, Xte, yte)
    return out


def run_cleaning(suite: dict, seeds: list[int], client: LLMClient, rows: Rows, fingerprints: dict) -> None:
    for name in suite["datasets"]:
        ld = load_dataset(name)
        X, y = ld.features, ld.target
        fingerprints[name] = int(pd.util.hash_pandas_object(ld.df, index=True).sum() % (2**63))
        for cond_name, cond in suite["conditions"].items():
            for seed in seeds:
                if y is not None:
                    tr_idx, te_idx = M.holdout_split(y, seed)
                    Xtr, ytr, Xte, yte = X.loc[tr_idx], y.loc[tr_idx], X.loc[te_idx], y.loc[te_idx]
                else:
                    Xtr, ytr, Xte, yte = X, None, None, None
                res = corrupt_tabular(Xtr, CorruptionConfig(**cond), seed)
                methods = {
                    "no_cleaning": lambda: _no_cleaning(res),
                    "naive": lambda: _naive_cleaning(res),
                    "agent": lambda: _agent_cleaning(res, client),
                    "oracle": lambda: _oracle(res),
                }
                for m in suite["methods"]:
                    repaired, flagged = methods[m]()
                    base = dict(suite="cleaning", dataset=name, condition=cond_name, method=m, seed=seed)
                    _add(rows, base, _score_cleaning(res, repaired, flagged, ytr, Xte, yte))


# ================================================================== tabular anomaly suite
def _build_anomaly_benchmark(X: pd.DataFrame, y: pd.Series, normal_label, contamination: float, seed: int):
    normal = X[y == normal_label]
    n_anom = max(1, int(np.ceil(contamination * len(normal) / (1 - contamination))))
    anom = X[y != normal_label].sample(n=n_anom, random_state=seed)
    data = pd.concat([normal, anom]).sample(frac=1.0, random_state=seed)
    return data, (~data.index.isin(normal.index))


def _lof_scores(df: pd.DataFrame) -> pd.Series:
    Xs = StandardScaler().fit_transform(df[numeric_cols(df)].astype(float))
    lof = LocalOutlierFactor(n_neighbors=20).fit(Xs)
    return pd.Series(-lof.negative_outlier_factor_, index=df.index)


def _zmax_scores(df: pd.DataFrame) -> pd.Series:
    return pd.concat([detectors.classic_z(df[c].astype(float)) for c in numeric_cols(df)], axis=1).max(axis=1)


def run_row_anomaly(suite: dict, seeds: list[int], client: LLMClient, rows: Rows, fingerprints: dict) -> None:
    for item in suite["datasets"]:
        ld = load_dataset(item["name"])
        fingerprints[item["name"]] = int(pd.util.hash_pandas_object(ld.df, index=True).sum() % (2**63))
        for seed in seeds:
            data, labels = _build_anomaly_benchmark(
                ld.features, ld.target, item["normal_label"], suite.get("contamination", 0.05), seed
            )
            scorers = {
                "isolation_forest": lambda: detectors.isolation_forest_scores(data, seed),
                "lof": lambda: _lof_scores(data),
                "zscore_max": lambda: _zmax_scores(data),
            }

            def _agent():
                s = Session(df=data.copy())
                run_agent("Find anomalous rows in this dataset.", s, client)
                return s.row_scores

            scorers["agent"] = _agent
            for m in suite["methods"]:
                scores = scorers[m]().loc[data.index].to_numpy()
                k = int(labels.sum())
                topk = np.zeros(len(scores), bool)
                topk[np.argsort(-scores)[:k]] = True
                base = dict(suite="row_anomaly", dataset=item["name"], condition="default", method=m, seed=seed)
                _add(rows, base, {"auroc": M.auroc(labels, scores), "auprc": M.auprc(labels, scores),
                                  "precision_at_k": M.prf(labels, topk)["precision"]})


# ================================================================== time-series suite
def run_timeseries(suite: dict, seeds: list[int], client: LLMClient, rows: Rows, fingerprints: dict) -> None:
    for name in suite["datasets"]:
        ld = load_dataset(name)
        fingerprints[name] = int(pd.util.hash_pandas_object(ld.df, index=True).sum() % (2**63))
        base_vals = ld.df["value"].to_numpy()
        for seed in seeds:
            dirty, labels = inject_ts_anomalies(base_vals, suite.get("n_events", 8), seed)

            def _agent():
                s = Session(df=pd.DataFrame({"value": dirty}))
                run_agent("Find anomalies in this time series.", s, client)
                return s.ts_scores, s.ts_flags

            naive_scores = detectors.classic_z(pd.Series(dirty)).to_numpy()
            det = {"naive_zscore": (naive_scores, naive_scores > 3.0)}
            if "agent" in suite["methods"]:
                det["agent"] = _agent()
            for m in suite["methods"]:
                scores, flags = det[m]
                base = dict(suite="timeseries", dataset=name, condition="default", method=m, seed=seed)
                _add(rows, base, {"auroc": M.auroc(labels, scores), "auprc": M.auprc(labels, scores),
                                  **M.event_f1(labels, flags)})


# ================================================================== orchestration
SUITES = {"cleaning": run_cleaning, "row_anomaly": run_row_anomaly, "timeseries": run_timeseries}
KEY_METRICS = {
    "cleaning": ["detect_f1", "repair_nrmse", "over_cleaning_rate", "downstream_acc"],
    "row_anomaly": ["auroc", "auprc", "precision_at_k"],
    "timeseries": ["auroc", "auprc", "event_f1"],
}


def _env_info(cfg: dict, fingerprints: dict) -> dict:
    def pkg(n):
        try:
            return version(n)
        except Exception:
            return "?"

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    except Exception:
        commit = ""
    return {
        "python": sys.version, "platform": platform.platform(), "git_commit": commit or None,
        "packages": {n: pkg(n) for n in ["pandas", "numpy", "scikit-learn", "langgraph", "plotly", "scipy"]},
        "data_fingerprints": fingerprints,
        "llm_provider": __import__("os").environ.get("ADA_LLM_PROVIDER", "scripted"),
    }


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def run_experiment(config_path: str | Path, out_root: str | Path | None = None, client: LLMClient | None = None) -> Path:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    client = client or make_client()
    seeds = cfg["seeds"]
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    out = Path(out_root or REPO_ROOT / "results") / f"{cfg['run_name']}_{cfg_hash}"
    out.mkdir(parents=True, exist_ok=True)

    rows: Rows = []
    fingerprints: dict = {}
    for suite_name, suite_cfg in cfg["suites"].items():
        print(f"[suite] {suite_name}")
        SUITES[suite_name](suite_cfg, seeds, client, rows, fingerprints)

    results = pd.DataFrame(rows)
    summary = summarize(results)
    paired = paired_comparison(results)
    results.to_csv(out / "results.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (out / "env.json").write_text(json.dumps(_env_info(cfg, fingerprints), indent=2), encoding="utf-8")

    # Human-readable report (HTML with Plotly charts + markdown tables)
    md, html = [f"# Results: {cfg['run_name']}\n", f"seeds: {seeds}\n"], []
    for (suite, ds, cond), g in summary.groupby(["suite", "dataset", "condition"], sort=False):
        g = g[g["metric"].isin(KEY_METRICS[suite])]
        md += [f"\n## {suite} / {ds} / {cond}\n", _markdown_table(g[["method", "metric", "mean", "ci_low", "ci_high", "n"]])]
        html.append(viz.metric_bars(g, f"{suite} / {ds} / {cond}").to_html(full_html=False, include_plotlyjs=False))
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (out / "report.html").write_text(
        "<html><head><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body>"
        + "".join(html) + "</body></html>", encoding="utf-8")
    print(f"[done] {out}")
    return out
