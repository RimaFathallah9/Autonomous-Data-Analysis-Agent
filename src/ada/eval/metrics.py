"""All metrics in one place, each small enough to unit-test with hand-computed answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ada.corruption import categorical_cols, numeric_cols


# ------------------------------------------------------------------ detection
def prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def auroc(y_true, scores) -> float:
    y_true = np.asarray(y_true, bool)
    if y_true.all() or not y_true.any():
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def auprc(y_true, scores) -> float:
    y_true = np.asarray(y_true, bool)
    if not y_true.any():
        return float("nan")
    return float(average_precision_score(y_true, scores))


def _events(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end_exclusive)."""
    m = np.asarray(mask, bool).astype(int)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def event_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Range/event-based F1 for time series.

    A true event is 'found' if any predicted point overlaps it; a predicted event is
    'correct' if it overlaps any true event. Unlike point-adjusted F1 this does NOT
    credit a detector for whole segments after one lucky hit on the *point* metric;
    it counts events, and false-alarm events hurt precision.
    """
    te, pe = _events(y_true), _events(y_pred)
    t, p = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    rec = np.mean([p[s:e].any() for s, e in te]) if te else 0.0
    prec = np.mean([t[s:e].any() for s, e in pe]) if pe else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"event_precision": float(prec), "event_recall": float(rec), "event_f1": float(f1)}


# ------------------------------------------------------------------ cleaning
def repair_nrmse(clean: pd.DataFrame, repaired: pd.DataFrame, error_mask: pd.DataFrame) -> float:
    """RMSE of repaired values on erroneous numeric cells, divided by column std (comparable across columns)."""
    idx = repaired.index.intersection(clean.index)
    errs = []
    for c in numeric_cols(clean):
        m = error_mask.loc[idx, c].to_numpy()
        if not m.any():
            continue
        diff = repaired.loc[idx, c].to_numpy(float)[m] - clean.loc[idx, c].to_numpy(float)[m]
        diff = diff[~np.isnan(diff)]
        errs.append(diff / (clean[c].std() or 1.0))
    if not errs:
        return float("nan")
    return float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))


def repair_cat_acc(clean: pd.DataFrame, repaired: pd.DataFrame, error_mask: pd.DataFrame) -> float:
    idx = repaired.index.intersection(clean.index)
    hits = []
    for c in categorical_cols(clean):
        m = error_mask.loc[idx, c].to_numpy()
        if m.any():
            hits.append((repaired.loc[idx, c].to_numpy()[m] == clean.loc[idx, c].to_numpy()[m]))
    return float(np.mean(np.concatenate(hits))) if hits else float("nan")


def over_cleaning_rate(dirty: pd.DataFrame, repaired: pd.DataFrame, error_mask: pd.DataFrame) -> float:
    """Of the cells that were CORRECT in the dirty table, what fraction did the method change?"""
    idx = repaired.index.intersection(error_mask.index)
    ok = ~error_mask.loc[idx].to_numpy()
    changed = ~(dirty.loc[idx].to_numpy(object) == repaired.loc[idx].to_numpy(object))
    return float((changed & ok).sum() / max(ok.sum(), 1))


def _make_model(X: pd.DataFrame) -> Pipeline:
    num, cat = numeric_cols(X), categorical_cols(X)
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            ),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])


def _prep(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in categorical_cols(X):
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


def downstream_accuracy(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
) -> float:
    """Train on (possibly dirty/cleaned) data, test on the CLEAN held-out split."""
    model = _make_model(train_X)
    model.fit(_prep(train_X), train_y.to_numpy())
    return float((model.predict(_prep(test_X)) == test_y.to_numpy()).mean())


def holdout_split(y: pd.Series, seed: int) -> tuple[pd.Index, pd.Index]:
    tr, te = train_test_split(y.index, test_size=0.3, random_state=seed, stratify=y)
    return pd.Index(tr), pd.Index(te)


# ------------------------------------------------------------------ claims
def verify_numeric_claims(df: pd.DataFrame, claims: list[dict], rel_tol: float = 1e-6) -> dict[str, float]:
    """Re-compute every numeric claim from the data. Claim = {stat, column, value}.

    Faithfulness = share of claims that match. This is how you would score an LLM's
    written report: never trust the prose, recompute from the source table.
    """
    ok = 0
    for c in claims:
        truth = getattr(df[c["column"]], c["stat"])()
        ok += bool(np.isclose(truth, c["value"], rtol=rel_tol, atol=1e-9))
    return {"claims": len(claims), "faithfulness": ok / len(claims) if claims else float("nan")}
