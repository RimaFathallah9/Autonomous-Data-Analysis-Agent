"""Statistics for reporting: never show a bare mean from a stochastic system."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

GROUP = ["suite", "dataset", "condition", "method", "metric"]


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean: resample runs with replacement, take quantiles."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in results.groupby(GROUP, sort=False):
        lo, hi = bootstrap_ci(g["value"].to_numpy())
        rows.append({**dict(zip(GROUP, key)), "n": int(g["value"].notna().sum()), "mean": g["value"].mean(),
                     "std": g["value"].std(), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def paired_comparison(results: pd.DataFrame, a: str = "agent") -> pd.DataFrame:
    """Wilcoxon signed-rank test of `a` vs every other method on the SAME seeds (paired).

    Needs >= 6 seeds to be able to reach p < 0.05; with fewer it returns NaN honestly.
    """
    out = []
    keys = ["suite", "dataset", "condition", "metric"]
    for key, g in results.groupby(keys, sort=False):
        wide = g.pivot_table(index="seed", columns="method", values="value")
        if a not in wide:
            continue
        for other in wide.columns:
            if other == a:
                continue
            d = (wide[a] - wide[other]).dropna()
            p = float("nan")
            if len(d) >= 6 and (d != 0).any():
                p = float(wilcoxon(d).pvalue)
            out.append({**dict(zip(keys, key)), "a": a, "b": other, "mean_diff": d.mean(), "n": len(d), "p_value": p})
    return pd.DataFrame(out)
