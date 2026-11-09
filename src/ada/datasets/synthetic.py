"""Generated datasets.

WHY synthetic data at all? Public datasets have *natural* dirt we cannot label, so we
cannot score a cleaning agent on them reliably. Here we know the exact clean truth.
The dataset itself uses a FIXED seed (it is "the dataset"); randomness that varies per
experiment seed lives in the corruption step instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_tabular(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Correlated features: correlated columns make imputation/outlier detection non-trivial.
    cov = np.array(
        [
            [1.0, 0.6, 0.2, 0.0, 0.0],
            [0.6, 1.0, 0.3, 0.0, 0.1],
            [0.2, 0.3, 1.0, 0.4, 0.0],
            [0.0, 0.0, 0.4, 1.0, 0.2],
            [0.0, 0.1, 0.0, 0.2, 1.0],
        ]
    )
    x = rng.multivariate_normal(mean=[10, 50, 0, 5, 100], cov=cov * 4, size=n)
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(1, 6)])
    df["segment"] = rng.choice(["alpha", "beta", "gamma"], size=n, p=[0.5, 0.3, 0.2])
    logits = 0.8 * (df["f1"] - 10) - 0.5 * (df["f3"]) + (df["segment"] == "alpha") * 0.7
    df["target"] = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    return df


def make_synthetic_timeseries(n: int = 2000, period: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = 5.0 * np.sin(2 * np.pi * t / period) + 2.0 * np.sin(4 * np.pi * t / period)
    trend = 0.002 * t
    noise = rng.normal(0, 1.0, n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "value": seasonal + trend + noise,
        }
    )
