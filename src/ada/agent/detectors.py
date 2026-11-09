"""Pure detection/repair functions (no LLM, no agent). Shared by tools AND baselines,
so a performance difference between methods comes from the *policy*, not from
different underlying code."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ada.corruption import categorical_cols, numeric_cols


def robust_z(x: pd.Series) -> pd.Series:
    """|x - median| / (1.4826 * MAD). Outliers inflate std but barely move the median/MAD."""
    med = x.median()
    mad = (x - med).abs().median()
    if not mad:
        return pd.Series(0.0, index=x.index)
    return (x - med).abs() / (1.4826 * mad)


def classic_z(x: pd.Series) -> pd.Series:
    sd = x.std()
    return (x - x.mean()).abs() / sd if sd else pd.Series(0.0, index=x.index)


def flag_outlier_cells(df: pd.DataFrame, threshold: float, method: str = "robust_z") -> pd.DataFrame:
    fn = robust_z if method == "robust_z" else classic_z
    flags = pd.DataFrame(False, index=df.index, columns=df.columns)
    for c in numeric_cols(df):
        flags[c] = (fn(df[c].astype(float)) > threshold).fillna(False)
    return flags


def flag_rare_categories(df: pd.DataFrame, min_freq: float) -> pd.DataFrame:
    flags = pd.DataFrame(False, index=df.index, columns=df.columns)
    for c in categorical_cols(df):
        freq = df[c].value_counts(normalize=True)
        rare = set(freq[freq < min_freq].index)
        flags[c] = df[c].isin(rare)
    return flags


def impute(df: pd.DataFrame, fix_mask: pd.DataFrame) -> pd.DataFrame:
    """Replace cells in `fix_mask` (or NaN) with column median / mode computed from trusted cells."""
    out = df.copy()
    for c in df.columns:
        bad = fix_mask[c] | df[c].isna()
        if not bad.any():
            continue
        good = df.loc[~bad, c]
        if c in numeric_cols(df):
            fill = good.astype(float).median()
            col = df[c].astype(float).copy()
        else:
            fill = good.mode().iloc[0] if len(good) else "unknown"
            col = df[c].astype(object).copy()
        col[bad] = fill
        out[c] = col
    return out


def isolation_forest_scores(df: pd.DataFrame, seed: int = 0) -> pd.Series:
    X = df[numeric_cols(df)].astype(float)
    X = X.fillna(X.median())
    Xs = StandardScaler().fit_transform(X)
    model = IsolationForest(n_estimators=200, random_state=seed).fit(Xs)
    return pd.Series(-model.score_samples(Xs), index=df.index)  # higher = more anomalous


def seasonal_residual_scores(values: np.ndarray, period: int) -> np.ndarray:
    """Robust anomaly score for seasonal series.

    1) remove the typical seasonal profile (median per phase),
    2) remove slow drift (centred rolling median of the residual),
    3) robust z-score of what is left.
    """
    s = pd.Series(values.astype(float))
    phase = np.arange(len(s)) % period
    profile = s.groupby(phase).transform("median")
    resid = s - profile
    resid = resid - resid.rolling(period * 3, center=True, min_periods=1).median()
    return robust_z(resid).to_numpy()
