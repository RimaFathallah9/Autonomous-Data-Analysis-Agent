"""Seeded fault injection = how we manufacture ground truth.

IDEA: take a clean table, break it in controlled, *recorded* ways, then ask a method to
find/fix the damage. Because we recorded exactly which cells we broke (`error_mask`),
precision/recall/repair-error are exact, not opinions.

Everything is driven by a numpy Generator created from the seed => same seed, same dirt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel


class CorruptionConfig(BaseModel):
    missing_rate: float = 0.0
    missing_mechanism: Literal["MCAR", "MAR"] = "MCAR"
    outlier_rate: float = 0.0
    outlier_min_sigma: float = 6.0
    outlier_max_sigma: float = 10.0
    typo_rate: float = 0.0
    duplicate_rate: float = 0.0


@dataclass
class CorruptionResult:
    clean: pd.DataFrame
    dirty: pd.DataFrame  # may contain extra duplicate rows (new index labels)
    error_mask: pd.DataFrame  # bool, indexed like `clean`
    source_index: pd.Series  # dirty index label -> clean index label it came from
    duplicate_rows: pd.Index = field(default_factory=lambda: pd.Index([]))


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def categorical_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in numeric_cols(df)]


def _typo(s: str, rng: np.random.Generator) -> str:
    kind = rng.integers(0, 3)
    if kind == 0 and len(s) > 2:  # swap two adjacent characters
        i = int(rng.integers(0, len(s) - 1))
        out = s[:i] + s[i + 1] + s[i] + s[i + 2 :]
    elif kind == 1 and len(s) > 2:  # drop a character
        i = int(rng.integers(0, len(s)))
        out = s[:i] + s[i + 1 :]
    else:  # trailing junk
        out = s + "_"
    return out if out != s else s + "_"


def corrupt_tabular(clean: pd.DataFrame, cfg: CorruptionConfig, seed: int) -> CorruptionResult:
    """Corrupt a FEATURES-only table (never pass the target column)."""
    rng = np.random.default_rng(seed)
    dirty = clean.copy()
    mask = pd.DataFrame(False, index=clean.index, columns=clean.columns)
    num, cat = numeric_cols(clean), categorical_cols(clean)

    # 1) Outliers: push a cell 6-10 standard deviations away from its true value.
    if cfg.outlier_rate > 0:
        for c in num:
            hit = rng.random(len(clean)) < cfg.outlier_rate
            sigma = clean[c].std()
            sign = rng.choice([-1.0, 1.0], size=len(clean))
            mag = rng.uniform(cfg.outlier_min_sigma, cfg.outlier_max_sigma, size=len(clean))
            vals = clean[c].to_numpy(dtype=float).copy()
            vals[hit] = vals[hit] + sign[hit] * mag[hit] * sigma
            dirty[c] = vals
            mask.loc[hit, c] = True

    # 2) Typos in text columns -> creates spurious rare categories.
    if cfg.typo_rate > 0:
        for c in cat:
            hit = rng.random(len(clean)) < cfg.typo_rate
            vals = clean[c].astype(object).to_numpy().copy()
            for i in np.flatnonzero(hit):
                vals[i] = _typo(str(vals[i]), rng)
            dirty[c] = vals
            mask.loc[hit, c] = True

    # 3) Missing values. Applied last so a deleted cell is just "missing".
    if cfg.missing_rate > 0:
        cols = list(clean.columns)
        if cfg.missing_mechanism == "MAR" and num:
            # Missingness depends on an OBSERVED column (the first numeric one):
            # rows with large driver values lose data more often. Mean rate is preserved.
            driver = clean[num[0]].rank(pct=True).to_numpy()
            p = np.clip(cfg.missing_rate * 2 * driver, 0, 1)
            cols = [c for c in cols if c != num[0]]
        else:
            p = np.full(len(clean), cfg.missing_rate)
        for c in cols:
            hit = rng.random(len(clean)) < p
            col = dirty[c].astype(object) if c in cat else dirty[c].astype(float)
            col = col.copy()
            col[hit] = np.nan
            dirty[c] = col
            mask.loc[hit, c] = True

    # 4) Exact duplicate rows appended at the end (copies of already-dirty rows).
    source = pd.Series(dirty.index, index=dirty.index)
    dup_idx = pd.Index([])
    if cfg.duplicate_rate > 0:
        k = int(round(cfg.duplicate_rate * len(dirty)))
        picks = rng.choice(len(dirty), size=k, replace=False)
        copies = dirty.iloc[picks].copy()
        start = int(clean.index.max()) + 1 if len(clean) else 0
        copies.index = pd.RangeIndex(start, start + k)
        source = pd.concat([source, pd.Series(dirty.index[picks].to_numpy(), index=copies.index)])
        dirty = pd.concat([dirty, copies])
        dup_idx = copies.index

    return CorruptionResult(clean=clean, dirty=dirty, error_mask=mask, source_index=source, duplicate_rows=dup_idx)


# --------------------------------------------------------------------------- time series
def inject_ts_anomalies(
    values: np.ndarray, n_events: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dirty_values, labels). Three event types: spike, level_shift, flatline."""
    rng = np.random.default_rng(seed)
    v = values.astype(float).copy()
    labels = np.zeros(len(v), dtype=bool)
    std = v.std()
    n = len(v)
    taken: list[tuple[int, int]] = []
    attempts = 0
    while len(taken) < n_events and attempts < 1000:
        attempts += 1
        kind = rng.choice(["spike", "level_shift", "flatline"])
        length = 1 if kind == "spike" else int(rng.integers(20, 40))
        start = int(rng.integers(100, n - length - 100))
        if any(start < e + 50 and s < start + length + 50 for s, e in taken):
            continue  # keep events apart so they are separate, countable events
        sl = slice(start, start + length)
        if kind == "spike":
            v[start] += rng.choice([-1, 1]) * 3.0 * std
        elif kind == "level_shift":
            v[sl] += rng.choice([-1, 1]) * 1.0 * std
        else:
            v[sl] = v[start]
        labels[sl] = True
        taken.append((start, start + length))
    return v, labels
