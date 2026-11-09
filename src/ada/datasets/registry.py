"""Dataset registry: typed metadata + checksum-verified loading.

WHY: reproducibility starts with knowing exactly which bytes you evaluated on, and
legally being allowed to use them. Every dataset has a license, source and (for remote
files) a SHA256. Pydantic validates the YAML so typos fail at load time, not mid-experiment.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from ada.datasets.synthetic import make_synthetic_tabular, make_synthetic_timeseries

REPO_ROOT = Path(__file__).resolve().parents[3]


def registry_path() -> Path:
    return Path(os.environ.get("ADA_REGISTRY", REPO_ROOT / "data" / "registry.yaml"))


def raw_dir() -> Path:
    return Path(os.environ.get("ADA_RAW_DIR", REPO_ROOT / "data" / "raw"))


class DatasetSpec(BaseModel):
    name: str
    loader: str
    kind: Literal["tabular", "timeseries"]
    license: str
    source: str
    has_labels: bool
    redistribution_allowed: bool
    task_tags: list[str] = Field(default_factory=list)
    target_col: str | None = None
    url: str | None = None
    sha256: str | None = None
    csv_options: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    df: pd.DataFrame

    @property
    def features(self) -> pd.DataFrame:
        """Feature columns only (target removed)."""
        if self.spec.target_col and self.spec.target_col in self.df.columns:
            return self.df.drop(columns=[self.spec.target_col])
        return self.df

    @property
    def target(self) -> pd.Series | None:
        if self.spec.target_col and self.spec.target_col in self.df.columns:
            return self.df[self.spec.target_col]
        return None


def load_registry(path: Path | None = None) -> dict[str, DatasetSpec]:
    with open(path or registry_path(), encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    specs = [DatasetSpec(**d) for d in raw["datasets"]]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate dataset names in registry")
    return {s.name: s for s in specs}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_remote(spec: DatasetSpec, force: bool = False) -> Path:
    """Download (if needed) and verify a remote dataset. Returns the local path."""
    import httpx

    if not spec.url:
        raise ValueError(f"{spec.name} has no url")
    dest = raw_dir() / f"{spec.name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if force or not dest.exists():
        with httpx.stream("GET", spec.url, follow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    digest = sha256_file(dest)
    if spec.sha256 is None:
        print(f"[pin me] {spec.name}: sha256 = {digest}  (paste into data/registry.yaml)")
    elif digest != spec.sha256:
        dest.unlink()
        raise ValueError(
            f"Checksum mismatch for {spec.name}: expected {spec.sha256}, got {digest}. "
            "Upstream data changed or download was corrupted; file deleted."
        )
    return dest


def load_dataset(name: str) -> LoadedDataset:
    spec = load_registry()[name]
    kind, _, arg = spec.loader.partition(":")
    if kind == "sklearn":
        from sklearn import datasets as skd

        df = getattr(skd, arg)(as_frame=True).frame
    elif kind == "synthetic":
        df = make_synthetic_tabular() if arg == "tabular" else make_synthetic_timeseries()
    elif kind == "url":
        df = pd.read_csv(fetch_remote(spec), **spec.csv_options)
    else:
        raise ValueError(f"Unknown loader '{spec.loader}'")
    return LoadedDataset(spec=spec, df=df)
