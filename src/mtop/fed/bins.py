"""Fixed, public value bands for federated tree learning.

Every model column gets a sorted list of edges from config/federated.yaml (binary and one-hot
columns split at 0). A value x falls in band b = number of edges strictly below x, so
"band <= k" is exactly "x <= edges[k]" and a tree learned on bands can be applied to raw values.
Missing values get their own band (the last one). Bands are agreed before any data is seen, so
they reveal nothing about any centre.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import yaml

from ..config import settings, stage_features
from ..paths import CONFIG


@lru_cache(maxsize=None)
def fed_settings() -> dict:
    with open(CONFIG / "federated.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _edges_for(f) -> list[float]:
    bands = fed_settings()["bands"]
    if f.type in ("binary", "categorical"):
        return [0.0]
    if f.key in bands:
        return [float(v) for v in bands[f.key]]
    lo, hi = float(f.spec.get("min", 0)), float(f.spec.get("max", 1))
    return [float(v) for v in np.linspace(lo, hi, 18)[1:-1]]


@dataclass(frozen=True)
class Bands:
    columns: tuple[str, ...]
    edges: tuple[tuple[float, ...], ...]

    @classmethod
    def for_stage(cls, stage: str) -> "Bands":
        cols, edges = [], []
        for f in stage_features(stage):
            e = tuple(sorted(set(_edges_for(f))))
            for c in f.columns:
                cols.append(c)
                edges.append(e)
        return cls(tuple(cols), tuple(edges))

    @property
    def nbins(self) -> np.ndarray:
        """Bands per column: len(edges) + 1 value bands and one missing band."""
        return np.array([len(e) + 2 for e in self.edges], dtype=np.int64)

    def missing_bin(self, j: int) -> int:
        return len(self.edges[j]) + 1

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Band index per cell (n x p, int16)."""
        X = np.asarray(X, float)
        out = np.empty(X.shape, dtype=np.int16)
        for j, e in enumerate(self.edges):
            x = X[:, j]
            miss = np.isnan(x)
            b = np.searchsorted(np.asarray(e, float), np.where(miss, 0.0, x), side="left")
            b[miss] = len(e) + 1
            out[:, j] = b
        return out

    def threshold(self, j: int, k: int) -> float:
        return float(self.edges[j][k])

    def public(self) -> dict:
        return {c: list(e) for c, e in zip(self.columns, self.edges)}


def min_leaf() -> int:
    return int(settings()["privacy"]["min_leaf"])
