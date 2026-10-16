"""Published clinical scores used as benchmarks.

THRIVE (Flint et al., Stroke 2010): NIHSS (<=10: 0, 11-20: 1, >=21: 2) + age (<=59: 0,
60-79: 1, >=80: 2) + 1 point each for hypertension, diabetes, atrial fibrillation. Range 0-9;
higher = worse outcome.

SPAN-100 (Saposnik et al., Stroke 2013): age + NIHSS >= 100 -> positive (poor prognosis).

Single-variable benchmarks for later stages: NIHSS at 24 h and NIHSS on day 7.
All scores are returned so that HIGHER = more likely mRS 0-2 (sign flipped when needed),
which makes AUROC directly comparable with the models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def thrive(df: pd.DataFrame) -> pd.Series:
    n = df["nihss_adm"]
    a = df["age"]
    pts_n = np.select([n <= 10, n <= 20], [0, 1], 2).astype(float)
    pts_a = np.select([a <= 59, a <= 79], [0, 1], 2).astype(float)
    s = pts_n + pts_a + df["htn"].fillna(0) + df["dm"].fillna(0) + df["arrhythmia"].fillna(0)
    s[n.isna() | a.isna()] = np.nan
    return s


def span100(df: pd.DataFrame) -> pd.Series:
    s = ((df["age"] + df["nihss_adm"]) >= 100).astype(float)
    s[df["age"].isna() | df["nihss_adm"].isna()] = np.nan
    return s


BENCHMARKS = {
    # stage -> {name: (function returning "higher = worse", label)}
    "admission": {
        "THRIVE": (thrive, "THRIVE score"),
        "SPAN-100": (span100, "SPAN-100 index"),
    },
    "procedure": {
        "THRIVE": (thrive, "THRIVE score"),
    },
    "h24": {
        "NIHSS 24 h": (lambda d: d["nihss_24h"], "NIHSS at 24 h alone"),
    },
    "d7": {
        "NIHSS day 7": (lambda d: d["nihss_7d"], "NIHSS on day 7 alone"),
    },
}


def benchmark_scores(df: pd.DataFrame, stage: str, target: str) -> dict[str, pd.Series]:
    """Benchmark scores oriented so that higher = more likely the target event."""
    out = {}
    for name, (fn, _label) in BENCHMARKS.get(stage, {}).items():
        worse = fn(df).astype(float)
        out[name] = -worse if target == "good_outcome" else worse
    return out
