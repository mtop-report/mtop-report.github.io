"""Turn the clean registry into model matrices for each stage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Feature, stage_features, targets


def _column(df: pd.DataFrame, f: Feature) -> pd.DataFrame:
    if f.type == "categorical":
        src = df[f.key]
        out = {}
        for col in f.columns:
            val = col.split("__", 1)[1]
            v = (src == val).astype("float64")
            v[src.isna()] = np.nan
            out[col] = v
        return pd.DataFrame(out, index=df.index)
    if f.type == "derived":
        if f.key == "nihss_change_24h":
            return pd.DataFrame({f.key: df["nihss_24h"] - df["nihss_adm"]}, index=df.index)
        raise KeyError(f"No rule for derived feature {f.key}")
    return pd.DataFrame({f.key: df[f.key].astype("float64")}, index=df.index)


def design_matrix(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Model matrix (float64, NaN = missing) for a stage, columns in registry order."""
    parts = [_column(df, f) for f in stage_features(stage)]
    return pd.concat(parts, axis=1).astype("float64")


def feature_columns(stage: str) -> list[str]:
    return [c for f in stage_features(stage) for c in f.columns]


def column_to_feature(stage: str) -> dict[str, str]:
    return {c: f.key for f in stage_features(stage) for c in f.columns}


def monotone_vector(stage: str, target: str) -> list[int]:
    sign = int(targets()[target]["sign"])
    out = []
    for f in stage_features(stage):
        for _ in f.columns:
            out.append(int(f.monotone) * sign)
    return out
