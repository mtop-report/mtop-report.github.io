"""Turn the clean registry into model matrices for each stage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Feature, stage_features, targets


def _num(df: pd.DataFrame, key: str) -> pd.Series:
    return df[key].astype("float64")


def derived(df: pd.DataFrame, key: str) -> pd.Series:
    """Derived predictors (the browser computes the same in web/model.js buildRow)."""
    if key == "nihss_change_24h":
        return _num(df, "nihss_24h") - _num(df, "nihss_adm")
    if key == "eni_24h":
        n0, n24 = _num(df, "nihss_adm"), _num(df, "nihss_24h")
        out = (((n0 - n24) >= 8) | (n24 <= 1)).astype("float64")
        unknown = n24.isna() | (n0.isna() & ~(n24 <= 1))
        return out.where(~unknown)
    if key == "night_onset":
        h = _num(df, "onset_hour")
        return ((h >= 22) | (h < 6)).astype("float64").where(h.notna())
    if key == "ivt_to_puncture":
        return _num(df, "onset_to_puncture") - _num(df, "onset_to_ivt")
    if key == "pulse_pressure":
        return _num(df, "sbp") - _num(df, "dbp")
    if key == "hyperglycemia":
        g = _num(df, "glucose")
        return (g > 7.8).astype("float64").where(g.notna())
    raise KeyError(f"No rule for derived feature {key}")


def _column(df: pd.DataFrame, f: Feature) -> pd.DataFrame:
    if f.type == "categorical":
        src = df[f.key]
        cats = [col.split("__", 1)[1] for col in f.columns]
        ref = f.spec.get("reference")
        # missing, or a category without a model term yet (e.g. ACA): not used by the model
        unused = ~src.isin(cats + ([ref] if ref is not None else []))
        out = {}
        for col, val in zip(f.columns, cats):
            v = (src == val).astype("float64")
            v[unused] = np.nan
            out[col] = v
        return pd.DataFrame(out, index=df.index)
    if f.type == "derived":
        return pd.DataFrame({f.key: derived(df, f.key)}, index=df.index)
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
