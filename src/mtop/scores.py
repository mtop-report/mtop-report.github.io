"""Published clinical scores and published predictor sets used as benchmarks.

Point scores (higher = worse outcome):
  THRIVE   (Flint et al., Stroke 2010): NIHSS <=10: 0, 11-20: 1, >=21: 2; age <=59: 0,
           60-79: 1, >=80: 2; +1 each for hypertension, diabetes, atrial fibrillation. 0-9.
  SPAN-100 (Saposnik et al., Stroke 2013): age + NIHSS >= 100 -> positive.
  HIAT2    (Sarraj et al., Stroke 2013): age <=59: 0, 60-79: 2, >=80: 4; NIHSS <=10: 0,
           11-20: 1, >=21: 2; glucose >=150 mg/dL: 1; ASPECTS <=7: 3. 0-10.
  PRE      (Rangaraju et al., J NeuroIntervent Surg 2015): age + 2 x NIHSS - 10 x ASPECTS.
  NIHSS at 24 h alone; NIHSS on day 7 alone.
Each point score is mapped to a probability with a logistic fit (recalibration) on the
development cohort, so discrimination is compared on equal terms.

Published predictor sets refitted on the registry (published coefficients not used):
  THRIVE-c (Flint et al., Int J Stroke 2015): continuous age and NIHSS, hypertension,
           diabetes, atrial fibrillation.
  MR PREDICTS@24H ("Development and validation of a postprocedural model to predict outcome
           after endovascular treatment for ischemic stroke", 2023; PMC10391355): age, NIHSS, pre-stroke mRS,
           diabetes, occlusion site, mTICI, NIHSS at 24 h (knot at 12), sICH. The CTA
           collateral score is not recorded in the FNO registry and is omitted.

Term specifications are shared with the browser (bundle "benchmarks"), which evaluates the
same formulas for the calculator's "other models" panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as Mx
from .lr import fit_logreg

GLU_150 = 150 / 18.016  # 150 mg/dL in mmol/L


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


def hiat2(df: pd.DataFrame) -> pd.Series:
    a, n, g, asp = df["age"], df["nihss_adm"], df["glucose"], df["aspects"]
    s = (np.select([a <= 59, a <= 79], [0, 2], 4).astype(float)
         + np.select([n <= 10, n <= 20], [0, 1], 2).astype(float)
         + (g >= GLU_150).astype(float)
         + np.where(asp <= 7, 3.0, 0.0))
    s = pd.Series(s, index=df.index)
    s[a.isna() | n.isna() | g.isna() | asp.isna()] = np.nan
    return s


def pre_score(df: pd.DataFrame) -> pd.Series:
    return df["age"] + 2 * df["nihss_adm"] - 10 * df["aspects"]


POINT_SCORES = {
    "THRIVE": thrive,
    "SPAN-100": span100,
    "HIAT2": hiat2,
    "PRE": pre_score,
    "NIHSS 24 h": lambda d: d["nihss_24h"],
    "NIHSS day 7": lambda d: d["nihss_7d"],
}

# term: (kind, key[, arg]); kinds: num, bin, in (membership), min (min(x, c)), max0 (max(x - c, 0))
REFIT_TERMS = {
    "THRIVE-c": [("num", "age"), ("num", "nihss_adm"), ("bin", "htn"), ("bin", "dm"), ("bin", "arrhythmia")],
    "MR PREDICTS@24H": [
        ("num", "age"), ("num", "nihss_adm"), ("num", "mrs_pre"), ("bin", "dm"),
        ("in", "occlusion", ["M1"]), ("in", "occlusion", ["distal", "basilar", "other_posterior"]),
        ("num", "tici"), ("min", "nihss_24h", 12), ("max0", "nihss_24h", 12), ("bin", "sich"),
    ],
}

BENCHMARKS = {
    "admission": ["THRIVE", "THRIVE-c", "SPAN-100", "HIAT2", "PRE"],
    "procedure": ["THRIVE"],
    "h24": ["NIHSS 24 h", "MR PREDICTS@24H"],
    "d7": ["NIHSS day 7"],
}

# shown in the calculator's "other models" panel (pre-treatment scores stay visible later on)
PANEL = {
    "admission": ["THRIVE", "THRIVE-c", "SPAN-100", "HIAT2", "PRE"],
    "procedure": ["THRIVE", "THRIVE-c", "SPAN-100", "HIAT2", "PRE"],
    "h24": ["MR PREDICTS@24H", "NIHSS 24 h", "THRIVE", "HIAT2"],
    "d7": ["NIHSS day 7", "MR PREDICTS@24H", "THRIVE"],
}


def design(df: pd.DataFrame, terms) -> pd.DataFrame:
    out = {}
    for i, t in enumerate(terms):
        kind, key = t[0], t[1]
        x = df[key]
        if kind in ("num", "bin"):
            v = x.astype(float)
        elif kind == "in":
            v = x.isin(t[2]).astype(float).where(x.notna())
        elif kind == "min":
            v = np.minimum(x.astype(float), t[2])
        elif kind == "max0":
            v = np.maximum(x.astype(float) - t[2], 0.0)
        else:
            raise ValueError(kind)
        out[f"t{i}_{kind}_{key}"] = v
    return pd.DataFrame(out, index=df.index).astype(float)


def fit_points_map(score: pd.Series, y: np.ndarray) -> dict:
    s = np.asarray(score, float)
    keep = ~np.isnan(s)
    a, b = Mx._fit_logistic(s[keep], np.asarray(y, float)[keep])
    return {"a": float(a), "b": float(b), "fill": float(np.nanmedian(s))}


def apply_points_map(mp: dict, score) -> np.ndarray:
    s = np.asarray(score, float)
    s = np.where(np.isnan(s), mp["fill"], s)
    return Mx.expit(mp["a"] + mp["b"] * s)


def benchmark_predictions(dev: pd.DataFrame, test: pd.DataFrame, stage: str, target: str, seed: int) -> dict:
    """Probabilities for the temporal-validation cohort from every benchmark of a stage."""
    y_dev = dev[target].to_numpy(float)
    out = {}
    for name in BENCHMARKS.get(stage, []):
        if name in POINT_SCORES:
            mp = fit_points_map(POINT_SCORES[name](dev), y_dev)
            out[name] = apply_points_map(mp, POINT_SCORES[name](test))
        else:
            terms = REFIT_TERMS[name]
            model = fit_logreg(design(dev, terms), y_dev, seed)
            out[name] = model.predict_proba(design(test, terms))[:, 1]
    return out
