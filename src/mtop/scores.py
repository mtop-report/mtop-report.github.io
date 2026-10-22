"""Published clinical scores and published predictor sets used as benchmarks.

Point scores (higher = worse outcome):
  THRIVE   (Flint et al., Stroke 2010): NIHSS <=10: 0, 11-20: 1, >=21: 2; age <=59: 0,
           60-79: 1, >=80: 2; +1 each for hypertension, diabetes, atrial fibrillation. 0-9.
  SPAN-100 (Saposnik et al., Stroke 2013): age + NIHSS >= 100 -> positive (the published cut-off).
  SPAN index age + NIHSS as a continuous number, as the 2026 FNO analysis scored SPAN-100.
  HIAT2    (Sarraj et al., Stroke 2013): age <=59: 0, 60-79: 2, >=80: 4; NIHSS <=10: 0,
           11-20: 1, >=21: 2; glucose >=150 mg/dL: 1; ASPECTS <=7: 3. 0-10.
  PRE      (Rangaraju et al., J NeuroIntervent Surg 2015): age + 2 x NIHSS - 10 x ASPECTS.
  Bedside  7-item score from the earlier FNO analysis (report "Predikce výsledku trombektomie",
           2026): age <60: 0, 60-79: 2, >=80: 5; NIHSS 0-9: 0, 10-14: 1, 15-19: 3, >=20: 5;
           glucose <7.8: 0, 7.8-11.0: 2, >=11.1 mmol/L: 3; pre-stroke mRS >=1: 2; ASPECTS 0-5: 2;
           onset to puncture 180-299 min: 1, >=300: 2; previous stroke: 1. 0-20 points; an item
           that is not recorded scores 0.
  THRIVE-EVT (Flint et al., Int J Stroke 2023): the published
           log-odds of mRS 0-2 with thrombectomy, 2.864 - 0.025 x age - 0.106 x NIHSS
           + 0.753 (thrombectomy) + chronic-disease term (none 0, one -0.419, two -0.502, all three
           -1.665; hypertension, diabetes, atrial fibrillation), as in Hahn et al., JNIS 2025.
  CLEAR    (thrombectomy score, as compared in Hahn et al., JNIS 2025): age <51: 2, 51-75: 1,
           >75: 0; ASPECTS >5: 1; NIHSS <6: 5, 6-12: 3, 13-18: 1, >18: 0. Higher = better.
  BET      (Bronx Endovascular Thrombectomy score, J Stroke Cerebrovasc Dis 2022): current smoker,
           diabetes, general anaesthesia, puncture to reperfusion >= 45 min, final mTICI below 3
           (2c counts as 3): 1 point each; NIHSS after the procedure >= 10: 2 points (the registry's
           NIHSS at 24 h). 0-7, higher = worse; an item that is not recorded scores 0.
  NIHSS at 24 h alone; NIHSS on day 7 alone.
Each point score is mapped to a probability with a logistic fit (recalibration) on the
development cohort, so discrimination is compared on equal terms.

Published predictor sets refitted on the registry (published coefficients not used):
  THRIVE-c (Flint et al., Int J Stroke 2015): continuous age and NIHSS, hypertension,
           diabetes, atrial fibrillation.
  Bedside variables (regression): the seven bedside items as a linear logistic regression (the
           2026 analysis's parsimonious model).
  MR PREDICTS (Venema et al., Stroke 2021: validated and updated in HERMES; PMC8378416): age,
           NIHSS, systolic blood pressure, glucose, IV thrombolysis, diabetes, pre-stroke mRS,
           ASPECTS, occlusion site, onset to groin puncture, as linear terms. The CTA collateral
           score is not recorded in the FNO registry and is omitted.
  MR PREDICTS@24H ("Development and validation of a postprocedural model to predict outcome
           after endovascular treatment for ischemic stroke", 2023; PMC10391355): age, NIHSS, pre-stroke mRS,
           diabetes, occlusion site, mTICI, NIHSS at 24 h (knot at 12), sICH. The CTA
           collateral score is not recorded in the FNO registry and is omitted.

The Validation tab tests every score the calculator shows at a stage (STAGE_SCORES serves both).
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


def span_index(df: pd.DataFrame) -> pd.Series:
    return df["age"] + df["nihss_adm"]


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


THRIVE_EVT_CHRONIC = {0: 0.0, 1: -0.419, 2: -0.502, 3: -1.665}


def thrive_evt(df: pd.DataFrame) -> pd.Series:
    """Published THRIVE-EVT log-odds of mRS 0-2 with thrombectomy (higher = better)."""
    k = (df["htn"].fillna(0) + df["dm"].fillna(0) + df["arrhythmia"].fillna(0)).clip(0, 3).astype(int)
    s = 2.864 - 0.025 * df["age"] - 0.106 * df["nihss_adm"] + 0.753 + k.map(THRIVE_EVT_CHRONIC).astype(float)
    s[df["age"].isna() | df["nihss_adm"].isna()] = np.nan
    return s


def clear_score(df: pd.DataFrame) -> pd.Series:
    """CLEAR points (higher = better): age, ASPECTS and NIHSS bands."""
    a, n, asp = df["age"], df["nihss_adm"], df["aspects"]
    s = pd.Series(np.select([a < 51, a <= 75], [2, 1], 0).astype(float)
                  + np.where(asp > 5, 1.0, 0.0)
                  + np.select([n < 6, n <= 12, n <= 18], [5, 3, 1], 0).astype(float), index=df.index)
    s[a.isna() | n.isna() | asp.isna()] = np.nan
    return s


def bet_score(df: pd.DataFrame) -> pd.Series:
    """BET points (0-7, higher = worse); NIHSS after the procedure = the NIHSS at 24 h."""
    one = lambda cond: np.where(cond, 1.0, 0.0)  # noqa: E731
    s = pd.Series(one(df["smoking"] == 1) + one(df["dm"] == 1) + one(df["anesthesia"] == "general")
                  + one(df["procedure_min"] >= 45) + one(df["tici"] < 4) + np.where(df["nihss_24h"] >= 10, 2.0, 0.0),
                  index=df.index)
    s[df["tici"].isna() | df["nihss_24h"].isna()] = np.nan
    return s


def bedside(df: pd.DataFrame) -> pd.Series:
    """FNO 7-item bedside score (0-20, higher = worse); missing items score 0."""
    a, n, g = df["age"], df["nihss_adm"], df["glucose"]
    otp, asp, pre = df["onset_to_puncture"], df["aspects"], df["mrs_pre"]
    pts = (np.select([a < 60, a < 80], [0, 2], 5).astype(float)
           + np.select([n <= 9, n <= 14, n <= 19], [0, 1, 3], 5).astype(float)
           + np.where(g >= 11.1, 3.0, np.where(g >= 7.8, 2.0, 0.0))
           + np.where(pre >= 1, 2.0, 0.0)
           + np.where(asp <= 5, 2.0, 0.0)
           + np.where(otp >= 300, 2.0, np.where(otp >= 180, 1.0, 0.0))
           + np.where(df["stroke_prior"] == 1, 1.0, 0.0))
    s = pd.Series(pts, index=df.index)
    s[a.isna() | n.isna()] = np.nan      # age and NIHSS are required
    return s


POINT_SCORES = {
    "THRIVE": thrive,
    "SPAN-100": span100,
    "SPAN index": span_index,
    "HIAT2": hiat2,
    "PRE": pre_score,
    "THRIVE-EVT": thrive_evt,
    "CLEAR": clear_score,
    "BET": bet_score,
    "Bedside score": bedside,
    "NIHSS 24 h": lambda d: d["nihss_24h"],
    "NIHSS day 7": lambda d: d["nihss_7d"],
}

# term: (kind, key[, arg]); kinds: num, bin, in (membership), min (min(x, c)), max0 (max(x - c, 0))
REFIT_TERMS = {
    "THRIVE-c": [("num", "age"), ("num", "nihss_adm"), ("bin", "htn"), ("bin", "dm"), ("bin", "arrhythmia")],
    "Bedside variables (regression)": [("num", "age"), ("num", "nihss_adm"), ("num", "glucose"), ("num", "mrs_pre"),
                                       ("num", "aspects"), ("num", "onset_to_puncture"), ("bin", "stroke_prior")],
    "MR PREDICTS": [
        ("num", "age"), ("num", "nihss_adm"), ("num", "sbp"), ("num", "glucose"), ("bin", "ivt"), ("bin", "dm"),
        ("num", "mrs_pre"), ("num", "aspects"),
        ("in", "occlusion", ["M1"]), ("in", "occlusion", ["M2", "basilar", "basilar_vertebral", "other_posterior"]),
        ("num", "onset_to_puncture"),
    ],
    "MR PREDICTS@24H": [
        ("num", "age"), ("num", "nihss_adm"), ("num", "mrs_pre"), ("bin", "dm"),
        ("in", "occlusion", ["M1"]), ("in", "occlusion", ["M2", "basilar", "basilar_vertebral", "other_posterior"]),
        ("num", "tici"), ("min", "nihss_24h", 12), ("max0", "nihss_24h", 12), ("bin", "sich"),
    ],
}

# Every published or bedside score shown at a stage: the calculator's "other models" panel lists
# them (pre-treatment scores stay visible later on) and the Validation tab tests the same list on
# the same stage's patients, so nothing is shown in one place without being tested in the other.
STAGE_SCORES = {
    "admission": ["Bedside score", "Bedside variables (regression)", "SPAN index", "SPAN-100", "THRIVE", "THRIVE-c",
                  "THRIVE-EVT", "HIAT2", "PRE", "CLEAR", "MR PREDICTS"],
    "procedure": ["Bedside score", "SPAN index", "THRIVE", "THRIVE-c", "THRIVE-EVT", "HIAT2", "PRE", "CLEAR",
                  "MR PREDICTS"],
    "h24": ["MR PREDICTS@24H", "BET", "NIHSS 24 h", "Bedside score", "THRIVE", "HIAT2"],
    "d7": ["NIHSS day 7", "MR PREDICTS@24H", "BET", "Bedside score", "THRIVE"],
}
BENCHMARKS = STAGE_SCORES
PANEL = STAGE_SCORES


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
