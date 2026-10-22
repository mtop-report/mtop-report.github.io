"""The shared data dictionary: read a site's registry export in the MTOP upload template.

One row per patient, one column per predictor (the calculator's inputs) plus `mrs90`. The same
template is used by the in-browser validation tool (web/view_network.js); values may use common
English or Czech spellings. Unrecognised or out-of-range values become missing and are counted.

Optional column `year` (year of treatment) lets the network work on ranges of years: monitoring
performance year by year, recalibrating to recent patients, refitting on a sliding window. It
never leaves the hospital either; jobs only select patients by it.

Optional columns for future versions (config/future.yaml: CT perfusion volumes, the CTA collateral
score, serum GFAP with its sampling time and assay) are read and range-checked when present. No job
uses them as a model input; the site only reports how many patients have them recorded, which
counts toward the rule that a new predictor is tested once 100 patients with mRS 0-2 and 100 with
mRS 3-6 have it (amendment 1).
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from ..config import features, future_columns

POSTERIOR = {"basilar", "basilar_vertebral", "other_posterior"}
TICI = {"0": 0.0, "1": 1.0, "2a": 2.0, "2": 2.5, "2b": 3.0, "2c": 4.0, "3": 4.0}
BIN = {"1": 1, "0": 0, "yes": 1, "no": 0, "y": 1, "n": 0, "true": 1, "false": 0, "ano": 1, "ne": 0}
SEX = {"1": 1, "0": 0, "m": 1, "male": 1, "man": 1, "muz": 1, "f": 0, "female": 0, "woman": 0, "w": 0, "z": 0, "zena": 0}
SIDE = {"1": 1, "0": 0, "left": 1, "l": 1, "leva": 1, "right": 0, "r": 0, "prava": 0}
CAT = {
    # ACA is accepted as its own site; it has no model term yet, so the models treat it as not recorded
    "occlusion": {"ica": "ICA", "intracranialica": "ICA", "m1": "M1", "m2": "M2", "m3": "M2", "m2m3": "M2",
                  "aca": "ACA", "tandem": "tandem", "tandemcca": "tandem", "cervicalica": "cervical_ICA",
                  "cca": "cervical_ICA", "basilar": "basilar", "ba": "basilar",
                  "basilarvertebral": "basilar_vertebral", "abav": "basilar_vertebral", "bava": "basilar_vertebral",
                  "vertebral": "other_posterior", "va": "other_posterior",
                  "pca": "other_posterior", "otherposterior": "other_posterior"},
    "technique": {"aspiration": "aspiration", "adapt": "aspiration", "stentretriever": "stent_retriever",
                  "sr": "stent_retriever", "solumbra": "solumbra", "combined": "combined",
                  "combinedprocedure": "combined", "kombinovanyvykon": "combined", "none": "none",
                  "diagnostic": "diagnostic", "pag": "diagnostic"},
    "anesthesia": {"general": "general", "ga": "general", "sedation": "sedation", "cs": "sedation",
                   "consciousseda": "sedation", "local": "local", "la": "local"},
}


def norm(s) -> str:
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]", "", s.lower())


FUTURE = {c["key"]: c for c in future_columns()["columns"]}
OPTIONAL_COLUMNS = ["year"] + list(FUTURE)


def template_columns() -> list[str]:
    return [f.key for f in features() if f.type != "derived"] + ["mrs90"]


def _year(raw) -> float:
    s = str(raw).strip()[:4]
    return float(s) if s.isdigit() and 1990 <= int(s) <= 2100 else np.nan


def _parse(f, raw, issues: dict):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan
    s = str(raw).strip()
    if s == "" or re.fullmatch(r"(?i)(na|nan|null|unknown|-)", s):
        return np.nan
    key = f.key if f is not None else "mrs90"

    def bad():
        issues[key] = issues.get(key, 0) + 1
        return np.nan

    if key == "mrs90":
        try:
            v = float(s.replace(",", "."))
        except ValueError:
            return bad()
        return v if v.is_integer() and 0 <= v <= 6 else bad()
    if key == "tici":
        k = re.sub(r"^m?e?tici", "", norm(s))
        return TICI.get(k, np.nan) if k in TICI else bad()
    if key == "sex_male":
        return SEX.get(norm(s), np.nan) if norm(s) in SEX else bad()
    if key == "side_left":
        return SIDE.get(norm(s), np.nan) if norm(s) in SIDE else bad()
    if f.type == "binary":
        return BIN[norm(s)] if norm(s) in BIN else bad()
    if f.type == "categorical":
        direct = [o["value"] for o in f.spec["options"] if norm(o["value"]) == norm(s)]
        if direct:
            return direct[0]
        m = CAT.get(key, {}).get(norm(s))
        return m if m is not None else bad()
    try:
        v = float(s.replace(",", "."))
    except ValueError:
        return bad()
    if f.type == "ordinal":
        return v if any(float(o["value"]) == v for o in f.spec["options"]) else bad()
    lo, hi = f.spec.get("min"), f.spec.get("max")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        return bad()
    return v


def _future(spec: dict, raw, issues: dict):
    """A value of a column collected for future versions; outside its range or unreadable -> missing."""
    s = "" if raw is None else str(raw).strip()
    if s == "" or re.fullmatch(r"(?i)(na|nan|null|unknown|-)", s):
        return None if spec["type"] == "text" else np.nan
    if spec["type"] == "text":
        return s[:80]
    if spec["type"] == "binary":
        if norm(s) in BIN:
            return float(BIN[norm(s)])
        issues[spec["key"]] = issues.get(spec["key"], 0) + 1
        return np.nan
    try:
        v = float(s.replace(",", "."))
    except ValueError:
        v = np.nan
    lo, hi = spec.get("min", -np.inf), spec.get("max", np.inf)
    if np.isnan(v) or v < lo or v > hi or (spec["type"] == "ordinal" and not v.is_integer()):
        issues[spec["key"]] = issues.get(spec["key"], 0) + 1
        return np.nan
    return v


def read_site_csv(path) -> tuple[pd.DataFrame, dict]:
    """Parse a registry export in the MTOP template. Returns typed data and a mapping report."""
    raw = pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)
    by_norm = {norm(c): c for c in raw.columns}
    feats = {f.key: f for f in features() if f.type != "derived"}
    issues: dict = {}
    out = {}
    found, missing = [], []
    for key in template_columns():
        col = by_norm.get(norm(key))
        if col is None:
            missing.append(key)
            out[key] = np.full(len(raw), np.nan, dtype=object if key in CAT else float)
            continue
        found.append(key)
        out[key] = [_parse(feats.get(key), v, issues) for v in raw[col].tolist()]
    df = pd.DataFrame(out)
    for key, f in feats.items():
        if f.type != "categorical":
            df[key] = pd.to_numeric(df[key], errors="coerce").astype(float)
        else:
            df[key] = df[key].astype(object).where(df[key].notna(), None)
    df["mrs90"] = pd.to_numeric(df["mrs90"], errors="coerce").astype(float)
    ycol = by_norm.get("year") or by_norm.get("rok")
    if ycol is not None:
        df["year"] = [_year(v) for v in raw[ycol].tolist()]
        found.append("year")
    future_found = []
    for key, spec in FUTURE.items():
        col = by_norm.get(norm(key))
        if col is not None:
            df[key] = [_future(spec, v, issues) for v in raw[col].tolist()]
            future_found.append(key)
    found.extend(future_found)
    df.loc[df["occlusion"].isin(POSTERIOR), "side_left"] = 0.0
    df["nihss_change_24h"] = df["nihss_24h"] - df["nihss_adm"]
    df["good_outcome"] = (df["mrs90"] <= 2).astype(float).where(df["mrs90"].notna())
    df["death90"] = (df["mrs90"] == 6).astype(float).where(df["mrs90"].notna())
    report = {"rows": int(len(df)), "columns_found": found, "columns_missing": missing,
              "unrecognised_values": issues, "with_outcome": int(df["mrs90"].notna().sum()),
              # optional columns for future versions: patients with the value and a known outcome
              "future_recorded": {k: int((df[k].notna() & df["mrs90"].notna()).sum()) for k in future_found}}
    return df, report


def write_site_csv(df: pd.DataFrame, path) -> None:
    """Write registry rows (clean variable names) in the template, as a partner centre would."""
    tici_label = {0.0: "0", 1.0: "1", 2.0: "2a", 2.5: "2", 3.0: "2b", 4.0: "3"}
    cols = template_columns() + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    out = df[cols].copy()
    out["tici"] = out["tici"].map(lambda v: tici_label.get(v, "") if pd.notna(v) else "")
    for col in cols:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else (str(int(v)) if float(v).is_integer() else f"{v:g}"))
    out.fillna("").to_csv(path, index=False)
