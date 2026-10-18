"""Data preprocessing report for the Registry page (artifacts/public/preprocessing.json).

Modelled on the reporting set of the earlier MTOP pipeline (cohort flow table and diagram, outcome
availability, missingness table, predictor dictionary, baseline table by split and by outcome with
SMD, exclusion and data-quality logs, EDA distributions) and on the 2026 analysis (missingness by
era, who lacks the 90-day mRS, why cholesterol was dropped). Aggregates only: every published count
is >= privacy.min_cell or shown as "<min_cell"; no dates, IDs or hospital names.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .clean import (ANESTHESIA, ANTITHROMBOTIC, DEVICE, ETIOLOGY, HEART, HEART3, OCCLUSION, PROCEDURE, RANGES, SIDE, TICI)
from .cohort import flow, split_temporal, splits, stage_cohort
from .config import features, settings, stage_ids, stage_spec
from .features import design_matrix
from .paths import PUBLIC

ERAS = [(0, 2012, "2005–2012"), (2013, 2015, "2013–2015"), (2016, 2019, "2016–2019"), (2020, 2022, "2020–2022"),
        (2023, 3000, "2023–2025")]

# registry column -> (MTOP variables, role); role: predictor | outcome | source | split | id | unused
RAW_MAP = {
    "Patient ID": ([], "id"),
    "Date of stroke onset": (["year", "weekend"], "split"),
    "Time of stroke onset": (["onset_hour", "night_onset"], "predictor"),
    "Age": (["age"], "predictor"),
    "Sex": (["sex_male"], "predictor"),
    "ASPECTS": (["aspects"], "predictor"),
    "Occlusion site": (["occlusion"], "predictor"),
    "Hemisphere": (["side_left"], "predictor"),
    "Intervention procedure": (["technique", "stenting", "angioplasty", "ia_lysis"], "predictor"),
    "Stent combo": (["stenting"], "source"),
    "Extraction system": (["technique"], "source"),
    "Thrombolytics": ([], "unused"),
    "Procedure duration": (["procedure_min"], "predictor"),
    "TICI": (["tici"], "predictor"),
    "IVT": (["ivt"], "predictor"),
    "IVT in different hospital": (["ivt_external"], "predictor"),
    "Transfer from different hospital": (["transfer"], "predictor"),
    "IAT": (["ia_lysis"], "source"),
    "Stroke onset to IVT (min)": (["onset_to_ivt", "ivt_to_puncture"], "predictor"),
    "Stroke onset to puncture (min)": (["onset_to_puncture", "ivt_to_puncture"], "predictor"),
    "Stroke onset to recanalization (min)": (["onset_to_recan"], "predictor"),
    "Admission NIHSS": (["nihss_adm", "nihss_change_24h", "eni_24h"], "predictor"),
    "NIHSS 24h": (["nihss_24h", "nihss_change_24h", "eni_24h"], "predictor"),
    "NIHSS 7d": (["nihss_7d"], "predictor"),
    "SICH": (["sich"], "predictor"),
    "non-SICH": (["ich_other"], "predictor"),
    "mRS before": (["mrs_pre"], "predictor"),
    "90-day mRS": (["good_outcome", "death90"], "outcome"),
    "Anti-thrombotics before": (["antithrombotic"], "predictor"),
    "Statins before": (["statin"], "predictor"),
    "Hypertension": (["htn"], "predictor"),
    "DM": (["dm"], "predictor"),
    "Hyperlipidemia": (["hld"], "predictor"),
    "Smoking": (["smoking"], "predictor"),
    "Alcohol abusus": (["alcohol"], "predictor"),
    "BMI": (["bmi"], "predictor"),
    "Arrhythmia": (["arrhythmia"], "predictor"),
    "TIA before": (["tia_prior"], "predictor"),
    "CMP before": (["stroke_prior"], "predictor"),
    "Heart condition": (["heart3"], "predictor"),
    "Systolic BP": (["sbp", "pulse_pressure"], "predictor"),
    "Diastolic BP": (["dbp", "pulse_pressure"], "predictor"),
    "Glycemia": (["glucose", "hyperglycemia"], "predictor"),
    "Cholesterol": ([], "unused"),
    "Etiology": ([], "unused"),
    "Cerebral edema": (["edema"], "predictor"),
    "Decompression surgery": (["decompression"], "predictor"),
    "Anesthesia": (["anesthesia"], "predictor"),
}

# how each coded column's values were mapped (registry value -> model category)
CODINGS = {
    "Occlusion site": {k: (v or "missing") for k, v in OCCLUSION.items()},
    "Hemisphere": SIDE,
    "Intervention procedure": {k: ("diagnostic" if f[4] else "thrombectomy" if f[0] else "no thrombectomy")
                               + ("; stent" if f[1] else "") + ("; angioplasty" if f[2] else "") + ("; IA lysis" if f[3] else "")
                               for k, f in PROCEDURE.items()},
    "Extraction system": DEVICE,
    "Anti-thrombotics before": ANTITHROMBOTIC,
    "Heart condition": {k: HEART3[v] for k, v in HEART.items()},
    "Anesthesia": ANESTHESIA,
    "TICI": {k: ("2 (no subgrade)" if k == "2" else k) for k in TICI},
    "Etiology": ETIOLOGY,
}


def _pct(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(100 * float(x), 1)


def _cnt(n: int, min_cell: int):
    return int(n) if n >= min_cell or n == 0 else None


# ------------------------------------------------------------------------------- cohort flow
def cohort_flow(clean: pd.DataFrame, min_cell: int) -> dict:
    base, steps = flow(clean)
    out = {"steps": steps, "splits": {}, "stages": []}
    for label, t0 in splits().items():
        dev, te = split_temporal(base, t0)
        out["splits"][label] = {
            "test_from_year": t0,
            "dev": {"n": int(len(dev)), "good": int(dev["good_outcome"].sum()), "death": int(dev["death90"].sum()),
                    "years": [int(dev["year"].min()), int(dev["year"].max())]},
            "test": {"n": int(len(te)), "good": int(te["good_outcome"].sum()), "death": int(te["death90"].sum()),
                     "years": [int(te["year"].min()), int(te["year"].max())]},
        }
    prim = splits()["primary"]
    for stage in stage_ids():
        d = stage_cohort(base, stage)
        dev, te = split_temporal(d, prim)
        reasons = {}
        for req in stage_spec(stage)["requires"]:
            miss = base[base[req].isna()]
            if req.startswith("nihss_"):
                st = miss[f"{req}_status"].value_counts()
                for k in ("died", "not_assessable", "missing"):
                    if k in st:
                        reasons[f"{req}:{k}"] = _cnt(int(st[k]), min_cell)
            else:
                reasons[f"{req}:missing"] = _cnt(int(len(miss)), min_cell)
        out["stages"].append({"stage": stage, "n": int(len(d)), "dev": int(len(dev)), "test": int(len(te)),
                              "good": int(d["good_outcome"].sum()), "death": int(d["death90"].sum()),
                              "excluded": int(len(base) - len(d)), "reasons": reasons})
    return out


# ------------------------------------------------------------------------------- variables
def variable_map(clean: pd.DataFrame, raw_prof: dict | None) -> list[dict]:
    """Registry column -> MTOP variables, role, completeness and the coding of its values."""
    prof = {c["column"]: c for c in (raw_prof or {}).get("columns", [])}
    used_at = {f.key: f.stage for f in features()}
    rows = []
    for col, (vars_, role) in RAW_MAP.items():
        p = prof.get(col, {})
        stages = sorted({used_at[v] for v in vars_ if v in used_at}, key=lambda s: stage_ids().index(s))
        r = {"column": col, "variables": vars_, "role": role, "stage": stages[0] if stages else None,
             "missing_pct": p.get("missing_pct")}
        if col in CODINGS:
            counts = {v["value"]: v for v in p.get("values", [])}
            r["coding"] = [{"value": k, "to": v, "n": counts.get(k, {}).get("n"), "small": counts.get(k, {}).get("small", False),
                            "present": k in counts} for k, v in CODINGS[col].items()]
        rows.append(r)
    return rows


def GROUPS(d: pd.DataFrame):
    """Sociodemographic groups for the completeness check (labels match the front end)."""
    return [("women", (d["sex_male"] == 0).to_numpy()), ("men", (d["sex_male"] == 1).to_numpy()),
            ("<80", (d["age"] < 80).to_numpy()), ("80+", (d["age"] >= 80).to_numpy())]


def predictor_table(base: pd.DataFrame) -> list[dict]:
    """Every model input: stage, type, monotone sign (LightGBM), missingness overall, by era and by
    split, and how missing values are handled."""
    rows = []
    prim = splits()["primary"]
    dev, te = split_temporal(base, prim)
    for f in features():
        d = stage_cohort(base, f.stage)
        X = design_matrix(d, f.stage)
        col = f.columns[0]
        miss = X[col].isna()
        r = {"key": f.key, "stage": f.stage, "type": f.type, "monotone": f.monotone, "optional": bool(f.spec.get("optional")),
             "missing_pct": _pct(miss.mean()), "by_era": {}}
        for a, b, lab in ERAS:
            m = (d["year"] >= a) & (d["year"] <= b)
            r["by_era"][lab] = _pct(miss[m].mean()) if m.sum() >= 10 else None
        dv, tt = d["year"] < prim, d["year"] >= prim
        r["dev_pct"], r["test_pct"] = _pct(miss[dv].mean()), _pct(miss[tt].mean())
        # was the data as complete for women and men, younger and older patients? (TRIPOD+AI 7)
        r["by_group"] = {}
        for lab, m in GROUPS(d):
            r["by_group"][lab] = _pct(miss[m].mean()) if m.sum() >= 10 else None
        n_miss = int(miss.sum())
        if n_miss >= 10 and (~miss).sum() >= 10:
            r["good_if_missing"] = _pct(d.loc[miss, "good_outcome"].mean())
            r["good_if_known"] = _pct(d.loc[~miss, "good_outcome"].mean())
        structural = f.key in ("onset_to_ivt", "ivt_to_puncture")
        r["handling"] = "structural" if structural else ("derived" if f.type == "derived" else "missing")
        rows.append(r)
    return rows


# ------------------------------------------------------------------------------- outcome
def _smd(a: pd.Series, b: pd.Series):
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    if len(a) < 10 or len(b) < 10:
        return None
    sd = float(np.sqrt((a.var() + b.var()) / 2))
    return None if sd == 0 else round(float((a.mean() - b.mean()) / sd), 2)


def outcome_availability(clean: pd.DataFrame, min_cell: int) -> dict:
    """90-day mRS availability among adults (before the outcome exclusion): overall, by split and
    by year; and who lacks it (2019 onwards): SMD between patients without and with the outcome."""
    adults = clean[~(clean["age"] < settings()["cohort"]["min_age"])]
    prim = splits()["primary"]
    out = {"overall": None, "dev": None, "test": None, "years": [], "who_missing": []}

    def avail(g):
        n = len(g)
        k = int(g["mrs90"].isna().sum())
        return {"n": int(n), "missing": _cnt(k, min_cell), "missing_pct": _pct(k / n) if n else None}

    out["overall"] = avail(adults)
    out["dev"] = avail(adults[adults["year"] < prim])
    out["test"] = avail(adults[adults["year"] >= prim])
    for yr, g in adults.assign(yr=adults["year"].clip(lower=2007)).groupby("yr"):
        if len(g) >= min_cell:
            out["years"].append({"year": int(yr), "label": "2005–2007" if yr == 2007 else str(int(yr)), "n": int(len(g)),
                                 "missing_pct": _pct(g["mrs90"].isna().mean())})
    rec = adults[adults["year"] >= 2019]
    mis, obs = rec[rec["mrs90"].isna()], rec[rec["mrs90"].notna()]
    out["who_missing_n"] = {"missing": int(len(mis)), "observed": int(len(obs))}
    checks = [
        ("age", "Age", "Věk", lambda g: g["age"]),
        ("nihss", "NIHSS on admission", "NIHSS při přijetí", lambda g: g["nihss_adm"]),
        ("premrs", "Pre-stroke mRS ≥ 1", "mRS před CMP ≥ 1", lambda g: (g["mrs_pre"] >= 1).astype(float).where(g["mrs_pre"].notna())),
        ("ivt", "IV thrombolysis", "IV trombolýza", lambda g: g["ivt"]),
        ("transfer", "Transferred", "Překlad", lambda g: g["transfer"]),
        ("smoking_known", "Smoking status recorded", "Kouření zaznamenáno", lambda g: g["smoking"].notna().astype(float)),
        ("nihss24_known", "NIHSS at 24 h recorded", "NIHSS za 24 h zaznamenáno", lambda g: (g["nihss_24h_status"] == "assessed").astype(float)),
        ("nihss7_known", "NIHSS on day 7 recorded", "NIHSS 7. den zaznamenáno", lambda g: (g["nihss_7d_status"] == "assessed").astype(float)),
        ("tici2b3", "mTICI 2b–3", "mTICI 2b–3", lambda g: g["reperfusion_2b3"]),
        ("posterior", "Posterior circulation", "Zadní povodí", lambda g: g["occlusion"].isin(["basilar", "other_posterior"]).astype(float).where(g["occlusion"].notna())),
        ("diagnostic", "Diagnostic angiography only", "Pouze diagnostická angiografie", lambda g: g["diagnostic_only"]),
    ]
    if len(mis) >= min_cell and len(obs) >= min_cell:
        for key, en, cs, fn in checks:
            a, b = fn(mis), fn(obs)
            binary = set(pd.unique(pd.concat([a, b]).dropna())) <= {0.0, 1.0}
            row = {"key": key, "label": {"en": en, "cs": cs}, "binary": bool(binary), "smd": _smd(a, b)}
            if binary:
                row["missing"], row["observed"] = _pct(a.mean()), _pct(b.mean())
            else:
                row["missing"], row["observed"] = round(float(a.median()), 1), round(float(b.median()), 1)
            out["who_missing"].append(row)
    return out


# ------------------------------------------------------------------------------- distributions
def _hist(s: pd.Series, edges, labels, min_cell: int) -> list[dict]:
    g = pd.cut(s, edges, labels=labels, right=False)
    vc = g.value_counts().reindex(labels, fill_value=0)
    return [{"bin": str(k), "n": _cnt(int(v), min_cell), "small": bool(0 < v < min_cell)} for k, v in vc.items()]


def distributions(base: pd.DataFrame, min_cell: int) -> dict:
    mrs = base["mrs90"].value_counts().reindex(range(7), fill_value=0)
    tici_lab = base["tici"].map({0.0: "0", 1.0: "1", 2.0: "2a", 2.5: "2", 3.0: "2b", 4.0: "3"})
    tici = tici_lab.value_counts().reindex(["0", "1", "2a", "2", "2b", "3"], fill_value=0)
    return {
        "mrs90": [{"bin": str(k), "n": _cnt(int(v), min_cell), "small": bool(0 < v < min_cell)} for k, v in mrs.items()],
        "tici": [{"bin": str(k), "n": _cnt(int(v), min_cell), "small": bool(0 < v < min_cell)} for k, v in tici.items()]
        + [{"bin": "missing", "n": _cnt(int(base["tici"].isna().sum()), min_cell), "small": bool(0 < base["tici"].isna().sum() < min_cell)}],
        "nihss_adm": _hist(base["nihss_adm"], [0, 5, 10, 15, 20, 25, 30, 43], ["0–4", "5–9", "10–14", "15–19", "20–24", "25–29", "30–42"], min_cell),
        "age": _hist(base["age"], [18, 40, 50, 60, 70, 80, 90, 120], ["18–39", "40–49", "50–59", "60–69", "70–79", "80–89", "90+"], min_cell),
        "onset_to_puncture": _hist(base["onset_to_puncture"], [0, 120, 180, 240, 360, 720, 1e9],
                                   ["<2 h", "2–3 h", "3–4 h", "4–6 h", "6–12 h", "≥12 h"], min_cell),
        "technique": [{"bin": k, "n": _cnt(int(v), min_cell), "small": bool(0 < v < min_cell)}
                      for k, v in base["technique"].fillna("missing").value_counts().items()],
        # follow-up NIHSS: scores of assessed patients, and how the registry codes 43 (died) and 99
        # (not assessable) were counted; neither code is ever used as a score
        "nihss_24h": _hist(base["nihss_24h"], [0, 2, 5, 10, 15, 20, 25, 43], ["0–1", "2–4", "5–9", "10–14", "15–19", "20–24", "25–42"], min_cell),
        "nihss_7d": _hist(base["nihss_7d"], [0, 2, 5, 10, 15, 20, 25, 43], ["0–1", "2–4", "5–9", "10–14", "15–19", "20–24", "25–42"], min_cell),
        "nihss_24h_status": [{"bin": k, "n": _cnt(int(v), min_cell), "small": bool(0 < v < min_cell)} for k, v in
                             base["nihss_24h_status"].value_counts().reindex(["assessed", "died", "not_assessable", "missing"], fill_value=0).items()],
        "nihss_7d_status": [{"bin": k, "n": _cnt(int(v), min_cell), "small": bool(0 < v < min_cell)} for k, v in
                            base["nihss_7d_status"].value_counts().reindex(["assessed", "died", "not_assessable", "missing"], fill_value=0).items()],
    }


def cholesterol_table(base: pd.DataFrame, min_cell: int) -> list[dict]:
    """Why cholesterol is not a predictor: patients without it die about twice as often at a similar
    admission status (2016 onwards, when it was routinely recorded)."""
    d = base[base["year"] >= 2016]
    rows = []
    for key, m in (("measured", d["cholesterol"].notna()), ("missing", d["cholesterol"].isna())):
        g = d[m]
        if len(g) < min_cell:
            continue
        rows.append({"group": key, "n": int(len(g)), "age": round(float(g["age"].median()), 1),
                     "nihss": round(float(g["nihss_adm"].median()), 1), "good": _pct(g["good_outcome"].mean()),
                     "death": _pct(g["death90"].mean())})
    return rows


def build_preprocessing(clean: pd.DataFrame) -> dict:
    cfg = settings()
    min_cell = int(cfg["privacy"]["min_cell"])
    base, _ = flow(clean)
    rp_path = PUBLIC / "raw_profile.json"
    raw_prof = json.loads(rp_path.read_text(encoding="utf-8")) if rp_path.exists() else None
    out = {
        "min_cell": min_cell,
        "records": int(len(clean)),
        "flow": cohort_flow(clean, min_cell),
        "variables": variable_map(clean, raw_prof),
        "predictors": predictor_table(base),
        "outcome": outcome_availability(clean, min_cell),
        "distributions": distributions(base, min_cell),
        "cholesterol": cholesterol_table(base, min_cell),
        "eras": [e[2] for e in ERAS],
        "ranges": {k: [float(a), float(b)] for k, (a, b) in RANGES.items()},
    }
    with open(PUBLIC / "preprocessing.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"preprocessing: {len(out['variables'])} registry columns, {len(out['predictors'])} model inputs")
    return out


# ------------------------------------------------------------------------------- data dictionary
def data_dictionary_md() -> str:
    """docs/data_dictionary.md from config/features.yaml and the cleaning code's mappings: every model
    input and every registry column with its coding. No counts, so it is the same for any data."""
    from .config import stage_spec

    stage_title = {s: stage_spec(s)["title"]["en"] for s in stage_ids()}
    L = ["# MTOP data dictionary", "",
         "Generated by `python -m mtop preprocessing` from `config/features.yaml` and the cleaning code",
         "(`src/mtop/clean.py`); the same information, with counts, is on the app's Registry page",
         "(Data and preprocessing).", "",
         "**How inputs enter the models.** Unordered categories become one yes/no column per option except",
         "the reference option (one-hot coding). Missing values are never filled in before modelling: the",
         "regressions use the development median plus a missing-value indicator, the tree models learn which",
         "branch missing values take. Values outside a plausible range become missing and are listed with the",
         "patient ID in `reports/data_quality.xlsx` (local only). NIHSS 43 (died) and 99 (not assessable) at",
         "24 h and day 7 are status codes, never scores.", "",
         "## Model inputs", "",
         "| Input | Key | Stage | Type | Values | Notes |", "|---|---|---|---|---|---|"]
    for f in features():
        sp = f.spec
        if f.type in ("categorical", "ordinal") and sp.get("options"):
            vals = "; ".join(f"{o['value']} = {o['label']['en']}" for o in sp["options"])
        elif f.type == "binary":
            vals = "1 = yes, 0 = no" if f.key not in ("sex_male", "side_left") else ("1 = male, 0 = female" if f.key == "sex_male" else "1 = left, 0 = right or posterior")
        elif f.type == "derived":
            vals = sp.get("formula", "derived")
        else:
            unit = f" {sp['unit']['en']}" if isinstance(sp.get("unit"), dict) else ""
            vals = f"{sp.get('min', '')}–{sp.get('max', '')}{unit}"
        notes = []
        if f.monotone:
            notes.append(f"monotone in LightGBM ({'+' if f.monotone > 0 else '−'} for mRS 0–2)")
        if sp.get("optional"):
            notes.append("may be left unknown in the calculator")
        if sp.get("requires_ivt"):
            notes.append("only with IV thrombolysis")
        if isinstance(sp.get("help"), dict):
            notes.append(sp["help"]["en"])
        L.append(f"| {sp['label']['en']} | `{f.key}` | {stage_title[f.stage]} | {f.type} | {vals} | {'; '.join(notes)} |")
    L += ["", "## Registry columns", "", "| Registry column | Role | MTOP variables |", "|---|---|---|"]
    for col, (vars_, role) in RAW_MAP.items():
        L.append(f"| `{col}` | {role} | {', '.join(f'`{v}`' for v in vars_) or '–'} |")
    L += ["", "## Codings of the registry's values", ""]
    for col, mp in CODINGS.items():
        L += [f"**{col}**", "", "| Registry value | MTOP value |", "|---|---|"]
        L += [f"| `{k}` | {v} |" for k, v in mp.items()]
        L.append("")
    return "\n".join(L) + "\n"


def write_data_dictionary() -> None:
    from .paths import ROOT

    (ROOT / "docs" / "data_dictionary.md").write_text(data_dictionary_md(), encoding="utf-8")
