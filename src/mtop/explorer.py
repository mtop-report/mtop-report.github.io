"""Aggregate, disclosure-controlled statistics for the public cohort explorer.

Rules (enforced here and re-checked by tests/test_privacy.py):
* one-way breakdowns, plus a single two-way table (age band x NIHSS band) whose cells are shown
  only with >= 3 * min_cell patients;
* a group is shown only if it has >= min_cell patients (default 10);
* full 7-level mRS distributions only for groups with >= 3 * min_cell patients;
* no dates, IDs, referring-hospital names, minima or maxima; percentages rounded to 0.1.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .cohort import flow, split_temporal
from .config import settings
from .paths import PUBLIC

PCT = 1


def _pct(x) -> float | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(100 * float(x), PCT)


def _med(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return None
    return float(np.round(s.median(), 1))


def _iqr(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return None
    q1, q3 = np.percentile(s, [25, 75])
    return [float(np.round(q1, 1)), float(np.round(q3, 1))]


def _group_stats(g: pd.DataFrame, min_cell: int) -> dict | None:
    n = len(g)
    if n < min_cell:
        return None
    out = {"n": n, "good": _pct(g["good_outcome"].mean()), "death": _pct(g["death90"].mean())}
    if n >= 3 * min_cell:
        dist = g["mrs90"].value_counts(normalize=True).reindex(range(7), fill_value=0.0)
        out["dist"] = [_pct(v) for v in dist.to_numpy()]
    return out


DIMENSIONS = {
    "era": ("Era", "Období"),
    "age_group": ("Age", "Věk"),
    "sex": ("Sex", "Pohlaví"),
    "mrs_pre_group": ("Pre-stroke mRS", "mRS před CMP"),
    "nihss_group": ("NIHSS on admission", "NIHSS při přijetí"),
    "glucose_group": ("Glucose on admission", "Glykémie při přijetí"),
    "af": ("Atrial fibrillation / arrhythmia", "Fibrilace síní / arytmie"),
    "dm": ("Diabetes", "Diabetes"),
    "anticoag": ("Anticoagulated before stroke", "Antikoagulace před CMP"),
    "occlusion": ("Occlusion site", "Místo okluze"),
    "aspects_group": ("ASPECTS (anterior circulation, 2016+)", "ASPECTS (přední povodí, od 2016)"),
    "otp_group": ("Onset to puncture", "Od vzniku do punkce"),
    "ivt": ("IV thrombolysis", "IV trombolýza"),
    "transfer": ("Admission route", "Způsob přijetí"),
    "technique": ("Technique", "Technika"),
    "anesthesia": ("Anaesthesia", "Anestezie"),
    "stenting": ("Acute stenting", "Akutní stenting"),
    "proc_group": ("Procedure duration", "Délka výkonu"),
    "tici_group": ("Final mTICI", "Výsledné mTICI"),
    "nihss_change": ("NIHSS change in the first 24 h", "Změna NIHSS za prvních 24 h"),
    "sich": ("Symptomatic ICH", "Symptomatické ICH"),
}

DIM_NOTES = {
    "nihss_change": ("Patients with an assessed NIHSS at 24 h only (alive and assessable), so deaths within the first day are not included.",
                     "Jen pacienti s hodnoceným NIHSS za 24 h (žijící a hodnotitelní), úmrtí během prvního dne proto nejsou zahrnuta."),
    "aspects_group": ("ASPECTS is recorded from 2016 and scores the middle cerebral artery territory, so posterior-circulation strokes are left out here.",
                      "ASPECTS se eviduje od roku 2016 a hodnotí povodí arteria cerebri media, proto zde CMP v zadním povodí nejsou."),
    "proc_group": ("From groin puncture to recanalisation or the end of the procedure.", "Od punkce do rekanalizace nebo ukončení výkonu."),
}

LABELS = {
    "age_group": {"<60": ("<60", "<60"), "60–69": ("60–69", "60–69"), "70–79": ("70–79", "70–79"), "≥80": ("≥80", "≥80")},
    "nihss_group": {"0–9": ("0–9", "0–9"), "10–15": ("10–15", "10–15"), "16–20": ("16–20", "16–20"), "≥21": ("≥21", "≥21")},
    "aspects_group": {"0–5": ("0–5", "0–5"), "6–7": ("6–7", "6–7"), "8–10": ("8–10", "8–10")},
    "otp_group": {"≤150 min": ("≤150 min", "≤150 min"), "151–240": ("151–240 min", "151–240 min"),
                  "241–360": ("241–360 min", "241–360 min"), "361–720": ("6–12 h", "6–12 h"), ">720": (">12 h", ">12 h")},
    "tici_group": {"0–1": ("0–1", "0–1"), "2a": ("2a", "2a"), "2": ("2 (not subgraded)", "2 (bez podskupiny)"),
                   "2b": ("2b", "2b"), "3": ("3", "3")},
    "occlusion": {"ICA": ("Intracranial ICA / T", "Intrakraniální ACI / T"), "M1": ("MCA M1", "ACM M1"),
                  "distal": ("MCA M2–M3 / ACA", "ACM M2–M3 / ACA"), "cervical_ICA": ("Cervical ICA/CCA (± tandem)", "Krční ACI/ACC (± tandem)"),
                  "basilar": ("Basilar", "Basilaris"), "other_posterior": ("Vertebral / PCA", "AV / ACP")},
    "technique": {"aspiration": ("Aspiration (ADAPT)", "Aspirace (ADAPT)"), "stent_retriever": ("Stent retriever", "Stentriever"),
                  "combined": ("Combined", "Kombinovaná"), "none": ("No mechanical MT", "Bez mechanické MT")},
    "anesthesia": {"general": ("General", "Celková"), "sedation": ("Conscious sedation", "Analgosedace"), "local": ("Local only", "Lokální")},
    "ivt": {"yes": ("IVT", "IVT"), "no": ("No IVT", "Bez IVT")},
    "transfer": {"direct": ("Direct to FNO", "Přímo do FNO"), "transfer": ("Drip-and-ship", "Drip-and-ship")},
    "sex": {"male": ("Men", "Muži"), "female": ("Women", "Ženy")},
    "sich": {"no": ("No sICH", "Bez sICH"), "yes": ("sICH", "sICH")},
    "mrs_pre_group": {"0": ("0 · no symptoms", "0 · bez příznaků"), "1": ("1", "1"), "2": ("2", "2"), "≥3": ("≥3", "≥3")},
    "glucose_group": {"<7.8": ("<7.8 mmol/L", "<7,8 mmol/l"), "7.8–11.0": ("7.8–11.0 mmol/L", "7,8–11,0 mmol/l"),
                      "≥11.1": ("≥11.1 mmol/L", "≥11,1 mmol/l")},
    "proc_group": {"≤30": ("≤30 min", "≤30 min"), "31–60": ("31–60 min", "31–60 min"), "61–90": ("61–90 min", "61–90 min"),
                   ">90": (">90 min", ">90 min")},
    "nihss_change": {"improved": ("Improved by ≥4", "Zlepšení o ≥4"), "stable": ("Within 3 points", "Změna do 3 bodů"),
                     "worse": ("Worsened by ≥4", "Zhoršení o ≥4")},
    "af": {"no": ("No AF / arrhythmia", "Bez FS / arytmie"), "yes": ("AF / arrhythmia", "FS / arytmie")},
    "dm": {"no": ("No diabetes", "Bez diabetu"), "yes": ("Diabetes", "Diabetes")},
    "anticoag": {"no": ("Not anticoagulated", "Bez antikoagulace"), "yes": ("Anticoagulated", "Antikoagulace")},
    "stenting": {"no": ("No acute stent", "Bez akutního stentu"), "yes": ("Acute stent", "Akutní stent")},
}


def _derive(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["era"] = pd.cut(d["year"], [0, 2014, 2017, 2020, 3000], labels=["2005–2014", "2015–2017", "2018–2020", "2021–2025"])
    d["age_group"] = pd.cut(d["age"], [0, 59.999, 69.999, 79.999, 200], labels=["<60", "60–69", "70–79", "≥80"])
    d["nihss_group"] = pd.cut(d["nihss_adm"], [-1, 9, 15, 20, 99], labels=["0–9", "10–15", "16–20", "≥21"])
    anterior = ~d["occlusion"].isin(["basilar", "other_posterior"])
    d["aspects_group"] = pd.cut(d["aspects"].where((d["year"] >= 2016) & anterior), [-1, 5, 7, 10], labels=["0–5", "6–7", "8–10"])
    d["otp_group"] = pd.cut(d["onset_to_puncture"], [0, 150, 240, 360, 720, 1e9],
                            labels=["≤150 min", "151–240", "241–360", "361–720", ">720"])
    d["tici_group"] = d["tici"].map({0.0: "0–1", 1.0: "0–1", 2.0: "2a", 2.5: "2", 3.0: "2b", 4.0: "3"})
    d["ivt_lbl"] = d["ivt"].map({1.0: "yes", 0.0: "no"})
    d["transfer_lbl"] = d["transfer"].map({1.0: "transfer", 0.0: "direct"})
    d["sex_lbl"] = d["sex_male"].map({1.0: "male", 0.0: "female"})
    d["sich_lbl"] = d["sich"].map({1.0: "yes", 0.0: "no"})
    d["mrs_pre_group"] = pd.cut(d["mrs_pre"], [-1, 0, 1, 2, 6], labels=["0", "1", "2", "≥3"])
    d["glucose_group"] = pd.cut(d["glucose"], [0, 7.75, 11.05, 100], labels=["<7.8", "7.8–11.0", "≥11.1"])
    d["proc_group"] = pd.cut(d["procedure_min"], [0, 30, 60, 90, 1e9], labels=["≤30", "31–60", "61–90", ">90"])
    d["nihss_change"] = pd.cut(d["nihss_24h"] - d["nihss_adm"], [-100, -4, 3, 100], labels=["improved", "stable", "worse"])
    for col, src in (("af_lbl", "arrhythmia"), ("dm_lbl", "dm"), ("anticoag_lbl", "anticoag"), ("stenting_lbl", "stenting")):
        d[col] = d[src].map({1.0: "yes", 0.0: "no"})
    return d


DIM_COL = {"ivt": "ivt_lbl", "transfer": "transfer_lbl", "sex": "sex_lbl", "sich": "sich_lbl", "af": "af_lbl", "dm": "dm_lbl",
           "anticoag": "anticoag_lbl", "stenting": "stenting_lbl"}


def _breakdown(d: pd.DataFrame, dim: str, min_cell: int) -> list[dict]:
    rows = []
    col = d[DIM_COL.get(dim, dim)]
    order = list(col.cat.categories) if hasattr(col, "cat") else list(LABELS.get(dim, {}).keys()) or sorted(col.dropna().unique())
    for val in order:
        g = d[col == val]
        st = _group_stats(g, min_cell)
        lab = LABELS.get(dim, {}).get(val, (str(val), str(val)))
        if st is None:
            rows.append({"key": str(val), "label": {"en": lab[0], "cs": lab[1]}, "suppressed": True})
        else:
            rows.append({"key": str(val), "label": {"en": lab[0], "cs": lab[1]}, **st})
    return rows


def _trend_rows(d: pd.DataFrame, min_cell: int) -> list[dict]:
    d = d.copy()
    d["yr"] = d["year"].clip(lower=2007)  # 2005-2006 (n=5) merged into 2007
    rows = []
    for yr, g in d.groupby("yr"):
        if len(g) < min_cell:
            continue
        mt = g["technique"]
        r = {
            "year": int(yr), "label": "2005–2007" if yr == 2007 else str(int(yr)), "n": int(len(g)),
            "good": _pct(g["good_outcome"].mean()), "death": _pct(g["death90"].mean()),
            "age": _med(g["age"]), "nihss": _med(g["nihss_adm"]), "otp": _med(g["onset_to_puncture"]),
            "ivt": _pct(g["ivt"].mean()), "transfer": _pct(g["transfer"].mean()),
            "tici2b3": _pct(g["reperfusion_2b3"].mean()),
            "ga": _pct((g["anesthesia"] == "general").mean()) if g["anesthesia"].notna().mean() > 0.5 else None,
            "aspiration": _pct((mt == "aspiration").mean()), "stent_retriever": _pct((mt == "stent_retriever").mean()),
            "combined": _pct((mt == "combined").mean()), "stenting": _pct(g["stenting"].mean()),
            "age80": _pct((g["age"] >= 80).mean()),
            "sich": _pct(g["sich"].mean()),
            "proc": _med(g["procedure_min"]),
        }
        rows.append(r)
    return rows


def _age_nihss(d: pd.DataFrame, min_cell: int) -> dict:
    """Outcome by age band and NIHSS band. The only two-way table: cells need >= 3 * min_cell."""
    ages, nihss = list(d["age_group"].cat.categories), list(d["nihss_group"].cat.categories)
    cells = []
    for a in ages:
        row = []
        for b in nihss:
            g = d[(d["age_group"] == a) & (d["nihss_group"] == b)]
            if len(g) < 3 * min_cell:
                row.append({"suppressed": True})
            else:
                row.append({"n": int(len(g)), "good": _pct(g["good_outcome"].mean()), "death": _pct(g["death90"].mean())})
        cells.append(row)
    return {"rows": ages, "cols": nihss, "cells": cells, "min_n": 3 * min_cell}


def _time_curve(d: pd.DataFrame, min_cell: int, reperfused_only: bool) -> list[dict]:
    x = d if not reperfused_only else d[d["reperfusion_2b3"] == 1]
    bins = [0, 120, 180, 240, 300, 360, 480, 720, 1e9]
    labels = ["≤2 h", "2–3 h", "3–4 h", "4–5 h", "5–6 h", "6–8 h", "8–12 h", ">12 h"]
    grp = pd.cut(x["onset_to_puncture"], bins, labels=labels)
    rows = []
    for lab in labels:
        g = x[grp == lab]
        if len(g) < min_cell:
            rows.append({"bin": lab, "suppressed": True})
            continue
        p = g["good_outcome"].mean()
        se = np.sqrt(p * (1 - p) / len(g))
        rows.append({"bin": lab, "n": int(len(g)), "good": _pct(p),
                     "lo": _pct(max(0, p - 1.96 * se)), "hi": _pct(min(1, p + 1.96 * se))})
    return rows


def _smd_cont(a: pd.Series, b: pd.Series):
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return None
    sd = float(np.sqrt((a.var() + b.var()) / 2))
    return None if sd == 0 else round(float((b.mean() - a.mean()) / sd), 2)


def _smd_prop(a: pd.Series, b: pd.Series):
    a, b = a.dropna(), b.dropna()
    if not len(a) or not len(b):
        return None
    p, q = float(a.mean()), float(b.mean())
    sd = float(np.sqrt((p * (1 - p) + q * (1 - q)) / 2))
    return None if sd == 0 else round((q - p) / sd, 2)


def _table1(base: pd.DataFrame, min_cell: int) -> dict:
    """Table 1 with the standardised mean difference (SMD) between the development (<= 2020) and
    validation (2021+) cohorts; |SMD| > 0.1 is the usual mark of a relevant shift in case mix."""
    base = base.assign(aspects_anterior=base["aspects"].where(~base["occlusion"].isin(["basilar", "other_posterior"])))
    dev, test = split_temporal(base)
    groups = {"all": base, "dev": dev, "test": test,
              "good": base[base["good_outcome"] == 1], "poor": base[base["good_outcome"] == 0]}

    def cont(name, en, cs, col, digits=0):
        return {"var": name, "label": {"en": en, "cs": cs}, "kind": "median", "smd": _smd_cont(dev[col], test[col]),
                **{k: {"median": _med(g[col]), "iqr": _iqr(g[col]), "missing": _pct(g[col].isna().mean())}
                   for k, g in groups.items()}}

    def prop(name, en, cs, mask_fn):
        out = {"var": name, "label": {"en": en, "cs": cs}, "kind": "pct", "smd": _smd_prop(mask_fn(dev), mask_fn(test))}
        for k, g in groups.items():
            m = mask_fn(g)
            known = m.notna()
            cnt = int(m[known].sum())
            out[k] = {"pct": _pct(m[known].mean()) if known.sum() else None,
                      "n": cnt if cnt >= min_cell or cnt == 0 else None,
                      "small": bool(0 < cnt < min_cell)}
        return out

    yn = lambda col: (lambda g: g[col])  # noqa: E731
    rows = [
        cont("age", "Age, years", "Věk, roky", "age"),
        prop("male", "Men", "Muži", yn("sex_male")),
        cont("nihss", "NIHSS on admission", "NIHSS při přijetí", "nihss_adm"),
        cont("aspects", "ASPECTS (anterior circulation)", "ASPECTS (přední povodí)", "aspects_anterior"),
        prop("premrs01", "Pre-stroke mRS 0–1", "mRS před CMP 0–1", lambda g: (g["mrs_pre"] <= 1).astype(float).where(g["mrs_pre"].notna())),
        prop("htn", "Hypertension", "Hypertenze", yn("htn")),
        prop("dm", "Diabetes mellitus", "Diabetes mellitus", yn("dm")),
        prop("af", "Atrial fibrillation / arrhythmia", "Fibrilace síní / arytmie", yn("arrhythmia")),
        prop("stroke_prior", "Previous stroke", "Předchozí CMP", yn("stroke_prior")),
        prop("anticoag", "Anticoagulated before stroke", "Antikoagulace před CMP", yn("anticoag")),
        cont("glucose", "Glucose, mmol/L", "Glykémie, mmol/l", "glucose"),
        cont("sbp", "Systolic BP, mmHg", "Systolický tlak, mmHg", "sbp"),
        prop("posterior", "Posterior circulation", "Zadní povodí", lambda g: g["occlusion"].isin(["basilar", "other_posterior"]).astype(float).where(g["occlusion"].notna())),
        prop("ivt", "IV thrombolysis", "IV trombolýza", yn("ivt")),
        prop("transfer", "Drip-and-ship", "Drip-and-ship", yn("transfer")),
        cont("otp", "Onset to puncture, min", "Od vzniku do punkce, min", "onset_to_puncture"),
        cont("proc", "Procedure duration, min", "Délka výkonu, min", "procedure_min"),
        prop("ga", "General anaesthesia", "Celková anestezie", lambda g: (g["anesthesia"] == "general").astype(float).where(g["anesthesia"].notna())),
        prop("tici", "mTICI 2b–3", "mTICI 2b–3", yn("reperfusion_2b3")),
        prop("sich", "Symptomatic ICH", "Symptomatické ICH", yn("sich")),
        cont("nihss24", "NIHSS at 24 h", "NIHSS za 24 h", "nihss_24h"),
        prop("good", "90-day mRS 0–2", "mRS 0–2 za 90 dní", yn("good_outcome")),
        prop("death", "90-day mortality", "Mortalita za 90 dní", yn("death90")),
    ]
    return {"n": {k: int(len(g)) for k, g in groups.items()}, "rows": rows}


def build_explorer(clean: pd.DataFrame) -> dict:
    cfg = settings()
    min_cell = int(cfg["privacy"]["min_cell"])
    base, steps = flow(clean)
    d = _derive(base)
    years = d["year"].dropna()
    reg = _derive(clean[clean["age"] >= 18])
    registry_years = []
    for yr, g in reg.assign(yr=reg["year"].clip(lower=2007)).groupby("yr"):
        if len(g) >= min_cell:
            registry_years.append({"year": int(yr), "label": "2005–2007" if yr == 2007 else str(int(yr)),
                                   "n": int(len(g)),
                                   "outcome_missing": _pct(g["mrs90"].isna().mean()),
                                   "nihss24_missing": _pct((g["nihss_24h_status"] != "assessed").mean())})
    out = {
        "min_cell": min_cell,
        "flow": steps,
        "period": [int(years.min()), int(years.max())],
        "kpis": {
            "n": int(len(d)),
            "age_median": _med(d["age"]), "nihss_median": _med(d["nihss_adm"]),
            "ivt": _pct(d["ivt"].mean()), "tici2b3": _pct(d["reperfusion_2b3"].mean()),
            "otp_median": _med(d["onset_to_puncture"]), "good": _pct(d["good_outcome"].mean()),
            "death": _pct(d["death90"].mean()), "sich": _pct(d["sich"].mean()),
        },
        "overall": _group_stats(d, min_cell),
        "dimensions": {k: {"label": {"en": v[0], "cs": v[1]}, "groups": _breakdown(d, k, min_cell),
                           **({"note": {"en": DIM_NOTES[k][0], "cs": DIM_NOTES[k][1]}} if k in DIM_NOTES else {})}
                       for k, v in DIMENSIONS.items()},
        "time_curve": {"all": _time_curve(d, min_cell, False), "reperfused": _time_curve(d, min_cell, True)},
        "age_nihss": _age_nihss(d, min_cell),
        "trends": _trend_rows(d, min_cell),
        "registry_years": registry_years,
        "table1": _table1(base, min_cell),
    }
    PUBLIC.mkdir(parents=True, exist_ok=True)
    with open(PUBLIC / "explorer.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"explorer: {len(out['dimensions'])} breakdowns, {len(out['trends'])} years")
    return out
