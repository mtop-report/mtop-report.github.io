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

from .clean import POSTERIOR_DETAIL, POSTERIOR_OCCLUSIONS
from .cohort import flow, split_temporal
from .config import features, settings
from .features import derived
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
    "occlusion": {"ICA": ("Intracranial ICA", "Intrakraniální ACI"), "tandem": ("Tandem (cervical ICA/CCA + intracranial)", "Tandem (krční ACI/ACC + intrakraniální)"),
                  "cervical_ICA": ("Cervical ICA / CCA", "Krční ACI / ACC"), "M1": ("MCA M1", "ACM M1"),
                  "M2": ("MCA M2 / M3", "ACM M2 / M3"),
                  "basilar": ("Basilar", "Basilaris"), "basilar_vertebral": ("Basilar + vertebral", "Basilaris + AV"),
                  "other_posterior": ("Vertebral / PCA", "AV / ACP")},
    "technique": {"aspiration": ("Thrombectomy: aspiration (ADAPT)", "Trombektomie: aspirace (ADAPT)"),
                  "stent_retriever": ("Thrombectomy: stent retriever", "Trombektomie: stentriever"),
                  "solumbra": ("Thrombectomy: Solumbra (aspiration + stent retriever)", "Trombektomie: Solumbra (aspirace + stentriever)"),
                  "combined": ("Thrombectomy: combined procedure (2+ techniques)", "Trombektomie: kombinovaný výkon (2+ techniky)"),
                  "none": ("No thrombectomy: stent, PTA or IA lysis", "Bez trombektomie: stent, PTA nebo IA lýza"),
                  "diagnostic": ("Diagnostic angiography only", "Pouze diagnostická angiografie")},
    "anesthesia": {"general": ("General", "Celková"), "sedation": ("Conscious sedation", "Analgosedace"), "local": ("Local only", "Lokální")},
    "ivt": {"yes": ("IVT", "IVT"), "no": ("No IVT", "Bez IVT")},
    "transfer": {"direct": ("Direct to FNO", "Přímo do FNO"), "transfer": ("Transferred from another hospital", "Překlad z jiné nemocnice")},
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
    d["era"] = pd.cut(d["year"], [0, 2012, 2015, 2019, 2022, 3000], labels=["2005–2012", "2013–2015", "2016–2019", "2020–2022", "2023–2025"])
    d["age_group"] = pd.cut(d["age"], [0, 59.999, 69.999, 79.999, 200], labels=["<60", "60–69", "70–79", "≥80"])
    d["nihss_group"] = pd.cut(d["nihss_adm"], [-1, 9, 15, 20, 99], labels=["0–9", "10–15", "16–20", "≥21"])
    anterior = ~d["occlusion"].isin(POSTERIOR_OCCLUSIONS)
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
            "solumbra": _pct((mt == "solumbra").mean()), "combined": _pct((mt == "combined").mean()),
            "stenting": _pct(g["stenting"].mean()),
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


# occlusion: the model's categories (config/features.yaml), built from the recorded site; ACA is its own
# category without a model term (fewer than min_cell patients)
OCC_GROUPS = {"ICA": ["ICA"], "tandem": ["tandem", "tandem_cca"], "cervical_ICA": ["cervical_ICA", "CCA"], "M1": ["M1"],
              "M2": ["M2", "M3"], "ACA": ["ACA"], "basilar": ["basilar"], "basilar_vertebral": ["basilar_vertebral"],
              "other_posterior": ["vertebral", "PCA"]}


def _feat(key: str) -> dict:
    return next(f.spec for f in features() if f.key == key)


def _opt(key: str, value) -> dict:
    """An option's label as the calculator and the data report show it (without its "· …" note)."""
    lab = next(o["label"] for o in _feat(key)["options"] if o["value"] == value)
    return {k: v.split(" · ")[0] if key == "occlusion" else v for k, v in lab.items()}


def _low(t: str) -> str:
    return t[0].lower() + t[1:] if len(t) > 1 and t[1].islower() else t


def _t1_specs() -> list[dict]:
    """Every model input (config/features.yaml), derived ones included, by the stage it becomes known,
    plus the 90-day outcomes. Categorical inputs keep the model's categories and labels. Each row: a section header, or a variable whose `fn` returns per patient either the
    value (kind "median") or 1/0 with NaN where not recorded (kind "pct"). `elig` limits a row to the
    patients it applies to (ASPECTS: anterior circulation; onset to IVT: thrombolysed)."""
    S = lambda en, cs: {"section": {"en": en, "cs": cs}}  # noqa: E731
    C = lambda var, en, cs, fn, elig=None: {"var": var, "label": {"en": en, "cs": cs}, "kind": "median", "fn": fn, "elig": elig}  # noqa: E731
    P = lambda var, en, cs, fn: {"var": var, "label": {"en": en, "cs": cs}, "kind": "pct", "fn": fn}  # noqa: E731
    col = lambda c: (lambda g: g[c])  # noqa: E731
    yn = lambda c: (lambda g: g[c].astype(float))  # noqa: E731
    eq = lambda c, v: (lambda g: (g[c] == v).astype(float).where(g[c].notna()))  # noqa: E731
    isin = lambda c, vs: (lambda g: g[c].isin(vs).astype(float).where(g[c].notna()))  # noqa: E731
    anterior = lambda g: ~g["occlusion"].isin(POSTERIOR_OCCLUSIONS)  # noqa: E731
    anterior_site = lambda g: g["occlusion_detail"].notna() & ~g["occlusion_detail"].isin(POSTERIOR_DETAIL)  # noqa: E731
    der = lambda key: (lambda g: derived(g, key))  # noqa: E731
    flab = lambda key: _feat(key)["label"]  # noqa: E731
    def opt_row(var, key, value, fn):
        f, o = flab(key), _opt(key, value)
        return P(var, f"{f['en']}: {_low(o['en'])}", f"{f['cs']}: {_low(o['cs'])}", fn)
    thrombolysed = lambda g: g["ivt"] == 1  # noqa: E731
    return [
        S("Before the procedure: patient", "Před výkonem: pacient"),
        C("age", "Age, years", "Věk, roky", col("age")),
        P("male", "Men", "Muži", yn("sex_male")),
        *[opt_row(f"mrs_pre{v}", "mrs_pre", v, eq("mrs_pre", v)) for v in (0, 1, 2)],
        P("mrs_pre3", "Pre-stroke mRS: 3–5 · moderate to severe disability", "mRS před CMP: 3–5 · středně těžké až těžké postižení",
          lambda g: (g["mrs_pre"] >= 3).astype(float).where(g["mrs_pre"].notna())),
        S("Before the procedure: the stroke", "Před výkonem: CMP"),
        C("nihss", "NIHSS on admission", "NIHSS při přijetí", col("nihss_adm")),
        C("aspects", "ASPECTS (anterior circulation)", "ASPECTS (přední povodí)", col("aspects"), anterior),
        *[opt_row(f"occ_{v}", "occlusion", v, isin("occlusion_detail", details)) for v, details in OCC_GROUPS.items()],
        {**P("left", "Left hemisphere (anterior circulation)", "Levá hemisféra (přední povodí)", yn("side_left")), "elig": anterior_site},
        S("Before the procedure: thrombolysis, route and timing", "Před výkonem: trombolýza, cesta a časy"),
        P("ivt", "IV thrombolysis", "IV trombolýza", yn("ivt")),
        P("ivt_external", "IV thrombolysis in another hospital", "IV trombolýza v jiné nemocnici", yn("ivt_external")),
        C("oti", "Onset to IV thrombolysis, min (thrombolysed)", "Od vzniku do IV trombolýzy, min (trombolyzovaní)", col("onset_to_ivt"), thrombolysed),
        P("transfer", "Transferred from another hospital", "Překlad z jiné nemocnice", yn("transfer")),
        C("otp", "Onset to groin puncture, min", "Od vzniku do punkce, min", col("onset_to_puncture")),
        C("itp", "IV thrombolysis to puncture, min (thrombolysed)", "Od IV trombolýzy do punkce, min (trombolyzovaní)", der("ivt_to_puncture"), thrombolysed),
        P("night", flab("night_onset")["en"], flab("night_onset")["cs"], der("night_onset")),
        P("weekend", "Onset at the weekend", "Vznik o víkendu", yn("weekend")),
        S("Before the procedure: on admission", "Před výkonem: při přijetí"),
        C("glucose", "Glucose, mmol/L", "Glykémie, mmol/l", col("glucose")),
        P("hyperglycemia", flab("hyperglycemia")["en"], flab("hyperglycemia")["cs"], der("hyperglycemia")),
        C("sbp", "Systolic BP, mmHg", "Systolický tlak, mmHg", col("sbp")),
        C("dbp", "Diastolic BP, mmHg", "Diastolický tlak, mmHg", col("dbp")),
        C("pulse_pressure", "Pulse pressure, mmHg", "Pulzní tlak, mmHg", der("pulse_pressure")),
        C("bmi", "Body mass index, kg/m²", "BMI, kg/m²", col("bmi")),
        S("Before the procedure: history and medication", "Před výkonem: anamnéza a medikace"),
        P("htn", "Hypertension", "Hypertenze", yn("htn")),
        P("dm", "Diabetes mellitus", "Diabetes mellitus", yn("dm")),
        P("hld", "Hyperlipidaemia", "Hyperlipidémie", yn("hld")),
        P("af", "Atrial fibrillation / arrhythmia", "Fibrilace síní / arytmie", yn("arrhythmia")),
        P("stroke_prior", "Previous stroke", "Předchozí CMP", yn("stroke_prior")),
        P("tia_prior", "Previous TIA", "Předchozí TIA", yn("tia_prior")),
        opt_row("coronary", "heart3", "coronary", eq("heart3", "coronary")),
        opt_row("heart_other", "heart3", "other", eq("heart3", "other")),
        P("smoking", "Smoker", "Kuřák", yn("smoking")),
        P("alcohol", "Alcohol abuse", "Abúzus alkoholu", yn("alcohol")),
        opt_row("antiplatelet", "antithrombotic", "antiplatelet", eq("antithrombotic", "antiplatelet")),
        opt_row("anticoag", "antithrombotic", "anticoagulant", eq("antithrombotic", "anticoagulant")),
        P("statin", "Statin before stroke", "Statin před CMP", yn("statin")),
        S("End of the procedure", "Konec výkonu"),
        P("mt", "Mechanical thrombectomy (any device)", "Mechanická trombektomie (jakýkoli prostředek)", isin("technique", ["aspiration", "stent_retriever", "solumbra", "combined"])),
        *[P(f"t_{v}", _opt("technique", v)["en"], _opt("technique", v)["cs"], eq("technique", v))
          for v in ("aspiration", "stent_retriever", "solumbra", "combined", "none", "diagnostic")],
        P("stenting", "Stent placed", "Zaveden stent", yn("stenting")),
        P("angioplasty", "Angioplasty", "Angioplastika", yn("angioplasty")),
        P("ia_lysis", "Intra-arterial thrombolysis", "Intraarteriální trombolýza", yn("ia_lysis")),
        C("proc", "Procedure duration, min", "Délka výkonu, min", col("procedure_min")),
        C("otr", "Onset to recanalisation, min", "Od vzniku do rekanalizace, min", col("onset_to_recan")),
        *[opt_row(f"an_{v}", "anesthesia", v, eq("anesthesia", v)) for v in ("general", "sedation", "local")],
        *[opt_row(f"tici_{v}", "tici", v, eq("tici", float(v))) for v in (0, 1, 2)],
        P("tici_2u", f"{flab('tici')['en']}: 2, no subgrade (registry)", f"{flab('tici')['cs']}: 2 bez podstupně (registr)", eq("tici", 2.5)),
        *[opt_row(f"tici_{v}", "tici", v, eq("tici", float(v))) for v in (3, 4)],
        P("tici", "Final mTICI 2b–3", "Výsledné mTICI 2b–3", yn("reperfusion_2b3")),
        S("24 hours after treatment", "24 hodin po léčbě"),
        C("nihss24", "NIHSS at 24 h", "NIHSS za 24 h", col("nihss_24h")),
        P("sich", "Symptomatic ICH", "Symptomatické ICH", yn("sich")),
        P("ich_other", "Asymptomatic ICH", "Asymptomatické ICH", yn("ich_other")),
        C("nihss_change", flab("nihss_change_24h")["en"], flab("nihss_change_24h")["cs"], der("nihss_change_24h")),
        P("eni", flab("eni_24h")["en"], flab("eni_24h")["cs"], der("eni_24h")),
        S("Day 7", "7. den"),
        C("nihss7", "NIHSS on day 7", "NIHSS 7. den", col("nihss_7d")),
        P("edema", "Cerebral oedema", "Mozkový edém", yn("edema")),
        P("decompression", "Decompressive craniectomy", "Dekompresivní kraniektomie", yn("decompression")),
        S("Outcome at 90 days", "Výsledek za 90 dní"),
        P("good", "90-day mRS 0–2", "mRS 0–2 za 90 dní", yn("good_outcome")),
        P("death", "90-day mortality", "Mortalita za 90 dní", yn("death90")),
    ]


def _t1_values(spec: dict, g: pd.DataFrame) -> pd.Series:
    v = spec["fn"](g)
    return v[spec["elig"](g)] if spec.get("elig") else v


def _t1_cell(spec: dict, g: pd.DataFrame, min_cell: int) -> dict:
    """One group's summary. A count under min_cell is reported only as `small`: neither the count nor
    the percentage it would give away is published."""
    v = _t1_values(spec, g)
    missing = _pct(v.isna().mean()) if len(v) else None
    if spec["kind"] == "median":
        return {"median": _med(v), "iqr": _iqr(v), "missing": missing}
    known = v.notna()
    cnt = int(v[known].sum())
    small = bool(0 < cnt < min_cell)
    return {"pct": None if small or not known.sum() else _pct(v[known].mean()),
            "n": None if small else cnt, "small": small, "missing": missing}


def _table1(base: pd.DataFrame, min_cell: int) -> dict:
    """Table 1 with standardised mean differences (SMD): development (2005-2022) vs validation
    (2023-2025) patients, and mRS 0-2 vs 3-6 (the earlier pipeline's baseline-by-outcome table).
    |SMD| > 0.1 is the usual mark of a relevant difference. Rows: every non-derived model input.
    Where a split holds a count under min_cell, the SMD of that split and the other half of the split
    (which would give the small count away by subtraction from All) are not published either."""
    dev, test = split_temporal(base)
    good, poor = base[base["good_outcome"] == 1], base[base["good_outcome"] == 0]
    groups = {"all": base, "dev": dev, "test": test, "good": good, "poor": poor}
    rows = []
    for spec in _t1_specs():
        if "section" in spec:
            rows.append(spec)
            continue
        smd = _smd_cont if spec["kind"] == "median" else _smd_prop
        out = {"var": spec["var"], "label": spec["label"], "kind": spec["kind"],
               "smd": smd(_t1_values(spec, dev), _t1_values(spec, test)),
               "smd_outcome": smd(_t1_values(spec, poor), _t1_values(spec, good)),
               **{k: _t1_cell(spec, g, min_cell) for k, g in groups.items()}}
        if spec["kind"] == "pct":
            # the two halves of a split add up to All: when one half is small, its partner is masked
            # too (All stays), and so is the SMD, which would give the small proportion away
            for a, b, smd_key in (("dev", "test", "smd"), ("good", "poor", "smd_outcome")):
                if out[a]["small"] or out[b]["small"]:
                    out[smd_key] = None
                    for k in (a, b):
                        if not out[k]["small"]:
                            out[k] = {"pct": None, "n": None, "small": False, "masked": True, "missing": out[k]["missing"]}
        rows.append(out)
    return {"n": {k: int(len(g)) for k, g in groups.items()}, "rows": rows}


def _period_table(d: pd.DataFrame, min_cell: int) -> dict:
    """Every Table 1 row by treatment period (the Era breakdown). A count under min_cell is published
    only as `small` (no percentage). Because the development periods add up to the development set
    and all periods to the cohort in Table 1, a lone small cell would follow by subtraction: then the
    smallest other development-period cell is withheld too (`masked`)."""
    eras = [(str(e), d[d["era"] == e]) for e in d["era"].cat.categories]
    eras = [(lab, g) for lab, g in eras if len(g) >= min_cell]
    test_from = int(settings()["split"]["temporal_test_from_year"])
    dev_idx = [i for i, (_, g) in enumerate(eras) if g["year"].max() < test_from]
    rows = []
    for spec in _t1_specs():
        if "section" in spec:
            rows.append(spec)
            continue
        cells = [_t1_cell(spec, g, min_cell) for _, g in eras]
        row = {"var": spec["var"], "label": spec["label"], "kind": spec["kind"]}
        if spec["kind"] == "median":
            row["values"] = [c["median"] for c in cells]
            row["iqr"] = [c["iqr"] for c in cells]
        else:
            small = [i for i, c in enumerate(cells) if c["small"]]
            small_dev = [i for i in small if i in dev_idx]
            test_small = any(i not in dev_idx for i in small)
            masked = []
            if len(small_dev) == 1 or (test_small and not small_dev):
                pool = [i for i in dev_idx if not cells[i]["small"] and cells[i]["n"]]
                if pool:
                    masked = [min(pool, key=lambda i: cells[i]["n"])]
            row["values"] = [None if (c["small"] or i in masked) else c["pct"] for i, c in enumerate(cells)]
            row["small"] = [bool(c["small"]) for c in cells]
            row["masked"] = [i in masked for i in range(len(cells))]
        rows.append(row)
    return {"periods": [{"label": lab, "n": int(len(g))} for lab, g in eras], "rows": rows}


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
        "periods": _period_table(d, min_cell),
    }
    PUBLIC.mkdir(parents=True, exist_ok=True)
    with open(PUBLIC / "explorer.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"explorer: {len(out['dimensions'])} breakdowns, {len(out['trends'])} years")
    return out
