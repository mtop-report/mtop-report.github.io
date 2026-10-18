"""Data behind the local data studio: every version of the registry, the cleaning log mapped onto
its cells, labels in English and Czech. No Streamlit here, so it can be tested without it.

Versions
  raw       the registry export as recorded (after the active corrections), 48 columns
  clean     every record after the cleaning rules (standard names and codes)
  cohort    the analysed patients (adults with a 90-day mRS) with their temporal split
  <stage>   the model inputs of one stage cohort (admission, procedure, h24, d7)

Cell marks: missing (the cleaning set the value to missing), flag (kept, listed for review),
repaired (changed automatically), kept (record kept, not analysed), corrected (changed by a
correction), confirmed (checked at the source: the value is right).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .clean import CHECK_GROUPS, RAW_COLUMNS, clean_registry
from .cohort import flow, split_temporal, stage_cohort
from .config import features, settings, stage_features, stage_ids, stage_spec
from .corrections import apply_corrections, load_corrections, norm_value
from .features import derived

LANGS = ("en", "cs")

REGISTRY_LABELS = {
    "Patient ID": ("Patient ID", "ID pacienta"), "Date of stroke onset": ("Date of stroke onset", "Datum vzniku CMP"),
    "Time of stroke onset": ("Time of stroke onset", "Čas vzniku CMP"), "Age": ("Age", "Věk"), "Sex": ("Sex", "Pohlaví"),
    "ASPECTS": ("ASPECTS", "ASPECTS"), "Occlusion site": ("Occlusion site", "Místo okluze"), "Hemisphere": ("Hemisphere", "Hemisféra"),
    "Intervention procedure": ("Intervention procedure", "Typ výkonu"), "Stent combo": ("Stent", "Stent"),
    "Extraction system": ("Extraction system", "Extrakční systém"), "Thrombolytics": ("Thrombolytic", "Trombolytikum"),
    "Procedure duration": ("Procedure duration (min)", "Délka výkonu (min)"), "TICI": ("TICI", "TICI"), "IVT": ("IVT", "IVT"),
    "IVT in different hospital": ("IVT in a different hospital", "IVT v jiné nemocnici"),
    "Transfer from different hospital": ("Transfer from a different hospital", "Překlad z jiné nemocnice"),
    "IAT": ("Intra-arterial thrombolysis (IAT)", "Intraarteriální trombolýza (IAT)"),
    "Stroke onset to IVT (min)": ("Onset to IVT (min)", "Od vzniku do IVT (min)"),
    "Stroke onset to puncture (min)": ("Onset to puncture (min)", "Od vzniku do punkce (min)"),
    "Stroke onset to recanalization (min)": ("Onset to recanalisation (min)", "Od vzniku do rekanalizace (min)"),
    "Admission NIHSS": ("NIHSS on admission", "NIHSS při přijetí"), "NIHSS 24h": ("NIHSS at 24 h", "NIHSS za 24 h"),
    "NIHSS 7d": ("NIHSS on day 7", "NIHSS 7. den"), "SICH": ("Symptomatic ICH", "Symptomatické ICH"),
    "non-SICH": ("Asymptomatic ICH", "Asymptomatické ICH"), "mRS before": ("Pre-stroke mRS", "mRS před CMP"),
    "90-day mRS": ("90-day mRS", "mRS za 90 dní"), "Anti-thrombotics before": ("Antithrombotics before", "Antitrombotika před CMP"),
    "Statins before": ("Statins before", "Statiny před CMP"), "Hypertension": ("Hypertension", "Hypertenze"),
    "DM": ("Diabetes mellitus", "Diabetes mellitus"), "Hyperlipidemia": ("Hyperlipidaemia", "Hyperlipidémie"),
    "Smoking": ("Smoking", "Kouření"), "Alcohol abusus": ("Alcohol abuse", "Abúzus alkoholu"), "BMI": ("BMI", "BMI"),
    "Arrhythmia": ("Arrhythmia", "Arytmie"), "TIA before": ("Previous TIA", "TIA v anamnéze"),
    "CMP before": ("Previous stroke", "CMP v anamnéze"), "Heart condition": ("Heart condition", "Srdeční onemocnění"),
    "Systolic BP": ("Systolic BP", "Systolický TK"), "Diastolic BP": ("Diastolic BP", "Diastolický TK"),
    "Glycemia": ("Glucose", "Glykémie"), "Cholesterol": ("Cholesterol", "Cholesterol"), "Etiology": ("Aetiology", "Etiologie"),
    "Cerebral edema": ("Cerebral oedema", "Mozkový edém"), "Decompression surgery": ("Decompressive craniectomy", "Dekompresní kraniektomie"),
    "Anesthesia": ("Anaesthesia", "Anestezie"),
}

EXTRA_LABELS = {
    "patient_id": ("Patient ID", "ID pacienta"), "onset_date": ("Onset date", "Datum vzniku"), "year": ("Year", "Rok"),
    "mrs90": ("90-day mRS", "mRS za 90 dní"), "occlusion_detail": ("Occlusion site, detailed", "Místo okluze, podrobně"),
    "occlusion_raw": ("Occlusion site as recorded", "Místo okluze dle registru"),
    "recanalised_at_angio": ("Artery already open at angiography", "Tepna při angiografii již průchodná"),
    "side": ("Hemisphere", "Hemisféra"), "posterior": ("Posterior circulation", "Zadní povodí"),
    "iat": ("IAT column", "Sloupec IAT"), "anticoag": ("Anticoagulant", "Antikoagulancia"),
    "antiplatelet": ("Antiplatelet", "Antiagregancia"), "heart": ("Heart condition, detailed", "Srdeční onemocnění, podrobně"),
    "coronary": ("Coronary heart disease", "ICHS"), "heart_structural": ("Structural heart disease", "Strukturální onemocnění srdce"),
    "cholesterol": ("Cholesterol", "Cholesterol"), "referring_hospital": ("Referring hospital", "Odesílající nemocnice"),
    "agent": ("Thrombolytic agent", "Trombolytikum"), "tenecteplase": ("Tenecteplase", "Tenekteplasa"),
    "procedure_raw": ("Procedure as recorded", "Výkon dle registru"), "mt_label": ("Thrombectomy in the label", "Trombektomie v označení"),
    "stent_proc": ("Stent in the label", "Stent v označení"), "ia_lysis_proc": ("IA lysis in the label", "IA lýza v označení"),
    "diagnostic_only": ("Diagnostic angiography only", "Pouze diagnostická angiografie"),
    "device_raw": ("Extraction system as recorded", "Extrakční systém dle registru"),
    "mt_performed": ("Thrombectomy performed", "Trombektomie provedena"), "tici_label": ("TICI", "TICI"),
    "reperfusion_2b3": ("Reperfusion TICI 2b–3", "Reperfuze TICI 2b–3"),
    "time_conflict": ("Impossible time order", "Nemožné pořadí časů"),
    "nihss_24h_status": ("NIHSS 24 h status", "Stav NIHSS 24 h"), "nihss_7d_status": ("NIHSS day 7 status", "Stav NIHSS 7. den"),
    "etiology": ("Aetiology", "Etiologie"), "good_outcome": ("mRS 0–2 at 90 days", "mRS 0–2 za 90 dní"),
    "death90": ("Death by 90 days", "Úmrtí do 90 dní"), "split": ("Temporal split", "Časové rozdělení"),
    "stage_cohorts": ("Stage cohorts", "Kohorty fází"), "excluded": ("Not analysed because", "Neanalyzováno, protože"),
    "_issues": ("Issues", "Nálezy"),
}

# cleaning-log column -> registry columns and cleaned columns it concerns
ISSUE_RAW = {
    "age": ["Age"], "aspects": ["ASPECTS"], "bmi": ["BMI"], "cholesterol": ["Cholesterol"], "dbp": ["Diastolic BP"],
    "sbp": ["Systolic BP"], "sbp/dbp": ["Systolic BP", "Diastolic BP"], "glucose": ["Glycemia"], "ivt": ["IVT"],
    "mrs90": ["90-day mRS"], "mrs_pre": ["mRS before"], "nihss_adm": ["Admission NIHSS"], "nihss_24h": ["NIHSS 24h"],
    "nihss_7d": ["NIHSS 7d"], "onset_date": ["Date of stroke onset"], "onset_time": ["Time of stroke onset"],
    "onset_to_ivt": ["Stroke onset to IVT (min)"], "onset_to_puncture": ["Stroke onset to puncture (min)"],
    "onset_to_recan": ["Stroke onset to recanalization (min)"], "procedure_min": ["Procedure duration"],
    "side": ["Hemisphere"], "procedure": ["Intervention procedure"], "device": ["Extraction system"],
    "stenting": ["Intervention procedure", "Stent combo"], "ia_lysis": ["Intervention procedure", "IAT"],
    "agent": ["Thrombolytics"], "occlusion": ["Occlusion site"], "heart": ["Heart condition"], "etiology": ["Etiology"],
    "antithrombotic": ["Anti-thrombotics before"], "anesthesia": ["Anesthesia"], "tici": ["TICI"], "sex": ["Sex"],
}
ISSUE_CLEAN = {
    "sbp/dbp": ["sbp", "dbp"], "onset_date": ["onset_date", "year"], "onset_time": ["onset_hour"],
    "procedure": ["technique"], "device": ["technique"], "side": ["side", "side_left"], "sex": ["sex_male"],
    "heart": ["heart", "heart3"], "occlusion": ["occlusion", "occlusion_detail"],
}
# raw column -> the issue column whose mark it shows (reverse of ISSUE_RAW, for corrections)
RAW_TO_CLEAN = {
    "Age": ["age"], "ASPECTS": ["aspects"], "BMI": ["bmi"], "Cholesterol": ["cholesterol"], "Diastolic BP": ["dbp"],
    "Systolic BP": ["sbp"], "Glycemia": ["glucose"], "IVT": ["ivt"], "90-day mRS": ["mrs90", "good_outcome", "death90"],
    "mRS before": ["mrs_pre"], "Admission NIHSS": ["nihss_adm"], "NIHSS 24h": ["nihss_24h"], "NIHSS 7d": ["nihss_7d"],
    "Date of stroke onset": ["onset_date", "year"], "Time of stroke onset": ["onset_hour"],
    "Stroke onset to IVT (min)": ["onset_to_ivt"], "Stroke onset to puncture (min)": ["onset_to_puncture"],
    "Stroke onset to recanalization (min)": ["onset_to_recan"], "Procedure duration": ["procedure_min"],
    "Hemisphere": ["side", "side_left"], "Intervention procedure": ["technique", "stenting", "angioplasty", "ia_lysis"],
    "Extraction system": ["technique"], "Stent combo": ["stenting"], "IAT": ["ia_lysis"], "Thrombolytics": ["agent"],
    "Occlusion site": ["occlusion", "occlusion_detail"], "Heart condition": ["heart3"], "Etiology": ["etiology"],
    "Anti-thrombotics before": ["antithrombotic"], "Anesthesia": ["anesthesia"], "TICI": ["tici"], "Sex": ["sex_male"],
    "IVT in different hospital": ["ivt_external"], "Transfer from different hospital": ["transfer"],
    "Hypertension": ["htn"], "DM": ["dm"], "Hyperlipidemia": ["hld"], "Smoking": ["smoking"], "Alcohol abusus": ["alcohol"],
    "Arrhythmia": ["arrhythmia"], "TIA before": ["tia_prior"], "CMP before": ["stroke_prior"], "Statins before": ["statin"],
    "SICH": ["sich"], "non-SICH": ["ich_other"], "Cerebral edema": ["edema"], "Decompression surgery": ["decompression"],
}

MARK_ORDER = ["missing", "flag", "repaired", "kept", "corrected", "confirmed"]
MARK_LABELS = {
    "missing": ("Set to missing by the cleaning", "Nastaveno jako chybějící"),
    "flag": ("Flagged for review (value kept)", "Ke kontrole (hodnota ponechána)"),
    "repaired": ("Repaired automatically", "Opraveno automaticky"),
    "kept": ("Kept in the registry, not analysed", "Ponecháno v registru, neanalyzováno"),
    "corrected": ("Corrected", "Opraveno"),
    "confirmed": ("Checked, correct", "Zkontrolováno, správně"),
}
VERSION_LABELS = {
    "raw": ("Registry export (as recorded)", "Export registru (jak je zapsán)"),
    "clean": ("Cleaned registry (all records)", "Vyčištěný registr (všechny záznamy)"),
    "cohort": ("Analysed patients", "Analyzovaní pacienti"),
}


def L(pair, lang: str) -> str:
    return pair[0] if lang == "en" or len(pair) < 2 else pair[1]


def column_label(col: str, lang: str, version: str = "clean") -> str:
    if version == "raw" and col in REGISTRY_LABELS:
        return L(REGISTRY_LABELS[col], lang)
    f = next((x for x in features() if x.key == col), None)
    if f is not None:
        return f.spec["label"][lang] if lang in f.spec["label"] else f.spec["label"]["en"]
    if col in EXTRA_LABELS:
        return L(EXTRA_LABELS[col], lang)
    if col in REGISTRY_LABELS:
        return L(REGISTRY_LABELS[col], lang)
    return col


def version_label(v: str, lang: str) -> str:
    if v in VERSION_LABELS:
        return L(VERSION_LABELS[v], lang)
    st = stage_spec(v)
    title = st["title"][lang] if lang in st["title"] else st["title"]["en"]
    return (f"Model inputs: {title}" if lang == "en" else f"Vstupy modelu: {title}")


# ----------------------------------------------------------------------------------------- the log
def issue_kind(action: str) -> str:
    if action.startswith("flag only"):
        return "flag"
    if action == "year typo repaired":
        return "repaired"
    if action.startswith("paediatric"):
        return "kept"
    return "missing"


def issue_text(action: str, lang: str) -> str:
    """The rule in words (Czech from the public check list where it has one)."""
    if lang == "en":
        return action.replace("flag only: ", "")
    for key, _, _, cs in CHECK_GROUPS:
        if key in action:
            return cs
    return action


_POSSIBLY = re.compile(r"possibly\s+(-?[0-9]+(?:[.,][0-9]+)?)")
NOTE_CS = [(r"possibly (\S+) \(decimal shift\)", r"možná \1 (posunutá desetinná čárka)"),
           (r"possibly (\S+) \(missing digit\)", r"možná \1 (chybějící číslice)"), (r"possibly", "možná"),
           (r"fits between neighbouring patient IDs", "zapadá mezi sousední ID pacientů"),
           (r"^occlusion ", "okluze "), (r"puncture at", "punkce v"), (r"^puncture ", "punkce "), (r"procedure (\d)", r"výkon \1"),
           (r"extraction system:", "extrakční systém:"), (r"pre-stroke mRS", "mRS před CMP"), (r"90-day mRS", "mRS za 90 dní"),
           (r"IVT in different hospital:", "IVT v jiné nemocnici:"), (r"Stent combo:", "Stent:"), (r"\(empty\)", "(prázdné)")]
EXCLUDED = {"age under 18": ("age under 18", "věk pod 18 let"), "90-day mRS missing": ("90-day mRS missing", "chybí mRS za 90 dní")}


def note_text(note, lang: str) -> str:
    s = "" if note is None or (isinstance(note, float) and np.isnan(note)) else str(note)
    if lang == "en" or not s:
        return s
    for a, b in NOTE_CS:
        s = re.sub(a, b, s)
    return s


def excluded_text(reason: str, lang: str) -> str:
    return L(EXCLUDED[reason], lang) if reason in EXCLUDED else reason


def suggestion(row) -> str | None:
    """Value the cleaning suggests for the source, if any: 'possibly 25.78 (decimal shift)' -> 25.78;
    a repaired onset year -> the repaired date."""
    note = str(row.get("note") or "")
    m = _POSSIBLY.search(note)
    if m:
        return m.group(1).replace(",", ".")
    if row.get("action") == "year typo repaired" and row.get("new_value") not in (None, ""):
        return norm_value(row.get("new_value"))
    return None


# ----------------------------------------------------------------------------------------- versions
def build(raw: pd.DataFrame, corrections: pd.DataFrame | None = None) -> dict:
    """Every version of the registry plus the cleaning log, with corrections applied."""
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    corr = load_corrections() if corrections is None else corrections
    fixed, report = apply_corrections(raw, corr)
    clean, issues = clean_registry(fixed)
    base, steps = flow(clean)
    issues = issues.copy()
    issues["kind"] = issues["action"].map(issue_kind)
    issues["suggested"] = [suggestion(r) for r in issues.to_dict("records")]
    issues["raw_columns"] = issues["column"].map(lambda c: ISSUE_RAW.get(c, []))
    # review status from the corrections file
    active = corr[corr["status"] == "active"] if len(corr) else corr
    status = {}
    for _, r in active.iterrows():
        status[(str(r["patient_id"]), r["column"])] = r
    def _status(row):
        for col in row["raw_columns"]:
            r = status.get((str(int(row["patient_id"])), col))
            if r is not None:
                return "confirmed" if r["kind"] == "confirmed" else "corrected"
        return "open"
    issues["status"] = [_status(r) for r in issues.to_dict("records")] if len(issues) else []
    clean = clean.copy()
    excl = pd.Series("", index=clean.index, dtype=object)
    excl[clean["age"] < settings()["cohort"]["min_age"]] = "age under 18"
    excl[(excl == "") & clean["mrs90"].isna()] = "90-day mRS missing"
    clean["excluded"] = excl
    t0 = settings()["split"]["temporal_test_from_year"]
    cohort = base.copy()
    cohort["split"] = np.where(cohort["year"] >= t0, f"test {t0}+", f"development <{t0}")
    member = {s: set(stage_cohort(base, s)["patient_id"]) for s in stage_ids()}
    cohort["stage_cohorts"] = [", ".join(s for s in stage_ids() if pid in member[s]) for pid in cohort["patient_id"]]
    out = {"raw_original": raw, "raw": fixed, "report": report, "clean": clean, "issues": issues, "cohort": cohort,
           "steps": steps}
    for s in stage_ids():
        d = stage_cohort(base, s)
        cols = {}
        for f in stage_features(s):
            cols[f.key] = derived(d, f.key) if f.type == "derived" else d[f.key]
        m = pd.DataFrame(cols, index=d.index)
        m.insert(0, "patient_id", d["patient_id"])
        m.insert(1, "split", np.where(d["year"] >= t0, f"test {t0}+", f"development <{t0}"))
        m["good_outcome"], m["death90"] = d["good_outcome"], d["death90"]
        out[s] = m
    return out


def versions() -> list[str]:
    return ["raw", "clean", "cohort"] + stage_ids()


CLEAN_VIEW = ["patient_id", "excluded", "year", "onset_date", "onset_hour", "age", "sex_male", "mrs_pre", "nihss_adm",
              "aspects", "occlusion_raw", "occlusion_detail", "occlusion", "side", "ivt", "ivt_external", "onset_to_ivt",
              "transfer", "onset_to_puncture", "glucose", "sbp", "dbp", "bmi", "cholesterol", "htn", "dm", "hld",
              "arrhythmia", "stroke_prior", "tia_prior", "smoking", "alcohol", "statin", "heart", "heart3",
              "antithrombotic", "procedure_raw", "device_raw", "technique", "stenting", "angioplasty", "ia_lysis",
              "procedure_min", "anesthesia", "tici_label", "reperfusion_2b3", "onset_to_recan", "time_conflict",
              "nihss_24h", "nihss_24h_status", "sich", "ich_other", "nihss_7d", "nihss_7d_status", "edema",
              "decompression", "etiology", "agent", "mrs90", "good_outcome", "death90"]


def table(data: dict, version: str) -> pd.DataFrame:
    """The table shown for a version (patient ID first)."""
    if version == "raw":
        df = data["raw"]
        cols = [c for c in RAW_COLUMNS if c in df.columns]
        return df[cols].reset_index(drop=True)
    if version == "clean":
        df = data["clean"]
        return df[[c for c in CLEAN_VIEW if c in df.columns]].reset_index(drop=True)
    if version == "cohort":
        df = data["cohort"]
        cols = ["patient_id", "split", "stage_cohorts"] + [c for c in CLEAN_VIEW if c in df.columns and c not in ("patient_id", "excluded")]
        return df[cols].reset_index(drop=True)
    return data[version].reset_index(drop=True)


def marks(data: dict, version: str, tab: pd.DataFrame, lang: str = "en") -> tuple[pd.DataFrame, pd.Series]:
    """Per-cell mark (see MARK_ORDER; "" = none) for a version's table, and a per-row summary."""
    mk = pd.DataFrame("", index=tab.index, columns=tab.columns)
    summary = pd.Series("", index=tab.index, dtype=object)
    pid_col = "Patient ID" if version == "raw" else "patient_id"
    pos = {int(p): i for i, p in zip(tab.index, pd.to_numeric(tab[pid_col], errors="coerce").fillna(-1).astype(int))}
    rank = {m: k for k, m in enumerate(MARK_ORDER)}

    def put(i, col, m):
        if col in mk.columns and (mk.at[i, col] == "" or rank[m] < rank.get(mk.at[i, col], 99)):
            mk.at[i, col] = m

    notes = {}
    for r in data["issues"].to_dict("records"):
        i = pos.get(int(r["patient_id"]))
        if i is None:
            continue
        kind = r["status"] if r["status"] in ("corrected", "confirmed") else r["kind"]
        cols = r["raw_columns"] if version == "raw" else ISSUE_CLEAN.get(r["column"], [r["column"]])
        for col in cols:
            put(i, col, kind)
        notes.setdefault(i, []).append((r["column"], r["action"], r.get("suggested")))
    rep = data["report"]
    if len(rep):
        for r in rep.to_dict("records"):
            if r["result"] != "applied" and r["result"] != "confirmed":
                continue
            try:
                i = pos.get(int(float(r["patient_id"])))
            except (TypeError, ValueError):
                i = None
            if i is None:
                continue
            m = "corrected" if r["result"] == "applied" else "confirmed"
            cols = [r["column"]] if version == "raw" else RAW_TO_CLEAN.get(r["column"], [])
            for col in cols:
                put(i, col, m)
    sug = "suggested" if lang == "en" else "návrh"
    for i, items in notes.items():
        summary.at[i] = "; ".join(f"{column_label(c, lang)}: {issue_text(a, lang)}" + (f" ({sug} {s})" if s else "")
                                  for c, a, s in items)
    return mk, summary


def display_value(v) -> str:
    return norm_value(v)
