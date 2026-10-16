"""Clean and harmonise the FNO mechanical-thrombectomy registry.

Input: the master Excel sheet (Czech coding, one row per patient).
Output: a tidy DataFrame with English variable names and numeric/categorical
codes, plus an issue log that lists every value that was changed or removed.

Design rules
------------
* Nothing is silently dropped: every correction is logged with the patient ID.
* Implausible values become missing (NaN) rather than being guessed. Where a
  likely correction exists (e.g. BMI 2448 -> 24.48) it is *suggested* in the
  log so it can be fixed in the source file, but it is not applied.
* The only automatic repair is an obvious one-year typo in the onset date,
  detected from the date sequence of neighbouring patient IDs.
* NIHSS 43 is the registry's code for "died"; 99 means "not assessable".
  Both are converted to NaN in the score and kept in a separate status field,
  because using them as scores would leak the outcome into the model.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

YES_NO = {"ANO": 1.0, "NE": 0.0}

RAW_COLUMNS = [
    "Patient ID", "Date of stroke onset", "Time of stroke onset", "Age", "Sex", "ASPECTS",
    "Occlusion site", "Hemisphere", "Intervention procedure", "Stent combo", "Extraction system",
    "Thrombolytics", "Procedure duration", "TICI", "IVT", "IVT in different hospital",
    "Transfer from different hospital", "IAT", "Stroke onset to IVT (min)",
    "Stroke onset to puncture (min)", "Stroke onset to recanalization (min)", "Admission NIHSS",
    "NIHSS 24h", "NIHSS 7d", "SICH", "non-SICH", "mRS before", "90-day mRS",
    "Anti-thrombotics before", "Statins before", "Hypertension", "DM", "Hyperlipidemia", "Smoking",
    "Alcohol abusus", "BMI", "Arrhythmia", "TIA before", "CMP before", "Heart condition",
    "Systolic BP", "Diastolic BP", "Glycemia", "Cholesterol", "Etiology", "Cerebral edema",
    "Decompression surgery", "Anesthesia",
]

OCCLUSION = {
    "M1-ACM": "M1",
    "M2-ACM": "distal", "M3-ACM": "distal", "ACA": "distal",
    "ACI-Intracranial": "ICA", "ACI+ACM": "ICA",
    "ACI-odstup": "cervical_ICA", "ACI-odstup +T typ": "cervical_ICA",
    "ACC": "cervical_ICA", "ACC+ACI+ACM": "cervical_ICA",
    "AB": "basilar", "AB+AV": "basilar",
    "AV": "other_posterior", "ACP": "other_posterior",
    "Rekanalizace": None,  # already recanalised at angiography: no occlusion to classify
}

SIDE = {"levá hemisféra": "left", "pravá hemisféra": "right", "zadní povodí": "posterior"}

# procedure label -> (mechanical thrombectomy, stenting, angioplasty, intra-arterial lysis, diagnostic only)
PROCEDURE = {
    "ADAPT": (1, 0, 0, 0, 0),
    "Solumbra": (1, 0, 0, 0, 0),
    "MT-stent retriever": (1, 0, 0, 0, 0),
    "kombinovaný výkon": (1, 0, 0, 0, 0),
    "MT+stent": (1, 1, 0, 0, 0),
    "mechanická extrakce+cPTA": (1, 0, 1, 0, 0),
    "cPTA": (0, 0, 1, 0, 0),
    "PTA": (0, 0, 1, 0, 0),
    "Stent": (0, 1, 0, 0, 0),
    "Stent+PTA": (0, 1, 1, 0, 0),
    "cPTA+stent": (0, 1, 1, 0, 0),
    "EKOS+stent": (0, 1, 0, 1, 0),
    "LIAT": (0, 0, 0, 1, 0),
    "EKOS+LIAT": (0, 0, 0, 1, 0),
    "cPTA+LIAT": (0, 0, 1, 1, 0),
    "PAG": (0, 0, 0, 0, 1),
}
# Explicit first-line technique. "kombinovaný výkon" (combined procedure) is mostly MT plus
# carotid stenting, so its technique is taken from the extraction device instead.
PROCEDURE_TECHNIQUE = {
    "ADAPT": "aspiration", "MT-stent retriever": "stent_retriever", "Solumbra": "combined",
}

DEVICE = {
    "PENUMBRA": "aspiration",
    "Catch": "stent_retriever", "Preset": "stent_retriever", "Solitaire": "stent_retriever",
    "TIGERTRIEVER": "stent_retriever", "Embotrap": "stent_retriever", "Bonnet": "stent_retriever",
    "Merci": "stent_retriever",
    "PENUMBRA + Catch": "combined", "PENUMBRA + Preset": "combined",
    "PENUMBRA + TIGERTRIEVER": "combined",
    "žádný": "none",
}

AGENT = {"Actilyse": "alteplase", "Tenektepláza": "tenecteplase", "Integrilin": "eptifibatide",
         "žádné": "none", "bez výkonu": "none"}

ANTITHROMBOTIC = {
    "žádná": "none",
    "AT-ASA": "antiplatelet", "AC-acylpyrin": "antiplatelet", "AP-apotic": "antiplatelet",
    "AT-Clopidogrel": "antiplatelet", "AT-dual ASA+Clop": "antiplatelet",
    "VKA-Warfarin": "anticoagulant", "NOAC-rivaroxaban": "anticoagulant",
    "NOAC-apixaban": "anticoagulant", "NOAC-dabigatran": "anticoagulant", "LMWH": "anticoagulant",
}

HEART = {
    "žádné": "none", "ICHS": "coronary", "Ao. chlopeň": "valve", "Mi. chlopeň": "valve",
    "Náhrada aorty": "valve", "kardiomyopatie": "cardiomyopathy", "KS. Kardio.": "other",
    "PFO": "pfo", "Trombus v Levém oušku": "laa_thrombus",
}

ETIOLOGY = {
    "kardioembolická": "cardioembolic", "aterotrombotická": "atherothrombotic",
    "kryptogenní": "cryptogenic", "disekce ACI": "dissection", "disekce AV": "dissection",
    "disekce ACM": "dissection", "disekce aorty": "dissection", "postradiační": "other",
    "tromboza stentu": "other",
}

ANESTHESIA = {"Celková anestezie": "general", "Analgosedace": "sedation", "bez ARO": "local"}

TICI = {"0": 0.0, "1": 1.0, "2a": 2.0, "2": 2.5, "2b": 3.0, "3": 4.0}
TICI_LABEL = {0.0: "0", 1.0: "1", 2.0: "2a", 2.5: "2 (not subgraded)", 3.0: "2b", 4.0: "3"}

# plausible ranges (inclusive) — outside -> NaN and logged
RANGES = {
    "age": (0, 110),
    "aspects": (0, 10),
    "nihss_adm": (0, 42),
    "mrs_pre": (0, 5),
    "mrs90": (0, 6),
    "bmi": (12, 70),
    "sbp": (60, 300),
    "dbp": (30, 200),
    "glucose": (1.5, 40),
    "cholesterol": (1.0, 20),
    "procedure_min": (1, 600),
    "onset_to_ivt": (1, 2880),
    "onset_to_puncture": (1, 5000),
    "onset_to_recan": (1, 5000),
}

NIHSS_DIED = 43
NIHSS_NOT_ASSESSABLE = 99


@dataclass
class IssueLog:
    rows: list = field(default_factory=list)

    def add(self, pid, column, raw, action, new=None, note=""):
        self.rows.append({"patient_id": pid, "column": column, "raw_value": raw,
                          "action": action, "new_value": new, "note": note})

    def frame(self) -> pd.DataFrame:
        cols = ["patient_id", "column", "raw_value", "action", "new_value", "note"]
        return pd.DataFrame(self.rows, columns=cols)


def _null(x) -> bool:
    """True for None / NaN / pd.NA (pandas may turn None into NaN inside .map)."""
    try:
        return x is None or bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _s(x):
    """Normalise a raw cell to a stripped string or None."""
    if _null(x):
        return None
    s = str(x).strip()
    return s if s else None


def _map(raw: pd.Series, mapping: dict, name: str, ids: pd.Series, log: IssueLog) -> pd.Series:
    out = []
    for pid, v in zip(ids, raw):
        s = _s(v)
        if s is None:
            out.append(None)
            continue
        if s in mapping:
            out.append(mapping[s])
        else:
            log.add(pid, name, s, "unknown category -> missing")
            out.append(None)
    return pd.Series(out, index=raw.index, dtype="object")


def _yes_no(raw: pd.Series, name: str, ids: pd.Series, log: IssueLog) -> pd.Series:
    out = []
    for pid, v in zip(ids, raw):
        s = _s(v)
        if s is None:
            out.append(np.nan)
        elif s.upper() in YES_NO:
            out.append(YES_NO[s.upper()])
        else:
            log.add(pid, name, s, "not ANO/NE -> missing")
            out.append(np.nan)
    return pd.Series(out, index=raw.index, dtype="float64")


def _num(raw: pd.Series) -> pd.Series:
    return pd.to_numeric(raw, errors="coerce").astype("float64")


def _range(series: pd.Series, name: str, ids: pd.Series, log: IssueLog, suggest=None) -> pd.Series:
    lo, hi = RANGES[name]
    s = series.copy()
    bad = s.notna() & ((s < lo) | (s > hi))
    for idx in s.index[bad]:
        raw = s.at[idx]
        note = suggest(raw) if suggest else ""
        log.add(ids.at[idx], name, raw, f"outside {lo}-{hi} -> missing", None, note)
    s[bad] = np.nan
    return s


def _suggest_decimal(v, lo, hi):
    for f in (10, 100):
        if lo <= v / f <= hi:
            return f"possibly {v / f:g} (decimal shift)"
    for f in (10,):
        if lo <= v * f <= hi:
            return f"possibly {v * f:g} (missing digit)"
    return "check source record"


def parse_onset_time(x):
    """Return hour as float (e.g. 6.5 for 06:30) or NaN; handles the registry's mixed formats."""
    if isinstance(x, dt.time):
        return x.hour + x.minute / 60
    if isinstance(x, dt.datetime):
        return x.hour + x.minute / 60
    s = _s(x)
    if s is None:
        return np.nan
    s = s.rstrip(".").replace(",", ":").replace(".", ":")
    m = re.fullmatch(r"(\d{1,2}):(\d{2})\d*", s)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
    elif re.fullmatch(r"\d{4}", s):
        h, mi = int(s[:2]), int(s[2:])
    elif re.fullmatch(r"\d{1,2}", s):
        h, mi = int(s), 0
    else:
        return np.nan
    if 0 <= h <= 23 and 0 <= mi <= 59:
        return h + mi / 60
    return np.nan


def repair_onset_years(ids: pd.Series, dates: pd.Series, log: IssueLog) -> pd.Series:
    """Fix isolated +/-1 year typos using the date sequence of neighbouring patient IDs.

    A date is changed only if (a) its neighbours by patient ID are in order,
    (b) the date itself falls outside them, and (c) shifting the year by exactly one
    places it between them.
    """
    order = ids.sort_values().index
    d = dates.loc[order]
    fixed = dates.copy()
    vals = list(d.items())
    for k in range(1, len(vals) - 1):
        idx, cur = vals[k]
        _, prev = vals[k - 1]
        _, nxt = vals[k + 1]
        if pd.isna(cur) or pd.isna(prev) or pd.isna(nxt) or prev > nxt:
            continue
        if prev <= cur <= nxt:
            continue
        for shift in (-1, 1):
            try:
                cand = cur.replace(year=cur.year + shift)
            except ValueError:
                continue
            if prev <= cand <= nxt:
                log.add(ids.at[idx], "onset_date", cur.date().isoformat(), "year typo repaired",
                        cand.date().isoformat(), "fits between neighbouring patient IDs")
                fixed.at[idx] = cand
                break
    return fixed


def _nihss_followup(raw: pd.Series, name: str, ids: pd.Series, log: IssueLog):
    score = _num(raw)
    status = pd.Series("assessed", index=raw.index, dtype="object")
    status[score.isna()] = "missing"
    died = score == NIHSS_DIED
    na = score == NIHSS_NOT_ASSESSABLE
    other = score.notna() & ~died & ~na & ((score < 0) | (score > 42))
    status[died] = "died"
    status[na | other] = "not_assessable"
    for idx in score.index[other]:
        log.add(ids.at[idx], name, score.at[idx], "outside 0-42 -> missing")
    score = score.where(~(died | na | other))
    return score, status


def clean_registry(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean, issues)."""
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    missing_cols = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    log = IssueLog()
    ids = raw["Patient ID"].astype("int64")
    if not ids.is_unique:
        raise ValueError("Patient ID is not unique")

    c = pd.DataFrame(index=raw.index)
    c["patient_id"] = ids

    # --- dates and times -------------------------------------------------------------------
    dates = pd.to_datetime(raw["Date of stroke onset"], errors="coerce")
    dates = repair_onset_years(ids, dates, log)
    c["onset_date"] = dates
    c["year"] = dates.dt.year.astype("float64")
    onset_hour = raw["Time of stroke onset"].map(parse_onset_time)
    for pid, rv, h in zip(ids, raw["Time of stroke onset"], onset_hour):
        if _s(rv) is not None and np.isnan(h):
            log.add(pid, "onset_time", _s(rv), "unparseable/unknown time -> missing")
    c["onset_hour"] = onset_hour.astype("float64")

    # --- demographics, baseline ------------------------------------------------------------
    c["age"] = _range(_num(raw["Age"]), "age", ids, log)
    for idx in c.index[c["age"] < 18]:
        log.add(ids.at[idx], "age", c.at[idx, "age"], "paediatric patient (kept; excluded from models)")
    sex = _map(raw["Sex"], {"Muž": "male", "Žena": "female"}, "sex", ids, log)
    c["sex_male"] = sex.map({"male": 1.0, "female": 0.0}).astype("float64")
    c["mrs_pre"] = _range(_num(raw["mRS before"]), "mrs_pre", ids, log)
    c["mrs90"] = _range(_num(raw["90-day mRS"]), "mrs90", ids, log)

    # --- admission status and imaging ------------------------------------------------------
    c["nihss_adm"] = _range(_num(raw["Admission NIHSS"]), "nihss_adm", ids, log)
    c["aspects"] = _range(_num(raw["ASPECTS"]), "aspects", ids, log)
    c["occlusion"] = _map(raw["Occlusion site"], OCCLUSION, "occlusion", ids, log)
    c["occlusion_raw"] = raw["Occlusion site"].map(_s)
    c["recanalised_at_angio"] = (raw["Occlusion site"].map(_s) == "Rekanalizace").astype("float64")
    c["side"] = _map(raw["Hemisphere"], SIDE, "side", ids, log)
    c["side_left"] = c["side"].map({"left": 1.0, "right": 0.0, "posterior": 0.0}).astype("float64")
    c["posterior"] = c["side"].map({"left": 0.0, "right": 0.0, "posterior": 1.0}).astype("float64")

    # --- history / comorbidities -------------------------------------------------------------
    yn = {
        "htn": "Hypertension", "dm": "DM", "hld": "Hyperlipidemia", "smoking": "Smoking",
        "alcohol": "Alcohol abusus", "arrhythmia": "Arrhythmia", "tia_prior": "TIA before",
        "stroke_prior": "CMP before", "statin": "Statins before", "ivt": "IVT", "iat": "IAT",
        "sich": "SICH", "ich_other": "non-SICH", "edema": "Cerebral edema",
        "decompression": "Decompression surgery",
    }
    for new, old in yn.items():
        c[new] = _yes_no(raw[old], new, ids, log)

    at = _map(raw["Anti-thrombotics before"], ANTITHROMBOTIC, "antithrombotic", ids, log)
    c["antithrombotic"] = at
    c["anticoag"] = at.map({"anticoagulant": 1.0, "antiplatelet": 0.0, "none": 0.0}).astype("float64")
    c["antiplatelet"] = at.map({"anticoagulant": 0.0, "antiplatelet": 1.0, "none": 0.0}).astype("float64")
    heart = _map(raw["Heart condition"], HEART, "heart", ids, log)
    c["heart"] = heart
    c["coronary"] = heart.map(lambda v: np.nan if _null(v) else float(v == "coronary")).astype("float64")
    c["heart_structural"] = heart.map(
        lambda v: np.nan if _null(v) else float(v in ("valve", "cardiomyopathy", "laa_thrombus"))
    ).astype("float64")

    bmi = _num(raw["BMI"])
    c["bmi"] = _range(bmi, "bmi", ids, log, suggest=lambda v: _suggest_decimal(v, 12, 70))
    c["sbp"] = _range(_num(raw["Systolic BP"]), "sbp", ids, log, suggest=lambda v: _suggest_decimal(v, 60, 300))
    c["dbp"] = _range(_num(raw["Diastolic BP"]), "dbp", ids, log)
    bad_bp = c["sbp"].notna() & c["dbp"].notna() & (c["sbp"] <= c["dbp"])
    for idx in c.index[bad_bp]:
        log.add(ids.at[idx], "sbp/dbp", f"{c.at[idx, 'sbp']:g}/{c.at[idx, 'dbp']:g}",
                "systolic <= diastolic -> both missing")
    c.loc[bad_bp, ["sbp", "dbp"]] = np.nan
    c["glucose"] = _range(_num(raw["Glycemia"]), "glucose", ids, log,
                          suggest=lambda v: _suggest_decimal(v, 1.5, 40))
    c["cholesterol"] = _range(_num(raw["Cholesterol"]), "cholesterol", ids, log,
                              suggest=lambda v: _suggest_decimal(v, 1.0, 20))

    # --- pathway / treatment -------------------------------------------------------------------
    tr = raw["Transfer from different hospital"].map(_s)
    c["transfer"] = tr.map(lambda v: np.nan if _null(v) else float(v not in ("FNO", "žádné"))).astype("float64")
    c["referring_hospital"] = tr.where(~tr.isin(["FNO", "žádné"]))
    c["agent"] = _map(raw["Thrombolytics"], AGENT, "agent", ids, log)
    c["tenecteplase"] = (c["agent"] == "tenecteplase").astype("float64")

    proc = raw["Intervention procedure"].map(_s)
    c["procedure_raw"] = proc
    flags = proc.map(lambda v: None if _null(v) else PROCEDURE.get(v))
    for pid, v in zip(ids, proc):
        if not _null(v) and v not in PROCEDURE:
            log.add(pid, "procedure", v, "unknown procedure -> missing")
    for j, name in enumerate(["mt_label", "stent_proc", "angioplasty", "ia_lysis_proc", "diagnostic_only"]):
        c[name] = flags.map(lambda f: np.nan if _null(f) else float(f[j])).astype("float64")

    # Technique: explicit procedure labels win (Solumbra lists only the stent retriever as
    # device); otherwise use the extraction-device class.
    device = _map(raw["Extraction system"], DEVICE, "device", ids, log)
    c["device_raw"] = raw["Extraction system"].map(_s)
    explicit = proc.map(lambda v: None if _null(v) else PROCEDURE_TECHNIQUE.get(v))
    tech = explicit.where(explicit.notna(), device)
    tech = tech.where(~((c["mt_label"] == 1) & (tech == "none")), None)   # MT label but "no device": unknown
    tech[(c["mt_label"] == 0) & tech.isna()] = "none"
    c["technique"] = tech
    c["mt_performed"] = tech.map(
        lambda v: np.nan if _null(v) else float(v in ("aspiration", "stent_retriever", "combined"))
    ).astype("float64")
    c.loc[c["mt_performed"].isna() & (c["mt_label"] == 1), "mt_performed"] = 1.0

    stent_raw = raw["Stent combo"].map(_s)
    stent_device = stent_raw.map(lambda v: np.nan if _null(v) else float(v != "žádný")).astype("float64")
    c["stenting"] = pd.concat([stent_device, c["stent_proc"]], axis=1).max(axis=1, skipna=True)
    c.loc[stent_device.isna() & c["stent_proc"].isna(), "stenting"] = np.nan
    c["ia_lysis"] = pd.concat([c["iat"], c["ia_lysis_proc"]], axis=1).max(axis=1, skipna=True)
    c.loc[c["iat"].isna() & c["ia_lysis_proc"].isna(), "ia_lysis"] = np.nan

    c["anesthesia"] = _map(raw["Anesthesia"], ANESTHESIA, "anesthesia", ids, log)
    c["procedure_min"] = _range(_num(raw["Procedure duration"]), "procedure_min", ids, log)

    tici_s = raw["TICI"].map(lambda v: None if _s(v) is None else _s(v).lower())
    c["tici"] = _map(tici_s, TICI, "tici", ids, log).astype("float64")
    c["tici_label"] = c["tici"].map(TICI_LABEL)
    c["reperfusion_2b3"] = c["tici"].map(lambda t: np.nan if np.isnan(t) or t == 2.5 else float(t >= 3))

    # --- times -----------------------------------------------------------------------------------
    c["onset_to_ivt"] = _range(_num(raw["Stroke onset to IVT (min)"]), "onset_to_ivt", ids, log)
    c["onset_to_puncture"] = _range(_num(raw["Stroke onset to puncture (min)"]), "onset_to_puncture", ids, log)
    c["onset_to_recan"] = _range(_num(raw["Stroke onset to recanalization (min)"]), "onset_to_recan", ids, log)
    bad_t = c["onset_to_recan"].notna() & c["onset_to_puncture"].notna() & (c["onset_to_recan"] < c["onset_to_puncture"])
    for idx in c.index[bad_t]:
        log.add(ids.at[idx], "onset_to_recan", c.at[idx, "onset_to_recan"],
                "recanalisation before puncture -> missing", None,
                f"puncture at {c.at[idx, 'onset_to_puncture']:g} min")
    c.loc[bad_t, "onset_to_recan"] = np.nan
    late_ivt = c["onset_to_ivt"].notna() & c["onset_to_puncture"].notna() & (c["onset_to_ivt"] > c["onset_to_puncture"] + 30)
    for idx in c.index[late_ivt]:
        log.add(ids.at[idx], "onset_to_ivt", c.at[idx, "onset_to_ivt"], "flag only: IVT >30 min after puncture",
                c.at[idx, "onset_to_ivt"], f"puncture at {c.at[idx, 'onset_to_puncture']:g} min")
    for idx in c.index[c["onset_to_puncture"] > 1440]:
        log.add(ids.at[idx], "onset_to_puncture", c.at[idx, "onset_to_puncture"],
                "flag only: puncture > 24 h after onset", c.at[idx, "onset_to_puncture"])

    # --- follow-up ---------------------------------------------------------------------------------
    c["nihss_24h"], c["nihss_24h_status"] = _nihss_followup(raw["NIHSS 24h"], "nihss_24h", ids, log)
    c["nihss_7d"], c["nihss_7d_status"] = _nihss_followup(raw["NIHSS 7d"], "nihss_7d", ids, log)
    inconsistent = (c["nihss_7d_status"] == "died") & c["mrs90"].notna() & (c["mrs90"] != 6)
    for idx in c.index[inconsistent]:
        log.add(ids.at[idx], "nihss_7d", NIHSS_DIED, "flag only: coded as died by day 7 but 90-day mRS is not 6",
                None, f"90-day mRS {c.at[idx, 'mrs90']:g}")
    c["nihss_change_24h"] = c["nihss_24h"] - c["nihss_adm"]
    c["etiology"] = _map(raw["Etiology"], ETIOLOGY, "etiology", ids, log)

    # --- outcomes -------------------------------------------------------------------------------------
    c["good_outcome"] = (c["mrs90"] <= 2).astype("float64").where(c["mrs90"].notna())
    c["death90"] = (c["mrs90"] == 6).astype("float64").where(c["mrs90"].notna())

    issues = log.frame().sort_values(["column", "patient_id"]).reset_index(drop=True)
    return c.reset_index(drop=True), issues


def load_raw(path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=0)
