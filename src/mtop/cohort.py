"""Inclusion criteria and stage-specific analysis cohorts."""
from __future__ import annotations

import pandas as pd

from .config import settings, stage_ids, stage_spec


def flow(clean: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the inclusion criteria; return the analysis cohort and a flow table (counts only).

    Each step records how many records remain and how many were removed and why. Records that stay
    but deserve a mention (diagnostic angiography only, an impossible time set to missing, unknown
    age) are listed as `kept` notes of the last step.
    """
    s = settings()["cohort"]
    steps = [{"step": "registry", "n": len(clean)}]
    d = clean
    young = d["age"] < s["min_age"]          # a missing age is kept (imputed like any predictor)
    d = d[~young]
    steps.append({"step": "adults", "n": len(d), "removed": int(young.sum()), "reason": "age_under_18"})
    if s.get("exclude_diagnostic_only", False):
        m = d["diagnostic_only"] == 1
        d = d[~m]
        steps.append({"step": "treated", "n": len(d), "removed": int(m.sum()), "reason": "diagnostic_only"})
    if s.get("exclude_impossible_times", False) and "time_conflict" in d:
        m = d["time_conflict"] == 1
        d = d[~m]
        steps.append({"step": "times_valid", "n": len(d), "removed": int(m.sum()), "reason": "impossible_time"})
    if s.get("require_outcome", True):
        m = d["mrs90"].isna()
        d = d[~m]
        steps.append({"step": "outcome_known", "n": len(d), "removed": int(m.sum()), "reason": "mrs90_missing"})
    kept = {"diagnostic_only": int((d["diagnostic_only"] == 1).sum())}
    if "time_conflict" in d:
        kept["impossible_time"] = int((d["time_conflict"] == 1).sum())
    kept["age_unknown"] = int(d["age"].isna().sum())
    steps[-1]["kept"] = kept
    return d.copy(), steps


def stage_cohort(base: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Patients eligible for a stage (settings: stages[].requires).

    Stage 2 (end of the procedure) requires a recorded final mTICI, stage 3 (24 h) also an assessed
    24-h NIHSS (alive, not coded 43/99) and stage 4 an assessed day-7 NIHSS (a missing 24-h NIHSS is
    allowed there). Deaths before a stage therefore leave its cohort, which is the intended use: the
    24-hour and day-7 estimates are for patients alive and assessable at that point.
    """
    m = pd.Series(True, index=base.index)
    for req in stage_spec(stage)["requires"]:
        m &= base[req].notna()
    return base[m]


def split_temporal(df: pd.DataFrame, test_from: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Development (onset before `test_from`) and temporal validation (from `test_from`) patients.
    Default: the primary split from settings (development 2005-2022, validation 2023-2025)."""
    y0 = settings()["split"]["temporal_test_from_year"] if test_from is None else int(test_from)
    return df[df["year"] < y0], df[df["year"] >= y0]


def splits() -> dict[str, int]:
    """The temporal splits that are analysed: primary and sensitivity (first test year)."""
    s = settings()["split"]
    out = {"primary": int(s["temporal_test_from_year"])}
    if s.get("sensitivity_test_from_year"):
        out["sensitivity"] = int(s["sensitivity_test_from_year"])
    return out
