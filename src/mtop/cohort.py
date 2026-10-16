"""Inclusion criteria and stage-specific analysis cohorts."""
from __future__ import annotations

import pandas as pd

from .config import settings, stage_ids, stage_spec


def flow(clean: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Apply inclusion criteria; return the base cohort and a flow table (counts only)."""
    s = settings()["cohort"]
    steps = [{"step": "registry", "n": len(clean)}]
    d = clean
    d = d[d["age"] >= s["min_age"]]
    steps.append({"step": f"age >= {s['min_age']}", "n": len(d)})
    if s.get("exclude_diagnostic_only", True):
        d = d[d["diagnostic_only"] != 1]
        steps.append({"step": "endovascular treatment performed", "n": len(d)})
    if s.get("require_outcome", True):
        d = d[d["mrs90"].notna()]
        steps.append({"step": "90-day mRS known", "n": len(d)})
    return d.copy(), steps


def stage_cohort(base: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Patients eligible for a stage: cumulative requirements of this and earlier stages.

    Stage 3 (24 h) requires an assessed 24-h NIHSS (alive, not coded 43/99), stage 4 an
    assessed day-7 NIHSS; earlier deaths therefore leave the later cohorts, which is the
    intended use: the 24-h and day-7 predictions are conditional on reaching that point.
    """
    m = pd.Series(True, index=base.index)
    for sid in stage_ids():
        for req in stage_spec(sid)["requires"]:
            m &= base[req].notna()
        if sid == stage:
            break
    return base[m]


def split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y0 = settings()["split"]["temporal_test_from_year"]
    return df[df["year"] < y0], df[df["year"] >= y0]
