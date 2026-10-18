"""Cleaning rules that protect the models from silent data errors."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from mtop.clean import RAW_COLUMNS, clean_registry, parse_onset_time
from mtop.synth import make_synthetic


def _raw(n=5):
    df = make_synthetic(n=n, seed=3)
    return df.reset_index(drop=True)


def test_schema_roundtrip():
    clean, issues = clean_registry(_raw(40))
    assert len(clean) == 40
    assert {"good_outcome", "death90", "tici", "occlusion", "technique"} <= set(clean.columns)
    assert set(issues.columns) == {"patient_id", "column", "raw_value", "action", "new_value", "note"}


def test_nihss_codes_are_not_scores():
    raw = _raw(6)
    raw.loc[0, "NIHSS 7d"] = 43
    raw.loc[1, "NIHSS 7d"] = 99
    raw.loc[2, "NIHSS 24h"] = 43
    clean, _ = clean_registry(raw)
    assert np.isnan(clean.loc[0, "nihss_7d"]) and clean.loc[0, "nihss_7d_status"] == "died"
    assert np.isnan(clean.loc[1, "nihss_7d"]) and clean.loc[1, "nihss_7d_status"] == "not_assessable"
    assert clean.loc[2, "nihss_24h_status"] == "died"


def test_implausible_values_become_missing_and_are_logged():
    raw = _raw(6)
    raw.loc[0, "BMI"] = 2448
    raw.loc[1, "Systolic BP"] = 12
    raw.loc[2, "Cholesterol"] = 40.5
    clean, issues = clean_registry(raw)
    assert np.isnan(clean.loc[0, "bmi"]) and np.isnan(clean.loc[1, "sbp"]) and np.isnan(clean.loc[2, "cholesterol"])
    bmi_issue = issues[(issues["column"] == "bmi")]
    assert "24.48" in bmi_issue["note"].iloc[0]


def test_year_typo_repair():
    raw = _raw(5)
    raw["Date of stroke onset"] = [dt.datetime(2025, 6, 10), dt.datetime(2025, 6, 13), dt.datetime(2026, 6, 14),
                                   dt.datetime(2025, 6, 17), dt.datetime(2025, 6, 20)]
    clean, issues = clean_registry(raw)
    assert clean.loc[2, "year"] == 2025
    assert (issues["action"] == "year typo repaired").sum() == 1


def test_categorical_mappings():
    raw = _raw(6)
    raw.loc[0, ["Intervention procedure", "Extraction system"]] = ["Solumbra", "Catch"]
    raw.loc[1, ["Intervention procedure", "Extraction system"]] = ["kombinovaný výkon", "PENUMBRA"]
    raw.loc[2, ["Intervention procedure", "Extraction system"]] = ["PAG", "žádný"]
    raw.loc[3, "Transfer from different hospital"] = np.nan
    raw.loc[4, "TICI"] = "2"
    raw.loc[5, "Occlusion site"] = "ACI-odstup +T typ"
    clean, _ = clean_registry(raw)
    assert clean.loc[0, "technique"] == "solumbra"          # Solumbra (aspiration + stent retriever) whatever the device says
    assert clean.loc[1, "technique"] == "combined"          # kombinovaný výkon: two or more techniques, its own category
    assert clean.loc[2, "diagnostic_only"] == 1 and clean.loc[2, "technique"] == "diagnostic"   # PAG kept as its own category
    assert np.isnan(clean.loc[3, "transfer"])               # missing stays missing (not "transferred")
    assert clean.loc[4, "tici"] == 2.5 and np.isnan(clean.loc[4, "reperfusion_2b3"])
    assert clean.loc[5, "occlusion_detail"] == "tandem"    # cervical ICA + carotid T = tandem
    assert clean.loc[5, "occlusion"] == "tandem"


def test_rare_occlusion_codes_are_pooled_for_models():
    from mtop.clean import pool_occlusion

    detail = pd.Series(["M3", "ACA", "tandem_cca", "CCA", "basilar_vertebral", "vertebral", "PCA", "M1", None])
    counts = pd.Series({"M3": 5, "ACA": 3, "tandem_cca": 4, "CCA": 6, "basilar_vertebral": 27, "vertebral": 24, "PCA": 4, "M1": 600})
    allowed = {"ICA", "tandem", "cervical_ICA", "M1", "M2", "basilar", "other_posterior"}
    out = pool_occlusion(detail, counts, 30, allowed).tolist()
    assert out[:8] == ["M2", "M2", "tandem", "cervical_ICA", "basilar", "other_posterior", "other_posterior", "M1"]
    assert pd.isna(out[8])
    # a code that reaches the threshold but is not yet a model option keeps being pooled
    assert pool_occlusion(pd.Series(["M3"]), pd.Series({"M3": 40}), 30, allowed).tolist() == ["M2"]


def test_onset_time_formats():
    assert parse_onset_time("4:30") == 4.5
    assert parse_onset_time("06.30") == 6.5
    assert parse_onset_time("6,20") == 6 + 20 / 60
    assert parse_onset_time("0700") == 7
    assert np.isnan(parse_onset_time("neznámý"))


def test_raw_columns_complete():
    df = pd.DataFrame(columns=RAW_COLUMNS[:-1])
    try:
        clean_registry(df)
    except ValueError as e:
        assert "Anesthesia" in str(e)
    else:
        raise AssertionError("missing column not detected")
