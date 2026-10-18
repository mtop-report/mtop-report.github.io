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


def test_procedure_device_checks_flag_without_changing_values():
    raw = _raw(7)
    raw["Stent combo"] = "žádný"
    raw["IAT"] = "NE"
    rows = [("MT+stent", "PENUMBRA + Preset"), ("MT+stent", "žádný"), ("cPTA+stent", "Catch"), ("ADAPT", "Preset"),
            ("Solumbra", "Catch"), ("kombinovaný výkon", "žádný"), ("LIAT", "žádný")]
    for i, (p, d) in enumerate(rows):
        raw.loc[i, ["Intervention procedure", "Extraction system"]] = [p, d]
    clean, issues = clean_registry(raw)
    tech = [None if pd.isna(v) else v for v in clean["technique"].tolist()[:7]]
    assert tech == ["solumbra", None, "stent_retriever", "aspiration", "solumbra", "combined", "none"]
    acts = issues.groupby("patient_id")["action"].apply(list).to_dict()
    pid = clean["patient_id"].tolist()
    assert any("technique unknown" in a for a in acts[pid[1]])                       # MT+stent without a device
    assert any("no thrombectomy in the procedure label" in a for a in acts[pid[2]])  # cPTA+stent with a stent retriever
    assert any("differs from the extraction device" in a for a in acts[pid[3]])      # ADAPT with a stent retriever
    assert pid[4] not in acts or not any("differs" in a for a in acts[pid[4]])        # Solumbra listing only the retriever: fine
    assert any("Solumbra or combined), but the extraction" in a for a in acts[pid[5]])
    assert any("stent in the procedure label" in a for a in acts[pid[0]])            # MT+stent, stent column 'žádný'
    assert any("intra-arterial lysis in the procedure label" in a for a in acts[pid[6]])   # LIAT with IAT = NE
    assert clean.loc[0, "stenting"] == 1 and clean.loc[6, "ia_lysis"] == 1            # values kept


def test_rare_occlusion_codes_are_pooled_for_models():
    from mtop.clean import pool_occlusion

    detail = pd.Series(["M3", "ACA", "tandem_cca", "CCA", "basilar_vertebral", "vertebral", "PCA", "M1", None, "M2"])
    counts = pd.Series({"M3": 5, "ACA": 3, "tandem_cca": 4, "CCA": 6, "basilar_vertebral": 26, "vertebral": 20, "PCA": 6,
                        "M1": 600, "M2": 130, "tandem": 200, "cervical_ICA": 60, "basilar": 130})
    allowed = {"ICA", "tandem", "cervical_ICA", "M1", "M2", "basilar", "basilar_vertebral", "other_posterior"}
    out = pool_occlusion(detail, counts, 10, allowed).tolist()
    # M3 joins M2 (same artery), CCA+ICA+MCA joins tandem, CCA joins cervical ICA, AB+AV (26) is its own
    # category, vertebral and PCA form vertebral/PCA, ACA is merged with nothing (not used by the models)
    assert out[:8] == ["M2", None, "tandem", "cervical_ICA", "basilar_vertebral", "other_posterior", "other_posterior", "M1"]
    assert out[8] is None and out[9] == "M2"
    # below the threshold AB+AV joins basilar
    assert pool_occlusion(pd.Series(["basilar_vertebral"]), pd.Series({"basilar_vertebral": 6, "basilar": 100}), 10, allowed).tolist() == ["basilar"]
    # a pooled group counts all its codes: vertebral 6 + PCA 5 = 11 is enough
    assert pool_occlusion(pd.Series(["PCA"]), pd.Series({"vertebral": 6, "PCA": 5}), 10, allowed).tolist() == ["other_posterior"]
    # a code with no modelled parent is not used even when common (until it becomes a model option)
    assert pool_occlusion(pd.Series(["ACA"]), pd.Series({"ACA": 50}), 10, allowed).tolist() == [None]
    # a code that reaches the threshold but is not a model option keeps being pooled
    assert pool_occlusion(pd.Series(["M3"]), pd.Series({"M3": 40, "M2": 100}), 10, allowed).tolist() == ["M2"]


def test_registry_occlusion_coding():
    """The user's coding: tandem = ACI+ACM, ACI-odstup +T typ and ACC+ACI+ACM (separate from cervical
    ICA/CCA and intracranial ICA); cervical ICA/CCA = ACI-odstup and ACC; ACA and AB+AV their own codes."""
    from mtop.clean import OCCLUSION_DETAIL, OCCLUSION_PARENT

    tandem = {k for k, v in OCCLUSION_DETAIL.items() if v in ("tandem", "tandem_cca")}
    assert tandem == {"ACI+ACM", "ACI-odstup +T typ", "ACC+ACI+ACM"}
    assert OCCLUSION_PARENT["tandem_cca"] == "tandem"
    cervical = {k for k, v in OCCLUSION_DETAIL.items() if v == "cervical_ICA" or OCCLUSION_PARENT.get(v) == "cervical_ICA"}
    assert cervical == {"ACI-odstup", "ACC"}
    assert OCCLUSION_DETAIL["ACI-Intracranial"] == "ICA"
    assert OCCLUSION_DETAIL["ACA"] == "ACA" and "ACA" not in OCCLUSION_PARENT
    assert OCCLUSION_DETAIL["AB+AV"] == "basilar_vertebral"


def test_unmodelled_category_is_missing_in_design():
    from mtop.config import features
    from mtop.features import _column

    f = next(x for x in features() if x.key == "occlusion")
    assert "occlusion__ACA" not in f.columns and "occlusion__basilar_vertebral" in f.columns
    X = _column(pd.DataFrame({"occlusion": ["ACA", "M1", None, "basilar_vertebral"]}), f)
    assert X.iloc[0].isna().all() and X.iloc[2].isna().all()      # ACA is used like a missing site
    assert X.iloc[1]["occlusion__M1"] == 1 and X.iloc[3]["occlusion__basilar_vertebral"] == 1


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
