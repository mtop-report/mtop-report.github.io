"""Corrections layer and the local data studio (synthetic data only)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtop import corrections as C
from mtop.synth import make_synthetic

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def studio_dir(tmp_path, monkeypatch):
    d = tmp_path / "studio"
    monkeypatch.setenv("MTOP_STUDIO_DIR", str(d))
    return d


def _raw(n=12):
    raw = make_synthetic(n=n, seed=5).reset_index(drop=True)
    raw.loc[0, "BMI"] = 257.8           # decimal shift -> suggestion 25.78
    raw.loc[1, "Systolic BP"] = 10
    raw.loc[2, "Intervention procedure"] = "ADAPT"
    raw.loc[2, "Extraction system"] = "Preset"   # flagged: label and device disagree
    return raw


def test_values_are_compared_in_one_text_form():
    assert C.norm_value(25.0) == "25" and C.norm_value(25.78) == "25.78" and C.norm_value(np.nan) == ""
    assert C.norm_value(pd.Timestamp("2012-08-27")) == "2012-08-27" and C.norm_value("  ANO ") == "ANO"


def test_correction_applies_only_against_the_recorded_value(studio_dir):
    raw = _raw()
    pid = int(raw.loc[0, "Patient ID"])
    cid = C.add_correction(pid, "BMI", raw.loc[0, "BMI"], "25.78", "decimal shift", "tester")
    fixed, rep = C.apply_corrections(raw)
    assert fixed.loc[0, "BMI"] == pytest.approx(25.78) and raw.loc[0, "BMI"] == 257.8     # the original is untouched
    assert rep.set_index("id").loc[cid, "result"] == "applied"
    # a later export that already holds the fix
    raw2 = raw.copy()
    raw2.loc[0, "BMI"] = 25.78
    assert C.apply_corrections(raw2)[1].set_index("id").loc[cid, "result"] == "already in the source"
    # ... or something else: not applied
    raw3 = raw.copy()
    raw3.loc[0, "BMI"] = 30.1
    f3, r3 = C.apply_corrections(raw3)
    assert r3.set_index("id").loc[cid, "result"] == "conflict" and f3.loc[0, "BMI"] == 30.1
    C.revert_correction(cid, "tester")
    assert not len(C.apply_corrections(raw)[1])
    log = C.load_audit()
    assert list(log["action"]) == ["correct", "revert"] and set(log["author"]) == {"tester"}


def test_new_correction_supersedes_and_confirm_changes_nothing(studio_dir):
    raw = _raw()
    pid = int(raw.loc[1, "Patient ID"])
    C.add_correction(pid, "Systolic BP", 10, "100", "missing digit", "a")
    C.add_correction(pid, "Systolic BP", 10, "110", "checked in the chart", "b")
    corr = C.load_corrections()
    assert list(corr["status"]) == ["superseded", "active"]
    fixed, _ = C.apply_corrections(raw)
    assert fixed.loc[1, "Systolic BP"] == 110
    pid2 = int(raw.loc[2, "Patient ID"])
    C.add_correction(pid2, "Extraction system", "Preset", "Preset", "checked", "a", kind="confirmed")
    fixed2, rep2 = C.apply_corrections(raw)
    assert fixed2.loc[2, "Extraction system"] == "Preset" and "confirmed" in set(rep2["result"])
    with pytest.raises(ValueError):
        C.add_correction(pid, "Systolic BP", 10, "10", "", "a")          # nothing to change
    with pytest.raises(ValueError):
        C.add_correction(pid, "Systolic BP", 10, "100", "", "")          # no author


def test_password_hash(studio_dir):
    assert not C.password_is_set() and not C.check_password("anything")
    with pytest.raises(ValueError):
        C.set_password("short")
    C.set_password("correct horse")
    assert C.check_password("correct horse") and not C.check_password("correct horsE")
    assert "correct horse" not in (studio_dir / "password.json").read_text()


def test_every_finding_is_placed_in_every_version(studio_dir):
    from mtop import studio_data as D

    raw = _raw(40)
    pid = int(raw.loc[0, "Patient ID"])
    C.add_correction(pid, "BMI", 257.8, "25.78", "decimal shift", "tester")
    dat = D.build(raw)
    iss = dat["issues"]
    assert set(iss["column"]) <= set(D.ISSUE_RAW)
    bmi = iss[(iss["patient_id"] == pid) & (iss["column"] == "bmi")]
    assert bmi.empty                                                # corrected before cleaning: no finding left
    sbp = iss[iss["column"] == "sbp"].iloc[0]
    assert sbp["kind"] == "missing" and sbp["status"] == "open"
    mk, summ = D.marks(dat, "raw", D.table(dat, "raw"))
    assert mk.loc[0, "BMI"] == "corrected" and mk.loc[1, "Systolic BP"] == "missing"
    assert mk.loc[2, "Extraction system"] == "" and mk.loc[2, "Intervention procedure"] == "flag"
    assert "Systolic" in summ.loc[1]
    for v in D.versions():
        tab = D.table(dat, v)
        mk, _ = D.marks(dat, v, tab)
        assert mk.shape == tab.shape
    assert D.column_label("Occlusion site", "cs", "raw") == "Místo okluze"
    assert D.suggestion({"note": "possibly 25.78 (decimal shift)", "action": "outside 12-70 -> missing"}) == "25.78"


def test_studio_app_runs_behind_the_password(studio_dir, tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    src = tmp_path / "registry.xlsx"
    _raw(60).to_excel(src, index=False)
    monkeypatch.setenv("MTOP_STUDIO_INPUT", str(src))
    C.set_password("correct horse")
    at = AppTest.from_file(str(ROOT / "src" / "mtop" / "studio_app.py"), default_timeout=60).run()
    assert not at.exception
    assert not at.tabs                                              # locked: only the password form
    at.text_input[0].input("wrong")                                 # a form: value and submit in one run
    at.button[0].click().run()
    assert any("Wrong password" in e.value for e in at.error) and not at.tabs
    at.text_input[0].input("correct horse")
    at.button[0].click().run()
    assert not at.exception and len(at.tabs) == 4
    at.text_input(key="author").input("tester").run()
    assert not at.exception
    for v in ("clean", "cohort", "admission", "procedure", "h24", "d7", "raw"):
        at.selectbox(key="ver").select(v).run()
        assert not at.exception, v
    at.button(key="prep-xl").click().run()                          # the corrected export, built on request
    assert not at.exception
    at.segmented_control(key="lang").set_value("cs").run()
    assert not at.exception and any("Tabulky" in tb.label for tb in at.tabs)
