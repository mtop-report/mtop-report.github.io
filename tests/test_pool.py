"""The reviewed pool of external validation results: what is accepted, and the published file."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtop import pool as P

ROOT = Path(__file__).resolve().parents[1]
FP = "a" * 64


def _result(**kw):
    m = {"n": 200, "events": 60, "auroc": 0.8, "auroc_ci": [0.74, 0.86], "se_logit": 0.15, "brier": 0.17,
         "citl": 0.1, "slope": 0.9, "calibration": [{"pred": 0.2, "obs": 0.25, "n": 40}, {"pred": 0.5, "obs": 0.45, "n": 8}]}
    small = {"n": 40, "events": 6, "auroc": 0.7, "auroc_ci": [0.5, 0.9], "se_logit": 0.4, "brier": 0.1, "citl": 0, "slope": 1,
             "calibration": []}
    j = {"format": "mtop-validation/2", "model_version": "1.0.0", "model_fingerprint": FP, "site": "X", "generated": "2026-10-18",
         "rows": 250, "columns_found": ["age"], "demo": False, "period": [2026, 2026], "development_period": [2005, 2025],
         "rows_with_year": 250, "rows_after_development": 250,
         "stages": {"admission": {"good_outcome": {"chosen": "LR", "models": {"LR": m}}, "death90": {"chosen": "LR", "models": {"LR": small}}}}}
    j.update(kw)
    return j


def test_demo_and_unidentified_results_are_refused():
    assert any("demo" in e for e in P.check_result(_result(demo=True), FP, 10)[1])
    assert any("fingerprint" in e for e in P.check_result(_result(model_fingerprint=None), FP, 10)[1])
    assert any("unexpected" in e for e in P.check_result(_result(predictions=[0.1, 0.2]), FP, 10)[1])
    assert P.check_result({"format": "other"}, FP, 10)[1]


def test_small_cells_are_removed_before_publication():
    clean, errors, notes = P.check_result(_result(), FP, 10)
    assert not errors
    st = clean["stages"]["admission"]
    assert st["death90"]["models"]["LR"] == {"n": 40, "insufficient": True}       # 6 events
    assert [b["n"] for b in st["good_outcome"]["models"]["LR"]["calibration"]] == [40]
    assert st["good_outcome"]["auroc"] == 0.8                                       # chosen model at the stage level
    assert any("earlier model" in n for n in P.check_result(_result(), "b" * 64, 10)[2])


def test_add_replaces_the_same_site_and_version(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "POOL_FILE", tmp_path / "validation_pool.json")
    monkeypatch.setattr(P, "current_fingerprint", lambda: FP)
    f = tmp_path / "r.json"
    f.write_text(json.dumps(_result()))
    P.add_result(f, "Hospital X", "DUA 1", "VG")
    P.add_result(f, "Hospital X", "DUA 1", "VG")
    pool = json.loads((tmp_path / "validation_pool.json").read_text())
    assert len(pool["results"]) == 1 and pool["results"][0]["result"]["site"] == "Hospital X"
    f.write_text(json.dumps(_result(demo=True)))
    with pytest.raises(SystemExit):
        P.add_result(f, "Hospital Y", "DUA 2", "VG")


def test_published_pool_holds_reviewed_aggregates_only():
    pool = json.loads((ROOT / "artifacts" / "public" / "validation_pool.json").read_text())
    assert pool["format"] == "mtop-validation-pool/1"
    for e in pool["results"]:
        assert e["site"] and e["agreement"] and e["reviewer"] and e["accepted"]
        _, errors, notes = P.check_result(e["result"], None, 10)
        assert not errors and not [n for n in notes if "fewer than" in n or "removed" in n]
