"""Public outputs must not contain patient-level information.

Runs against artifacts/public (the files that are committed and published). Rules:
* explorer groups shown only with n >= min_cell; mRS distributions only with n >= 3 * min_cell;
* calibration / reality-check bins have n >= min_cell;
* every tree leaf summarises >= min_leaf training patients;
* no identifiers, dates, hospital names or raw value extremes anywhere.
"""
from __future__ import annotations

import json
import re

import pytest

from mtop.config import settings
from mtop.paths import PUBLIC

FILES = ["model_bundle.json", "validation.json", "explorer.json", "extras.json", "adaptive.json", "federated.json",
         "associations.json", "data_checks.json", "preprocessing.json", "raw_profile.json", "reporting.json"]
# column names that may appear as labels (never with values) in the data report
COLUMN_LABELS = {"preprocessing.json": ("Patient ID", "onset_date"), "raw_profile.json": ("Patient ID",),
                 "data_checks.json": ("onset_date",)}


def _load(name):
    f = PUBLIC / name
    if not f.exists():
        pytest.skip(f"{name} not built")
    return json.loads(f.read_text(encoding="utf-8"))


def _walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, f"{path}.{k}")
            yield path, k, v
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, f"{path}[{i}]")


def test_no_identifiers_or_dates():
    hospitals = ["Krnov", "Třinec", "Karviná", "Opava", "Havířov", "Fifejdy", "Nový Jičín", "Vítkovice", "F-M"]
    for name in FILES:
        raw = (PUBLIC / name).read_text(encoding="utf-8") if (PUBLIC / name).exists() else ""
        if not raw:
            continue
        for bad in ("patient_id", "Patient ID", "onset_date", "referring_hospital"):
            if bad in COLUMN_LABELS.get(name, ()):
                continue
            assert bad not in raw, f"{bad} found in {name}"
        for hname in hospitals:
            assert f'"{hname}"' not in raw, f"hospital name {hname} in {name}"
        dates = set(re.findall(r"\b(?:19|20)\d\d-\d\d-\d\d\b", raw))
        allowed = set()
        data = json.loads(raw)
        if isinstance(data, dict) and "generated" in data:
            allowed.add(data["generated"])
        assert dates <= allowed, f"unexpected dates in {name}: {dates - allowed}"


def test_explorer_two_way_table_is_coarse():
    """The only cross-tabulation (age band x NIHSS band) needs >= 3 * min_cell patients per cell."""
    e = _load("explorer.json")
    k = int(settings()["privacy"]["min_cell"])
    m = e.get("age_nihss")
    if not m:
        pytest.skip("no two-way table")
    assert m["min_n"] >= 3 * k
    for row in m["cells"]:
        for c in row:
            if c.get("suppressed"):
                assert set(c) == {"suppressed"}
            else:
                assert c["n"] >= 3 * k


def test_explorer_small_cells_suppressed():
    e = _load("explorer.json")
    k = int(settings()["privacy"]["min_cell"])
    for dim in e["dimensions"].values():
        for g in dim["groups"]:
            if g.get("suppressed"):
                assert "n" not in g and "dist" not in g and "good" not in g
            else:
                assert g["n"] >= k
                if "dist" in g:
                    assert g["n"] >= 3 * k
    for series in e["time_curve"].values():
        for b in series:
            assert b.get("suppressed") or b["n"] >= k
    for r in e["trends"]:
        assert r["n"] >= k
    for r in e["registry_years"]:
        assert r["n"] >= k
    for row in e["table1"]["rows"]:
        if "section" in row:
            continue
        for col in ("all", "dev", "test", "good", "poor"):
            v = row[col]
            if row["kind"] == "pct" and v.get("n") is not None:
                assert v["n"] >= k or v["n"] == 0


def test_validation_bins_and_no_extremes():
    v = _load("validation.json")
    k = int(settings()["privacy"]["min_cell"])
    for sp in v["splits"].values():
        for st in sp["stages"].values():
            for tv in st["targets"].values():
                for bins in tv["curves"]["calibration"].values():
                    for b in bins:
                        assert b["n"] >= k
                for curve in tv["curves"]["roc"].values():
                    assert len(curve) == 51  # fixed FPR grid, not raw operating points
                for rows in tv.get("thresholds", {}).values():   # rates only; small 2x2 cells suppressed
                    for r in rows:
                        assert set(r) in ({"t", "suppressed"}, {"t", "sens", "spec", "ppv", "npv", "pos", "acc"})
                for rows in tv.get("subgroups", {}).values():
                    for r in rows:
                        assert r.get("suppressed") or r["n"] >= k
    for per in v["deployed"].values():
        for dep in per.values():
            for lr in dep["learners"].values():
                for b in lr.get("reality_bins", []):
                    assert b["n"] >= k
                ex = lr.get("explain") or {}
                for rows in ex.get("dependence", {}).values():
                    for r in rows:
                        assert r["n"] >= k
                        assert "lo" not in r and "hi" not in r


def test_tree_leaves_are_large():
    b = _load("model_bundle.json")
    min_leaf = int(settings()["privacy"]["min_leaf"])
    for per_target in b["models"].values():
        for entry in per_target.values():
            for name, spec in entry["learners"].items():
                assert "feature_infos" not in json.dumps(spec)
                if spec["kind"] in ("lgbm", "xgb"):
                    # counted on the training patients at export (treejson.min_leaf_count / leaf_counts)
                    assert spec["min_leaf_count"] >= min_leaf, name
                # LightGBM publishes leaf counts; XGBoost publishes cover (sum of p(1-p) <= n/4)
                floor = min_leaf if spec["kind"] == "lgbm" else min_leaf / 4
                for t in spec.get("trees", []):
                    if len(t["f"]) == 1:
                        continue
                    for f, c in zip(t["f"], t["c"]):
                        if f == -1:
                            assert c >= floor, f"{name}: leaf weight {c}"


def test_adaptive_outputs_are_aggregates():
    a = _load("adaptive.json")
    k = int(settings()["privacy"]["min_cell"])
    me = int(a["settings"]["min_events"])

    def flat(block):   # one result per learner (or a single model in older files)
        for per_target in block.values():
            for r in per_target.values():
                yield from (r.values() if a.get("learners") else [r])

    for r in flat(a["results"]):
        for yr in r["years"]:
            assert yr["n"] >= k
            for m in yr["by"].values():
                if m["auroc"] is not None:
                    assert yr["events"] >= me and yr["n"] - yr["events"] >= me
        for m in r["monitor"]:
            assert m["n"] >= k
    for r in flat(a["iecv"]):
        for e in r["eras"]:
            if not e.get("suppressed"):
                assert e["n"] >= 30 and e["events"] >= me
    assert all(n >= k for n in a["availability"]["n"])


def test_extras_are_aggregates():
    x = _load("extras.json")
    k = int(settings()["privacy"]["min_cell"])
    for per_target in x["uncertainty"].values():
        for bins in per_target.values():
            for b in bins:
                assert b["n"] >= k
    for ov in x["ordinal_validation"].values():
        assert ov["n"] >= 3 * k


def test_federated_outputs_are_aggregates():
    f = _load("federated.json")
    k = int(settings()["privacy"]["min_cell"])
    groups = [c[t] for c in f["centres"] for t in ("good_outcome", "death90")]
    for res in f["outcomes"].values():
        groups += res["per_centre"] + res["loso"]
        for r in res.get("regression", {}).values():
            groups += r["per_centre"]
    for st in f["stages"]:
        groups += st["per_centre"]
    for g in groups:
        assert g["n"] >= k and g["events"] >= k and g["n"] - g["events"] >= k
    raw = json.dumps(f)
    for key in ('"nodes"', '"thr"', '"miss_left"', '"edges"', '"beta"', '"calibration"'):
        assert key not in raw, f"{key} should not be published (model internals or score-band counts)"
    for path, k2, v in _walk(f):
        if k2 == "trees":                       # tree counts and learning-curve positions only
            assert isinstance(v, int) or all(isinstance(t, int) for t in v), path
    assert f["equivalence"]["gbdt"]["min_leaf"] >= int(settings()["privacy"]["min_leaf"])


def test_associations_are_aggregates():
    """Odds ratios only; a yes/no factor needs >= min_cell patients in each group."""
    a = _load("associations.json")
    k = int(settings()["privacy"]["min_cell"])
    assert a["n"] >= 10 * k
    keys = {t["key"] for t in a["terms"]}
    for t in a["terms"]:
        if t["kind"] in ("bin", "occ"):
            assert t["n1"] >= k and t["n0"] >= k
    for res in a["outcomes"].values():
        assert set(res["terms"]) == keys
        assert min(res["events"], res["n"] - res["events"]) >= 10 * k
        for r in res["terms"].values():
            assert r["unadjusted"]["n"] >= 10 * k
            assert set(r["adjusted"]) == {"or", "lo", "hi"}


def test_data_checks_are_counts_only():
    """The cleaning summary lists how many records each check touched; small counts are hidden."""
    d = _load("data_checks.json")
    k = int(settings()["privacy"]["min_cell"])
    for r in d["checks"]:
        assert set(r) <= {"kind", "label", "n", "small", "by_column"}
        assert (r["n"] is None and r["small"]) or (r["n"] >= k and not r["small"])
        for c in r.get("by_column", []):
            assert (c["n"] is None and c["small"]) or (c["n"] >= k and not c["small"])


def _counts(o):
    """Every {"n": ..., "small": ...} cell in a nested structure."""
    if isinstance(o, dict):
        if "n" in o and "small" in o:
            yield o
        for v in o.values():
            yield from _counts(v)
    elif isinstance(o, list):
        for v in o:
            yield from _counts(v)


def test_preprocessing_report_is_aggregate():
    """Data report: suppressed small cells, no raw values of identifiers, dates or free text."""
    k = int(settings()["privacy"]["min_cell"])
    for name in ("preprocessing.json", "raw_profile.json"):
        p = _load(name)
        for c in _counts(p):
            if c.get("present") is False:      # a coding listed for a value absent from this export
                continue
            assert (c["n"] is None and c["small"]) or c["n"] == 0 or (c["n"] >= k and not c["small"]), (name, c)
    raw = _load("raw_profile.json")
    for col in raw["columns"]:
        if col["column"] in ("Patient ID", "Date of stroke onset", "Time of stroke onset", "Age"):
            assert "values" not in col
        if col["column"] in ("IVT in different hospital", "Transfer from different hospital"):
            assert {v["value"] for v in col.get("values", [])} <= {"FNO", "žádné", "another hospital"}
    pp = _load("preprocessing.json")
    for r in pp["cholesterol"]:
        assert r["n"] >= k
    who = pp["outcome"]["who_missing_n"]
    assert who["missing"] >= k and who["observed"] >= k


def test_reporting_is_aggregate():
    r = _load("reporting.json")
    raw = json.dumps(r)
    assert "patient" not in raw.lower()
    for per in r["development"].values():
        for v in per.values():
            assert v["n_available"] >= 100


def test_demo_csv_is_synthetic_template():
    from mtop.extras import template_columns

    f = PUBLIC / "demo_synthetic.csv"
    if not f.exists():
        pytest.skip("demo not built")
    assert f.read_text(encoding="utf-8").splitlines()[0].split(",") == template_columns()
