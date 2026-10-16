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

FILES = ["model_bundle.json", "validation.json", "explorer.json"]


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
            assert bad not in raw, f"{bad} found in {name}"
        for hname in hospitals:
            assert f'"{hname}"' not in raw, f"hospital name {hname} in {name}"
        dates = set(re.findall(r"\b(?:19|20)\d\d-\d\d-\d\d\b", raw))
        allowed = set()
        data = json.loads(raw)
        if isinstance(data, dict) and "generated" in data:
            allowed.add(data["generated"])
        assert dates <= allowed, f"unexpected dates in {name}: {dates - allowed}"


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
        for col in ("all", "dev", "test", "good", "poor"):
            v = row[col]
            if row["kind"] == "pct" and v.get("n") is not None:
                assert v["n"] >= k or v["n"] == 0


def test_validation_bins_and_no_extremes():
    v = _load("validation.json")
    k = int(settings()["privacy"]["min_cell"])
    for st in v["stages"].values():
        for tv in st["targets"].values():
            for b in tv["curves"]["calibration"]:
                assert b["n"] >= k
            for b in tv["deployed"].get("reality_bins", []):
                assert b["n"] >= k
            for rows in tv["explain"]["dependence"].values():
                for r in rows:
                    assert r["n"] >= k
                    assert "lo" not in r and "hi" not in r
            for curve in tv["curves"]["roc"].values():
                assert len(curve) == 51  # fixed FPR grid, not raw operating points


def test_tree_leaves_are_large():
    b = _load("model_bundle.json")
    min_leaf = int(settings()["privacy"]["min_leaf"])
    for per_target in b["models"].values():
        for spec in per_target.values():
            for t in spec["trees"]:
                if len(t["f"]) == 1:
                    continue
                for f, c in zip(t["f"], t["c"]):
                    if f == -1:
                        assert c >= min_leaf
            assert "feature_infos" not in json.dumps(spec)
