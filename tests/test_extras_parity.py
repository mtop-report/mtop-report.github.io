"""The browser versions of the comparators, the mRS distribution layer and the validation
metrics must reproduce the Python implementations."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from mtop import metrics as M
from mtop.adaptive import random_effects
from mtop.extras import full_distribution, ordinal_probs
from mtop.lr import export_logreg, fit_logreg, predict_exported
from mtop.scores import POINT_SCORES, REFIT_TERMS, design

HERE = Path(__file__).parent
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _node(job: dict, tmp_path: Path) -> dict:
    inp, out = tmp_path / "in.json", tmp_path / "out.json"
    inp.write_text(json.dumps(job, allow_nan=False))
    subprocess.run([NODE, str(HERE / "node_extras.js"), str(inp), str(out)], check=True)
    return json.loads(out.read_text())


def _cohort(n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "age": rng.integers(30, 95, n).astype(float), "nihss_adm": rng.integers(0, 40, n).astype(float),
        "aspects": rng.integers(0, 11, n).astype(float), "glucose": rng.normal(7.5, 2.5, n).round(1).clip(3, 25),
        "htn": rng.integers(0, 2, n).astype(float), "dm": rng.integers(0, 2, n).astype(float),
        "arrhythmia": rng.integers(0, 2, n).astype(float), "mrs_pre": rng.integers(0, 4, n).astype(float),
        "occlusion": rng.choice(["ICA", "tandem", "M1", "M2", "cervical_ICA", "basilar", "other_posterior"], n),
        "tici": rng.choice([0.0, 1.0, 2.0, 2.5, 3.0, 4.0], n), "nihss_24h": rng.integers(0, 42, n).astype(float),
        "sich": rng.integers(0, 2, n).astype(float), "nihss_7d": rng.integers(0, 42, n).astype(float),
        "onset_to_puncture": rng.integers(40, 900, n).astype(float), "stroke_prior": rng.integers(0, 2, n).astype(float),
    })
    # exact band edges of the bedside score
    d.loc[:5, "glucose"] = [7.8, 11.0, 11.1, 7.79, 11.09, 3.0]
    d.loc[:5, "onset_to_puncture"] = [179, 180, 299, 300, 60, 1440]
    d.loc[:5, "age"] = [59, 60, 79, 80, 18, 100]
    for col, frac in [("aspects", 0.2), ("glucose", 0.15), ("htn", 0.05), ("occlusion", 0.05), ("nihss_24h", 0.1),
                      ("onset_to_puncture", 0.1), ("stroke_prior", 0.05), ("mrs_pre", 0.05)]:
        d.loc[rng.random(n) < frac, col] = np.nan if col != "occlusion" else None
    z = 1.5 - 0.04 * (d["age"] - 70) - 0.1 * d["nihss_adm"] + 0.15 * d["aspects"].fillna(7)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    return d, y


def _records(d: pd.DataFrame):
    return [{k: (None if pd.isna(v) else (v if isinstance(v, str) else float(v))) for k, v in r.items()} for r in d.to_dict("records")]


def test_logistic_regression_matches_sklearn(tmp_path):
    d, y = _cohort()
    X = design(d, REFIT_TERMS["MR PREDICTS@24H"])
    model = fit_logreg(X, y, 1)
    spec = export_logreg(model, list(X.columns), shift=-0.3)
    rows = [[None if np.isnan(v) else float(v) for v in r] for r in X.to_numpy(float)]
    js = np.array(_node({"lr": {"spec": spec, "rows": rows}}, tmp_path)["lr"])
    sk = 1 / (1 + np.exp(-(np.log(model.predict_proba(X)[:, 1] / (1 - model.predict_proba(X)[:, 1])) - 0.3)))
    assert np.max(np.abs(js - sk)) < 1e-9
    assert np.max(np.abs(predict_exported(spec, X.to_numpy(float)) - sk)) < 1e-9


def test_spline_regression_matches_python(tmp_path):
    from mtop.splines import export_spline_logreg, fit_spline_logreg

    d, y = _cohort(500, 7)
    X = d[["age", "nihss_adm", "aspects", "glucose", "onset_to_puncture", "mrs_pre", "htn"]].astype(float)
    model = fit_spline_logreg(X, y, 1)
    spec = export_spline_logreg(model, shift=0.2)
    assert set(spec["knots"]) >= {"age", "nihss_adm", "glucose", "onset_to_puncture"}
    rows = [[None if np.isnan(v) else float(v) for v in r] for r in X.to_numpy(float)]
    js = np.array(_node({"spline": {"spec": spec, "rows": rows}}, tmp_path)["spline"])
    p = model.predict_proba(X)[:, 1]
    py = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + 0.2)))
    assert np.max(np.abs(js - py)) < 1e-9


def test_points_scores_and_terms_match_python(tmp_path):
    d, _ = _cohort(300, 3)
    recs = _records(d)
    out = _node({"points": recs, "terms": {"spec": [list(t) for t in REFIT_TERMS["MR PREDICTS@24H"]], "rows": recs}}, tmp_path)
    for name, fn in POINT_SCORES.items():
        py = fn(d).to_numpy(float)
        js = np.array([np.nan if v is None else v for v in out["points"][name]], float)
        assert np.array_equal(np.isnan(py), np.isnan(js)), name
        assert np.allclose(py[~np.isnan(py)], js[~np.isnan(js)]), name
    py_terms = design(d, REFIT_TERMS["MR PREDICTS@24H"]).to_numpy(float)
    js_terms = np.array([[np.nan if v is None else v for v in r] for r in out["terms"]], float)
    assert np.array_equal(np.isnan(py_terms), np.isnan(js_terms))
    assert np.allclose(py_terms[~np.isnan(py_terms)], js_terms[~np.isnan(js_terms)])


def test_ordinal_layer_matches_python(tmp_path):
    rng = np.random.default_rng(5)
    spec = {"bg": -0.9, "bd": 0.7, "theta": [-2.0, -1.0, -0.2, 0.5, 1.2, 2.1]}
    rows = [[float(a), float(b), float(c), float(e)] for a, b, c, e in
            zip(rng.normal(0, 1.5, 50), rng.normal(-1, 1.5, 50), rng.uniform(0.02, 0.9, 50), rng.uniform(0.01, 0.6, 50))]
    out = _node({"ordinal": {"spec": spec, "rows": rows}}, tmp_path)["ordinal"]
    r = np.array(rows)
    q = ordinal_probs(r[:, 0], r[:, 1], spec["bg"], spec["bd"], spec["theta"])
    dist = full_distribution(r[:, 2], r[:, 3], q)
    assert np.max(np.abs(np.array([o["q"] for o in out]) - q)) < 1e-12
    assert np.max(np.abs(np.array([o["dist"] for o in out]) - dist)) < 1e-12
    assert np.allclose(dist.sum(1), 1.0)


def test_metrics_match_python(tmp_path):
    rng = np.random.default_rng(11)
    p = rng.uniform(0.02, 0.95, 500).round(3)          # rounding creates ties
    y = (rng.random(500) < p * 0.8).astype(float)
    out = _node({"metrics": {"y": y.tolist(), "p": p.tolist()}, "meta": {"est": [0.9, 1.1, 1.4, 1.0], "se": [0.15, 0.12, 0.2, 0.14]}}, tmp_path)
    m = out["metrics"]
    citl, slope = M.calibration_intercept_slope(y, p)
    assert abs(m["auroc"] - roc_auc_score(y, p)) < 1e-12
    assert abs(m["brier"] - brier_score_loss(y, p)) < 1e-12
    assert abs(m["citl"] - citl) < 1e-5
    assert abs(m["slope"] - slope) < 1e-5
    re = random_effects([0.9, 1.1, 1.4, 1.0], [0.15, 0.12, 0.2, 0.14])
    for k_js, k_py in [("mu", "mu"), ("lo", "lo"), ("hi", "hi"), ("tau2", "tau2"), ("i2", "i2"), ("piLo", "pi_lo"), ("piHi", "pi_hi")]:
        assert abs(out["meta"][k_js] - re[k_py]) < 1e-3, k_js
