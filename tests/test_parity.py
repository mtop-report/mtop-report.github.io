"""The browser models (web/model.js) must reproduce the Python learners exactly.

LightGBM: raw scores and TreeSHAP against `predict(raw_score=True)` / `predict(pred_contrib=True)`;
XGBoost: margins and TreeSHAP against `output_margin=True` / `pred_contribs=True`; the regressions
(linear and with splines): log-odds against scikit-learn and attributions against
learners.contributions (coefficient x standardised value, summing to the log-odds). Rows include
every missing-value pattern we use.
"""
from __future__ import annotations

import json
import pickle
import shutil
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from mtop import learners as Lr
from mtop.treejson import lgbm_to_json

HERE = Path(__file__).parent
ROOT = HERE.parent
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _node_eval(spec: dict, X: np.ndarray, tmp_path: Path):
    rows = [[None if np.isnan(v) else float(v) for v in r] for r in X]
    inp, out = tmp_path / "in.json", tmp_path / "out.json"
    inp.write_text(json.dumps({"spec": spec, "rows": rows}))
    subprocess.run([NODE, str(HERE / "node_eval.js"), str(inp), str(out)], check=True)
    return json.loads(out.read_text())


def _synthetic(n=900, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "age": rng.normal(70, 12, n).round(),
        "nihss_adm": rng.integers(0, 40, n).astype(float),
        "aspects": rng.integers(0, 11, n).astype(float),
        "onset_to_puncture": rng.gamma(3, 60, n).round(),
        "glucose": rng.normal(7.5, 2.5, n).round(1).clip(3, 25),
        "flag": rng.integers(0, 2, n).astype(float),
        "zeroish": np.where(rng.random(n) < 0.3, 0.0, rng.normal(5, 2, n)),
        "cat__a": rng.integers(0, 2, n).astype(float),
    })
    for col, frac in [("aspects", 0.3), ("onset_to_puncture", 0.1), ("flag", 0.05), ("cat__a", 0.05), ("glucose", 0.15)]:
        X.loc[rng.random(n) < frac, col] = np.nan
    z = 2 - 0.04 * (X["age"] - 70) - 0.12 * X["nihss_adm"] + 0.2 * X["aspects"].fillna(7) - 0.004 * X["onset_to_puncture"].fillna(200)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    return X, y


def _eval_rows(X):
    Xe = X.sample(250, random_state=1).to_numpy(float)
    edge = np.array([[np.nan] * X.shape[1], [0.0] * X.shape[1], [120, 42, 10, 1440, 30, 1, 0, 1]], float)
    return np.vstack([Xe, edge])


@pytest.mark.parametrize("use_missing", [True, False])
def test_js_matches_lightgbm(tmp_path, use_missing):
    X, y = _synthetic()
    params = dict(objective="binary", num_leaves=9, max_depth=4, min_child_samples=20, learning_rate=0.05,
                  bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, verbosity=-1, seed=3,
                  deterministic=True, force_row_wise=True, use_missing=use_missing, zero_as_missing=False,
                  monotone_constraints=[-1, -1, 1, -1, -1, 0, 0, 0], monotone_constraints_method="advanced")
    booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=120)
    mj = lgbm_to_json(booster)
    spec = {"kind": "lgbm", "trees": mj["trees"], "sigmoid": mj["sigmoid"], "intercept_recent": 0.0}
    Xe = _eval_rows(X)
    js = _node_eval(spec, Xe, tmp_path)
    raw_py = booster.predict(Xe, raw_score=True)
    contrib_py = booster.predict(Xe, pred_contrib=True)
    raw_js = np.array([r["raw"] for r in js])
    phi_js = np.array([r["phi"] for r in js])
    base_js = np.array([r["base"] for r in js])
    assert np.max(np.abs(raw_js - raw_py)) < 1e-6
    assert np.max(np.abs(phi_js - contrib_py[:, :-1])) < 1e-6
    assert np.max(np.abs(base_js - contrib_py[:, -1])) < 1e-6
    assert np.max(np.abs(base_js + phi_js.sum(1) - raw_js)) < 1e-6


def test_js_matches_xgboost(tmp_path):
    import xgboost as xgb

    from mtop.tuning import fit_xgb_fixed

    X, y = _synthetic(seed=4)
    s = {"learning_rate": 0.05, "max_depth": 4, "min_child_weight": 5.0, "subsample": 0.8, "colsample_bytree": 0.8,
         "gamma": 0.01, "lambda": 1.0, "trees": 150}
    f = Lr.Fitted("XGBoost", fit_xgb_fixed(X, y, s, 3), s, list(X.columns))
    spec = Lr.export(f, 0.0, X, min_leaf=20)
    Xe = _eval_rows(X)
    js = _node_eval(spec, Xe, tmp_path)
    d = xgb.DMatrix(pd.DataFrame(Xe, columns=X.columns), missing=np.nan)
    margin = f.model.booster.predict(d, output_margin=True)
    contrib = f.model.booster.predict(d, pred_contribs=True)
    raw_js = np.array([r["raw"] for r in js])
    phi_js = np.array([r["phi"] for r in js])
    base_js = np.array([r["base"] for r in js])
    assert np.max(np.abs(raw_js - margin)) < 1e-4
    assert np.max(np.abs(phi_js - contrib[:, :-1])) < 1e-4
    assert np.max(np.abs(base_js - contrib[:, -1])) < 1e-4


@pytest.mark.parametrize("name", ["Logistic regression", "Logistic regression (splines)"])
def test_js_matches_regressions(tmp_path, name):
    X, y = _synthetic(seed=2)
    f = Lr.tune(name, X, y, 1, trials=1, mono=[0] * X.shape[1])
    spec = Lr.export(f, 0.0, X)
    Xe = _eval_rows(X)
    js = _node_eval(spec, Xe, tmp_path)
    De = pd.DataFrame(Xe, columns=X.columns)
    lg = f.logit(De)
    phi_py, base_py = Lr.contributions(f, De)
    raw_js = np.array([r["raw"] for r in js])
    phi_js = np.array([r["phi"] for r in js])
    assert np.max(np.abs(raw_js - lg)) < 1e-8
    assert np.max(np.abs(phi_js - phi_py.to_numpy())) < 1e-8
    assert np.max(np.abs(phi_py.to_numpy().sum(1) + base_py - lg)) < 1e-8


def test_js_matches_interaction_and_imputation_regressions(tmp_path):
    X, y = _synthetic(seed=5)
    f = Lr.tune("LR + interactions", X, y, 1, trials=1, mono=[0] * X.shape[1])
    spec = Lr.export(f, 0.0, X)
    assert len(spec["spec"]["products"]) == 2
    Xe = _eval_rows(X)
    js = _node_eval(spec, Xe, tmp_path)
    De = pd.DataFrame(Xe, columns=X.columns)
    phi_py, base_py = Lr.contributions(f, De)
    assert np.max(np.abs(np.array([r["raw"] for r in js]) - f.logit(De))) < 1e-8
    assert np.max(np.abs(np.array([r["phi"] for r in js]) - phi_py.to_numpy())) < 1e-8
    assert np.max(np.abs(phi_py.to_numpy().sum(1) + base_py - f.logit(De))) < 1e-8
    # LR + multiple imputation: one spline regression on stacked imputations; complete rows are exact
    g = Lr.tune("LR + multiple imputation", X, y, 1, trials=1, mono=[0] * X.shape[1])
    spec = Lr.export(g, 0.0, X)
    full = X.dropna().sample(120, random_state=2)
    js = _node_eval(spec, full.to_numpy(float), tmp_path)
    assert np.max(np.abs(np.array([r["raw"] for r in js]) - g.logit(full))) < 1e-8


def test_js_matches_ebm_and_stacking(tmp_path):
    from mtop.extra_learners import ebm_spec_logit, ebm_spec_terms

    X, y = _synthetic(seed=6)
    f = Lr.tune("EBM", X, y, 1, trials=1, mono=[0] * X.shape[1])
    spec = Lr.export(f, 0.0, X)
    assert spec["spec"]["min_samples_leaf"] >= 20
    Xe = _eval_rows(X)
    # values exactly on a published cut point go to the upper interval, as in interpret
    cut_rows = []
    for t in spec["spec"]["terms"]:
        if len(t["f"]) == 1 and t["cuts"][0]:
            r = Xe[0].copy()
            r[t["f"][0]] = t["cuts"][0][0]
            cut_rows.append(r)
    Xe = np.vstack([Xe] + cut_rows[:10])
    js = _node_eval(spec, Xe, tmp_path)
    raw_js = np.array([r["raw"] for r in js])
    assert np.max(np.abs(raw_js - ebm_spec_logit(spec["spec"], Xe))) < 1e-10
    assert np.max(np.abs(raw_js - f.logit(pd.DataFrame(Xe, columns=X.columns)))) < 1e-5   # 7-digit scores
    phi_ref = np.zeros((len(Xe), X.shape[1]))
    for i, row in enumerate(Xe):
        sc, owners = ebm_spec_terms(spec["spec"], row)
        for v, o in zip(sc, owners):
            for j in o:
                phi_ref[i, j] += v / len(o)
    assert np.max(np.abs(np.array([r["phi"] for r in js]) - phi_ref)) < 1e-10
    s = Lr.tune("Stacking", X, y, 1, trials=2, mono=[0] * X.shape[1])
    spec = Lr.export(s, 0.0, X)
    js = _node_eval(spec, Xe, tmp_path)
    De = pd.DataFrame(Xe, columns=X.columns)
    raw_js = np.array([r["raw"] for r in js])
    assert np.max(np.abs(raw_js - s.logit(De))) < 1e-4
    phi_js, base_js = np.array([r["phi"] for r in js]), np.array([r["base"] for r in js])
    assert np.max(np.abs(base_js + phi_js.sum(1) - raw_js)) < 1e-8


def test_exported_bundle_matches_deployed_models(tmp_path):
    """If the real pipeline has been run, check every published learner against the saved models."""
    bundle_file = ROOT / "artifacts" / "public" / "model_bundle.json"
    private = ROOT / "artifacts" / "private"
    if not bundle_file.exists() or not any(private.glob("deployed_*.pkl")):
        pytest.skip("pipeline outputs not present")
    bundle = json.loads(bundle_file.read_text())
    rng = np.random.default_rng(0)
    for stage, per_target in bundle["models"].items():
        for target, entry in per_target.items():
            pk = pickle.loads((private / f"deployed_{stage}_{target}.pkl").read_bytes())
            k = len(entry["columns"])
            for name, spec in entry["learners"].items():
                if spec["kind"] == "stack":      # bases published once (export.share_stack_bases)
                    spec = {**spec, "bases": [entry["learners"][b["same_as"]] if "same_as" in b else b for b in spec["bases"]]}
                f = pk["models"][name]["fitted"]
                Xe = rng.normal(0, 1, (40, k))
                if spec["kind"] in ("lgbm", "xgb"):
                    thr = {}
                    for t in spec["trees"]:
                        for fj, th in zip(t["f"], t["t"]):
                            if fj >= 0:
                                thr.setdefault(fj, []).append(th)
                    for j in range(k):
                        vals = np.array(thr.get(j, [0.0]))
                        Xe[:, j] = rng.choice(np.concatenate([vals, vals + 0.5, vals - 0.5]), size=len(Xe))
                Xe[rng.random(Xe.shape) < 0.15] = np.nan
                js = _node_eval(spec, Xe, tmp_path)
                lg = f.logit(pd.DataFrame(Xe, columns=entry["columns"]))
                tol = 1e-4 if spec["kind"] in ("xgb", "stack") else 1e-5 if spec["kind"] == "ebm" else 1e-6
                assert np.max(np.abs(np.array([r["raw"] for r in js]) - lg)) < tol, (stage, target, name)
