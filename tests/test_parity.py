"""The browser model (web/model.js) must reproduce LightGBM exactly.

Checks raw scores and TreeSHAP attributions against `predict(raw_score=True)` and
`predict(pred_contrib=True)` on rows with every missing-value pattern we use.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from mtop.export import booster_to_json

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
        "nihss": rng.integers(0, 40, n).astype(float),
        "aspects": rng.integers(0, 11, n).astype(float),
        "time": rng.gamma(3, 60, n).round(),
        "flag": rng.integers(0, 2, n).astype(float),
        "zeroish": np.where(rng.random(n) < 0.3, 0.0, rng.normal(5, 2, n)),
        "cat__a": rng.integers(0, 2, n).astype(float),
    })
    for col, frac in [("aspects", 0.3), ("time", 0.1), ("flag", 0.05), ("cat__a", 0.05)]:
        X.loc[rng.random(n) < frac, col] = np.nan
    z = 2 - 0.04 * (X["age"] - 70) - 0.12 * X["nihss"] + 0.2 * X["aspects"].fillna(7) - 0.004 * X["time"].fillna(200)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    return X, y


@pytest.mark.parametrize("use_missing", [True, False])
def test_js_matches_lightgbm(tmp_path, use_missing):
    X, y = _synthetic()
    params = dict(objective="binary", num_leaves=9, max_depth=4, min_child_samples=20, learning_rate=0.05,
                  bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, verbosity=-1, seed=3,
                  deterministic=True, force_row_wise=True, use_missing=use_missing, zero_as_missing=False,
                  monotone_constraints=[-1, -1, 1, -1, 0, 0, 0], monotone_constraints_method="advanced")
    booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=120)
    mj = booster_to_json(booster)
    spec = {"columns": list(X.columns), "column_feature": list(X.columns), "trees": mj["trees"],
            "sigmoid": mj["sigmoid"], "intercept_recent": 0.0}

    # evaluation rows: training rows plus all-missing and edge rows
    Xe = X.sample(250, random_state=1).to_numpy(float)
    edge = np.array([[np.nan] * X.shape[1], [0.0] * X.shape[1], [120, 42, 10, 1440, 1, 0, 1]], float)
    Xe = np.vstack([Xe, edge])

    js = _node_eval(spec, Xe, tmp_path)
    raw_py = booster.predict(Xe, raw_score=True)
    contrib_py = booster.predict(Xe, pred_contrib=True)
    raw_js = np.array([r["raw"] for r in js])
    phi_js = np.array([r["phi"] for r in js])
    base_js = np.array([r["base"] for r in js])

    assert np.max(np.abs(raw_js - raw_py)) < 1e-6
    assert np.max(np.abs(phi_js - contrib_py[:, :-1])) < 1e-6
    assert np.max(np.abs(base_js - contrib_py[:, -1])) < 1e-6
    # additivity: base + sum(phi) == raw
    assert np.max(np.abs(base_js + phi_js.sum(1) - raw_js)) < 1e-6


def test_exported_bundle_matches_private_models(tmp_path):
    """If the real pipeline has been run, check the published bundle against the saved boosters."""
    bundle_file = ROOT / "artifacts" / "public" / "model_bundle.json"
    private = ROOT / "artifacts" / "private"
    if not bundle_file.exists() or not any(private.glob("lgbm_*.txt")):
        pytest.skip("pipeline outputs not present")
    bundle = json.loads(bundle_file.read_text())
    rng = np.random.default_rng(0)
    for stage, per_target in bundle["models"].items():
        for target, spec in per_target.items():
            mf = private / f"lgbm_{stage}_{target}.txt"
            if not mf.exists():
                continue
            booster = lgb.Booster(model_file=str(mf))
            k = len(spec["columns"])
            Xe = rng.normal(0, 1, (60, k))
            # plausible magnitudes: reuse thresholds from the trees for each column
            thr = {}
            for t in spec["trees"]:
                for f, th in zip(t["f"], t["t"]):
                    if f >= 0:
                        thr.setdefault(f, []).append(th)
            for j in range(k):
                vals = np.array(thr.get(j, [0.0]))
                Xe[:, j] = rng.choice(np.concatenate([vals, vals + 0.5, vals - 0.5]), size=len(Xe))
            Xe[rng.random(Xe.shape) < 0.15] = np.nan
            js = _node_eval(spec, Xe, tmp_path)
            raw_py = booster.predict(Xe, raw_score=True)
            contrib_py = booster.predict(Xe, pred_contrib=True)
            assert np.max(np.abs(np.array([r["raw"] for r in js]) - raw_py)) < 1e-6, (stage, target)
            assert np.max(np.abs(np.array([r["phi"] for r in js]) - contrib_py[:, :-1])) < 1e-6, (stage, target)
