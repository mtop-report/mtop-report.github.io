"""Version 1.1 reference model (src/mtop/reference.py).

- with every change off, the new preprocessing equals the v1.0.0 scikit-learn pipeline exactly, and the
  Newton solver reaches the optimum that L-BFGS approaches;
- changes 2.1-2.4 do what the plan says (one indicator per input, only from 20 missing, derived inputs
  out, a small occlusion category joins its parent);
- the browser (web/model.js) reproduces the exported model, its log-odds and attributions.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtop import reference as R

HERE = Path(__file__).parent
NODE = shutil.which("node")


def _data(n=900, seed=3):
    rng = np.random.default_rng(seed)
    occ = rng.choice(["M1", "basilar", "basilar_vertebral"], size=n, p=[0.8, 0.185, 0.015])
    X = pd.DataFrame({
        "age": rng.normal(70, 12, n).round(),
        "nihss_adm": rng.integers(0, 40, n).astype(float),
        "occlusion__M1": (occ == "M1").astype(float),
        "occlusion__basilar": (occ == "basilar").astype(float),
        "occlusion__basilar_vertebral": (occ == "basilar_vertebral").astype(float),
        "glucose": rng.normal(7.5, 2.5, n).round(1).clip(3, 25),
        "sbp": rng.normal(150, 25, n).round(),
        "dbp": rng.normal(80, 12, n).round(),
        "flag": rng.integers(0, 2, n).astype(float),
    })
    X["pulse_pressure"] = X["sbp"] - X["dbp"]
    occ_missing = rng.random(n) < 0.05
    X.loc[occ_missing, ["occlusion__M1", "occlusion__basilar", "occlusion__basilar_vertebral"]] = np.nan
    X.loc[rng.random(n) < 0.2, "glucose"] = np.nan
    X.loc[rng.random(n) < 0.01, "flag"] = np.nan            # fewer than 20 missing: no indicator under 2.2
    sb = rng.random(n) < 0.03
    X.loc[sb, ["sbp", "pulse_pressure"]] = np.nan
    z = 1.5 - 0.04 * (X["age"] - 70) - 0.1 * X["nihss_adm"] + 0.5 * X["occlusion__basilar"].fillna(0)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    return X, y


def test_v100_layout_equals_sklearn_pipeline():
    from sklearn.linear_model import LogisticRegression

    from mtop.lr import _pipeline
    from mtop.splines import choose_knots, expand
    X, y = _data()
    for k in (3, 4, 5):
        lay = R.fit_layout(X, R.V100_NEWTON, k)
        E = expand(X, choose_knots(X, k=k))
        pipe = _pipeline(LogisticRegression(C=0.05, max_iter=5000, tol=1e-12)).fit(E, y)
        imp, sc, clf = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[2][1]
        assert lay.columns == list(E.columns)
        assert lay.ind == [int(i) for i in imp.indicator_.features_]
        np.testing.assert_array_equal(lay.medians, imp.statistics_)
        np.testing.assert_allclose(lay.Z(X), sc.transform(imp.transform(E)), atol=1e-12)
        beta = R.newton(R._with_ones(lay.Z(X)), y, np.ones(len(y)), 0.05)
        np.testing.assert_allclose(beta[:-1], clf.coef_.ravel(), atol=1e-5)


def test_changes_2_1_to_2_4():
    X, y = _data()
    lay0 = R.fit_layout(X, R.V100_NEWTON, 3)
    lay1 = R.fit_layout(X, R.SIMPLIFIED, 3)
    # 2.3: pulse pressure out; 2.4: basilar + vertebral (about 14 patients) joins basilar
    assert "pulse_pressure" not in lay1.base and "occlusion__basilar_vertebral" not in lay1.base
    j = lay1.base.index("occlusion__basilar")
    assert sorted(lay1.sources[j]) == sorted([X.columns.get_loc("occlusion__basilar"), X.columns.get_loc("occlusion__basilar_vertebral")])
    # 2.1 + 2.2: one indicator per input, only for inputs missing in >= 20 patients
    ind_inputs = [R.input_of(lay1.columns[i]) for i in lay1.ind]
    assert len(ind_inputs) == len(set(ind_inputs))
    assert set(ind_inputs) == {"occlusion", "glucose", "sbp"}
    # v1.0.0: an indicator for every column with a missing value (3 identical ones for occlusion)
    assert sum(R.input_of(lay0.columns[i]) == "occlusion" for i in lay0.ind) == 3
    assert any(R.input_of(lay0.columns[i]) == "flag" for i in lay0.ind)


def test_repeat_one_is_the_v100_split():
    from mtop.tuning import shared_folds
    _, y = _data()
    f5 = R.make_folds(y, 11, 5)
    assert len(f5) == 25
    for (a, b), (c, d) in zip(f5[:5], shared_folds(y, 11)):
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)


def test_bootstrap_copies_equal_weights():
    """A patient's copies in a bootstrap sample act as one weighted patient (same optimum)."""
    X, y = _data(400)
    idx = np.random.default_rng(1).integers(0, len(y), len(y))
    Xb, yb = X.iloc[idx].reset_index(drop=True), y[idx]
    m = R.fit(Xb, yb, 5, R.SIMPLIFIED, groups=idx, settings={"knots": 3, "C": 0.05})
    lay = m.layout
    beta = R.newton(R._with_ones(lay.Z(Xb)), yb, np.ones(len(yb)), 0.05)
    np.testing.assert_allclose(np.r_[m.coef, m.intercept], beta, atol=1e-8)


def test_slope_correction_keeps_calibration_in_the_large():
    X, y = _data()
    m = R.fit(X, y, 7, R.SIMPLIFIED, settings={"knots": 3, "C": 0.05})
    m2 = R.with_slope(m, X, y, 0.8)
    np.testing.assert_allclose(m2.coef, 0.8 * m.coef)
    assert abs(m2.predict_proba(X)[:, 1].mean() - y.mean()) < 1e-9


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_parity(tmp_path):
    X, y = _data()
    m = R.fit(X, y, 7, R.V110.replace(repeats=2))
    spec = {"kind": "spline", "spec": m.export(0.0), "intercept_recent": 0.0}
    assert "from" in spec["spec"]
    rows = X.sample(200, random_state=2).to_numpy(float)
    rows = np.vstack([rows, np.full(X.shape[1], np.nan)])
    payload = [[None if np.isnan(v) else float(v) for v in r] for r in rows]
    inp, out = tmp_path / "in.json", tmp_path / "out.json"
    inp.write_text(json.dumps({"spec": spec, "rows": payload}))
    subprocess.run([NODE, str(HERE / "node_eval.js"), str(inp), str(out)], check=True)
    res = json.loads(out.read_text())
    Xr = pd.DataFrame(rows, columns=X.columns)
    lg = m.logit(Xr)
    contrib, base = m.contributions(Xr)
    for i, r in enumerate(res):
        assert abs(r["raw"] - lg[i]) < 1e-9
        assert abs(r["base"] - base) < 1e-9
        np.testing.assert_allclose(r["phi"], contrib.iloc[i].to_numpy(), atol=1e-9)
