"""The bootstrap-optimism module repeats the published reference pipeline exactly."""
import numpy as np
import pandas as pd

from mtop import optimism
from mtop.splines import fit_spline_logreg


def _data(n=300, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"age": rng.normal(70, 12, n), "nihss_adm": rng.integers(0, 30, n).astype(float),
                      "ivt": rng.integers(0, 2, n).astype(float)})
    X.loc[rng.choice(n, 15, replace=False), "nihss_adm"] = np.nan     # a missing-value indicator
    lp = -0.04 * (X["age"] - 70) - 0.12 * (X["nihss_adm"].fillna(15) - 15) + 0.3 * X["ivt"]
    y = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(float)
    return X, y.to_numpy()


def test_published_folds_reproduce_the_reference_pipeline():
    X, y = _data()
    ref = fit_spline_logreg(X, y, seed=7)
    m, s, _ = optimism.fit_reference(X, y, seed=7)
    assert s["knots"] == ref.settings["knots"]
    assert np.isclose(s["C"], ref.settings["C"]) and np.isclose(s["cv_logloss"], ref.settings["cv_logloss"])
    assert np.allclose(m.predict_proba(X)[:, 1], ref.predict_proba(X)[:, 1], atol=1e-12)


def test_grouped_folds_keep_copies_of_a_patient_together():
    X, y = _data()
    idx = np.random.default_rng(3).integers(0, len(y), len(y))
    for train, val in optimism.grouped_folds(y[idx], idx, seed=7):
        assert not set(idx[train]) & set(idx[val])


def test_optimism_arithmetic():
    apparent = {k: 1.0 for k in optimism.MEASURES}
    recs = [{"optimism": {k: v for k in optimism.MEASURES}, "knots": 3, "C": 0.01, "C_at_grid_end": False,
             "convergence_warnings": 0, "unique_patients": 10} for v in (0.1, 0.3)]
    out = optimism.summarise(apparent, recs, np.zeros((2, 5), np.float32))
    for k in optimism.MEASURES:
        assert np.isclose(out["optimism"][k]["mean_optimism"], 0.2)
        assert np.isclose(out["optimism"][k]["corrected"], 0.8)
    assert out["stability"]["knots_share"]["3"] == 1.0
