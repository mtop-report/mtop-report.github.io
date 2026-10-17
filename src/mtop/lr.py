"""Penalised logistic regression used as a comparator, and its browser export.

Pipeline: median imputation with missing-value indicators -> standardisation -> L2 logistic
regression. The penalty strength C, its only setting, is chosen from 50 values (1e-4 ... 1e2, log
grid) by 5-fold cross-validated log-loss on the same folds every learner is tuned on
(tuning.shared_folds). The browser reproduces the arithmetic from the exported numbers
(tests/test_extras_parity.py).
"""
from __future__ import annotations

import inspect

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = np.logspace(-4, 2, 50)


def cv_logloss(clf) -> float:
    """Best mean cross-validated log-loss of a fitted LogisticRegressionCV (either attribute layout)."""
    s = clf.scores_
    if isinstance(s, dict):
        s = next(iter(s.values()))
    return float(-np.max(np.asarray(s, float).mean(axis=0)))


def _pipeline(clf):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler(), clf)


def fit_logreg_fixed(X, y, C: float):
    """The same pipeline with a given penalty strength (no search): out-of-fold predictions for a
    setting already chosen, as for LightGBM's out-of-fold recalibration."""
    return _pipeline(LogisticRegression(C=C, max_iter=5000)).fit(X, y)


def fit_logreg(X, y, seed):
    from .tuning import shared_folds

    kw = dict(Cs=C_GRID, cv=shared_folds(y, seed), scoring="neg_log_loss", max_iter=5000, random_state=seed)
    accepted = inspect.signature(LogisticRegressionCV).parameters
    if "use_legacy_attributes" in accepted:      # scikit-learn >= 1.9: silence the transition warning
        kw.update(use_legacy_attributes=False, l1_ratios=(0.0,))
    model = _pipeline(LogisticRegressionCV(**kw))
    model.fit(X, y)
    clf = model.steps[2][1]
    model.settings = {"C": float(np.ravel(clf.C_)[0]), "cv_logloss": cv_logloss(clf), "candidates": len(C_GRID)}
    return model


def export_logreg(model, columns: list[str], shift: float = 0.0) -> dict:
    """Numbers needed to evaluate the fitted pipeline in the browser."""
    imp, sc, clf = model.steps[0][1], model.steps[1][1], model.steps[2][1]
    ind = [] if imp.indicator_ is None else [int(i) for i in imp.indicator_.features_]
    return {
        "columns": list(columns),
        "medians": [float(v) for v in imp.statistics_],
        "indicators": ind,
        "mean": [float(v) for v in sc.mean_],
        "scale": [float(v) for v in sc.scale_],
        "coef": [float(v) for v in np.ravel(clf.coef_)],
        "intercept": float(np.ravel(clf.intercept_)[0]),
        "shift": float(shift),
    }


def predict_exported(spec: dict, X: np.ndarray) -> np.ndarray:
    """Reference implementation of the browser arithmetic (used by tests)."""
    X = np.asarray(X, float)
    filled = np.where(np.isnan(X), np.asarray(spec["medians"])[None, :], X)
    parts = [filled]
    if spec["indicators"]:
        parts.append(np.isnan(X[:, spec["indicators"]]).astype(float))
    Z = (np.hstack(parts) - np.asarray(spec["mean"])) / np.asarray(spec["scale"])
    lin = Z @ np.asarray(spec["coef"]) + spec["intercept"] + spec.get("shift", 0.0)
    return 1 / (1 + np.exp(-lin))
