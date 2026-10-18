"""One interface to the five learners.

    tune(name, X, y, ...)            tune on training patients (equal budget, same folds) and fit
    refit(name, settings, X, y, ...) the same settings on other patients (out-of-fold, bootstrap, deployment)
    Fitted.proba(X) / .logit(X)      predicted probability / log-odds (no recalibration)
    contributions(fitted, X)         exact per-column contributions on the log-odds scale, summing with
                                     the base value to the log-odds: TreeSHAP for the tree models, and
                                     coefficient x standardised value for the regressions (the exact
                                     Shapley values of a linear model); None for the MLP
    export(fitted, shift, X)         the numbers the browser needs (web/model.js, web/models_extra.js)

Learners: "Logistic regression (splines)" (the reference, as in the 2026 analysis), "Logistic
regression" (linear), "LightGBM", "XGBoost" and "MLP". The MLP is validated but not published: its
weights have no "at least 20 patients" guarantee like the tree leaves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import lgbm as LG
from .lr import export_logreg, fit_logreg, fit_logreg_fixed
from .splines import SplineLogReg, expand, export_spline_logreg, fit_spline_logreg, fit_spline_logreg_fixed
from .tuning import fit_mlp_fixed, fit_xgb_fixed, jsonable, lgbm_settings, tune_mlp, tune_xgb

REFERENCE = "Logistic regression (splines)"
ORDER = ["Logistic regression (splines)", "Logistic regression", "LightGBM", "XGBoost", "MLP"]
DEPLOYABLE = ("Logistic regression (splines)", "Logistic regression", "LightGBM", "XGBoost")
KIND = {"Logistic regression (splines)": "spline", "Logistic regression": "lr", "LightGBM": "lgbm",
        "XGBoost": "xgb", "MLP": "mlp"}


class Fitted:
    def __init__(self, name: str, model, settings: dict, columns: list[str]):
        self.name, self.model, self.settings, self.columns = name, model, jsonable(settings), list(columns)
        self.kind = KIND[name]

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.columns]
        if self.kind == "lgbm":
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]

    def logit(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "lgbm":
            return self.model.predict(X[self.columns], raw_score=True)
        p = np.clip(self.proba(X), 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))


def tune(name: str, X: pd.DataFrame, y, seed: int, trials: int, mono: list[int], min_leaf: int = 20,
         max_rounds: int = 3000, es: int = 100) -> Fitted:
    y = np.asarray(y, float)
    cols = list(X.columns)
    if name == "LightGBM":
        best, rounds, cv_ll = LG.tune(X, y, mono, seed, trials, max_rounds, es, min_leaf)
        model = LG.fit(X, y, LG.params(best, mono, seed, min_leaf), rounds)
        return Fitted(name, model, lgbm_settings(best, rounds, cv_ll, trials), cols)
    if name == "XGBoost":
        m = tune_xgb(X, y, seed, trials)
        return Fitted(name, m, m.settings, cols)
    if name == "MLP":
        m = tune_mlp(X, y, seed, trials)
        return Fitted(name, m, m.settings, cols)
    if name == "Logistic regression":
        m = fit_logreg(X, y, seed)
        return Fitted(name, m, m.settings, cols)
    if name == "Logistic regression (splines)":
        m = fit_spline_logreg(X, y, seed)
        return Fitted(name, m, m.settings, cols)
    raise KeyError(name)


def refit(name: str, settings: dict, X: pd.DataFrame, y, seed: int, mono: list[int], min_leaf: int = 20,
          cv_trees: bool = False) -> Fitted:
    """Fit with fixed settings. `cv_trees`: re-estimate the number of trees by 5-fold CV on X (used for
    the deployed models, which are fitted on more patients than the development cohort)."""
    y = np.asarray(y, float)
    cols = list(X.columns)
    s = dict(settings)
    if name == "LightGBM":
        prm = LG.params(s, mono, seed, min_leaf)
        rounds = LG.cv_rounds(X, y, prm, seed) if cv_trees else int(s["trees"])
        return Fitted(name, LG.fit(X, y, prm, rounds), {**s, "trees": int(rounds)}, cols)
    if name == "XGBoost":
        m = fit_xgb_fixed(X, y, s, seed, cv_trees=cv_trees)
        return Fitted(name, m, m.settings, cols)
    if name == "MLP":
        return Fitted(name, fit_mlp_fixed(X, y, s, seed), s, cols)
    if name == "Logistic regression":
        m = fit_logreg_fixed(X, y, float(s["C"]))
        m.settings = s
        return Fitted(name, m, s, cols)
    if name == "Logistic regression (splines)":
        m = fit_spline_logreg_fixed(X, y, int(s["knots"]), float(s["C"]))
        m.settings = s
        return Fitted(name, m, s, cols)
    raise KeyError(name)


def oof(name: str, settings: dict, X: pd.DataFrame, y, seed: int, mono: list[int], folds: int = 5,
        min_leaf: int = 20) -> np.ndarray:
    """Out-of-fold probabilities with fixed settings (one 5-fold split)."""
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y, float)
    p = np.zeros(len(y))
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        f = refit(name, settings, X.iloc[tr], y[tr], seed, mono, min_leaf)
        p[va] = f.proba(X.iloc[va])
    return p


# ------------------------------------------------------------------------------- explanations
def _linear_parts(pipe, Xexp: pd.DataFrame):
    """Per expanded-column contributions (coefficient x standardised value) and the intercept."""
    imp, sc, clf = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[2][1]
    A = Xexp.to_numpy(float)
    filled = np.where(np.isnan(A), imp.statistics_[None, :], A)
    parts = [filled]
    ind = [] if imp.indicator_ is None else [int(i) for i in imp.indicator_.features_]
    if ind:
        parts.append(np.isnan(A[:, ind]).astype(float))
    Z = (np.hstack(parts) - sc.mean_) / sc.scale_
    coef = np.ravel(clf.coef_)
    owner = list(range(A.shape[1])) + ind          # expanded column each coefficient belongs to
    return Z * coef[None, :], owner, float(np.ravel(clf.intercept_)[0])


def contributions(f: Fitted, X: pd.DataFrame):
    """(DataFrame of contributions per model column, base value) on the log-odds scale."""
    X = X[f.columns]
    if f.kind == "lgbm":
        c = f.model.predict(X, pred_contrib=True)
        return pd.DataFrame(c[:, :-1], columns=f.columns, index=X.index), float(np.mean(c[:, -1]))
    if f.kind == "xgb":
        import xgboost as xgb

        c = f.model.booster.predict(xgb.DMatrix(X, missing=np.nan), pred_contribs=True)
        return pd.DataFrame(c[:, :-1], columns=f.columns, index=X.index), float(np.mean(c[:, -1]))
    if f.kind == "lr":
        parts, owner, b0 = _linear_parts(f.model, X)
        out = np.zeros((len(X), len(f.columns)))
        for j, o in enumerate(owner):
            out[:, o] += parts[:, j]
        return pd.DataFrame(out, columns=f.columns, index=X.index), b0
    if f.kind == "spline":
        m: SplineLogReg = f.model
        Xe = expand(X, m.knots)
        parts, owner, b0 = _linear_parts(m.model, Xe)
        base_of = {c: (c.split("__rcs")[0] if "__rcs" in c else c) for c in Xe.columns}
        pos = {c: i for i, c in enumerate(f.columns)}
        out = np.zeros((len(X), len(f.columns)))
        for j, o in enumerate(owner):
            out[:, pos[base_of[Xe.columns[o]]]] += parts[:, j]
        return pd.DataFrame(out, columns=f.columns, index=X.index), b0
    return None


# ------------------------------------------------------------------------------------- export
def export(f: Fitted, shift: float, X_train: pd.DataFrame, min_leaf: int = 20) -> dict:
    """Browser specification; raises if a published tree leaf summarises fewer than `min_leaf` patients."""
    from .treejson import leaf_counts, lgbm_to_json, min_leaf_count, xgb_to_json

    if f.kind == "lgbm":
        mj = lgbm_to_json(f.model)
        if mj["feature_names"] != f.columns:
            raise ValueError("LightGBM column mismatch")
        ml = min_leaf_count(mj)
        if ml < min_leaf:
            raise ValueError(f"LightGBM: a leaf summarises only {ml} patients (< {min_leaf})")
        return {"kind": "lgbm", "sigmoid": mj["sigmoid"], "trees": mj["trees"], "min_leaf_count": int(ml),
                "intercept_recent": round(float(shift), 6)}
    if f.kind == "xgb":
        ml = leaf_counts(f.model.booster, X_train[f.columns])
        if ml < min_leaf:
            raise ValueError(f"XGBoost: a leaf summarises only {ml} patients (< {min_leaf})")
        xj = xgb_to_json(f.model.booster, f.columns, X_train[f.columns])
        return {"kind": "xgb", "op": "lt", "base_margin": xj["base_margin"], "trees": xj["trees"],
                "min_leaf_count": int(ml), "intercept_recent": round(float(shift), 6)}
    if f.kind == "lr":
        spec = export_logreg(f.model, f.columns, 0.0)
        return {"kind": "lr", "spec": spec, "intercept_recent": round(float(shift), 6)}
    if f.kind == "spline":
        spec = export_spline_logreg(f.model, 0.0)
        return {"kind": "spline", "spec": spec, "intercept_recent": round(float(shift), 6)}
    raise ValueError(f"{f.name} is not published")
