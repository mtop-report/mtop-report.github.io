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
from .tuning import fit_mlp_fixed, fit_xgb_fixed, jsonable, lgbm_settings, shared_folds, threads, tune_mlp, tune_xgb

REFERENCE = "Logistic regression (splines)"
ORDER = ["Logistic regression (splines)", "Logistic regression", "LightGBM", "XGBoost", "MLP"]
# amendment 1 (docs/amendment_1.md): four more publishable candidates (extra_learners.py)
NEW = ["EBM", "LR + interactions", "LR + multiple imputation", "Stacking"]
DEPLOYABLE = ("Logistic regression (splines)", "Logistic regression", "LightGBM", "XGBoost") + tuple(NEW)
KIND = {"Logistic regression (splines)": "spline", "Logistic regression": "lr", "LightGBM": "lgbm",
        "XGBoost": "xgb", "MLP": "mlp", "EBM": "ebm", "LR + interactions": "spline_int",
        "LR + multiple imputation": "spline_mi", "Stacking": "stack"}


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
        if self.kind == "ebm":
            return self.model.decision_function(X[self.columns])
        if self.kind == "stack":
            return self.model.logit(X[self.columns])
        p = np.clip(self.proba(X), 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))


def _new_tune(name: str, X: pd.DataFrame, y, seed: int, trials: int, mono, min_leaf: int,
              base_settings: dict | None) -> Fitted:
    """The amendment-1 learners. `base_settings`: tuned settings of the reference and XGBoost on these
    patients (from train.py), reused instead of tuning them again."""
    from . import extra_learners as XL
    from .splines import fit_spline_logreg

    cols = list(X.columns)
    base_settings = base_settings or {}

    def ref_settings():
        return base_settings.get(REFERENCE) or fit_spline_logreg(X, y, seed).settings

    if name == "EBM":
        m = XL.fit_ebm(X, y, seed, threads=threads())
        return Fitted(name, m, m.settings, cols)
    if name == "LR + interactions":
        m = XL.fit_interaction_spline(X, y, seed)
        return Fitted(name, m, m.settings, cols)
    if name == "LR + multiple imputation":
        s = ref_settings()
        m = XL.fit_mi_spline(X, y, seed, int(s["knots"]), float(s["C"]))
        return Fitted(name, m, m.settings, cols)
    if name == "Stacking":
        s_ref = ref_settings()
        s_xgb = base_settings.get("XGBoost") or tune_xgb(X, y, seed, trials).settings
        settings = {"reference": dict(s_ref), "xgboost": dict(s_xgb),
                    "ebm": {"min_samples_leaf": XL.EBM_MIN_LEAF, "outer_bags": XL.STACK_EBM_BAGS}}
        return _fit_stack(X, y, seed, settings, mono, min_leaf, cv_trees=False)
    raise KeyError(name)


def _stack_bases(X: pd.DataFrame, y, seed: int, settings: dict, mono, min_leaf: int, cv_trees: bool) -> list:
    from . import extra_learners as XL

    spl = refit(REFERENCE, settings["reference"], X, y, seed, mono, min_leaf)
    xgb = refit("XGBoost", settings["xgboost"], X, y, seed, mono, min_leaf, cv_trees=cv_trees)
    e = XL.fit_ebm(X, y, seed, outer_bags=int(settings["ebm"]["outer_bags"]), threads=threads())
    return [spl, xgb, Fitted("EBM", e, e.settings, list(X.columns))]


def _fit_stack(X: pd.DataFrame, y, seed: int, settings: dict, mono, min_leaf: int, cv_trees: bool) -> Fitted:
    """Meta-model on the out-of-fold log-odds of the three base models (the shared folds), then the
    base models refitted on all of X."""
    from sklearn.linear_model import LogisticRegression

    from . import extra_learners as XL

    y = np.asarray(y, float)
    oof = np.zeros((len(y), 3))
    for tr, va in shared_folds(y, seed):
        bases = _stack_bases(X.iloc[tr], y[tr], seed, settings, mono, min_leaf, cv_trees=False)
        oof[va] = np.column_stack([b.logit(X.iloc[va]) for b in bases])
    meta = LogisticRegression(C=1.0, max_iter=1000).fit(oof, y)
    bases = _stack_bases(X, y, seed, settings, mono, min_leaf, cv_trees)
    settings = {**settings, "xgboost": dict(bases[1].settings),
                "meta": {"coef": [round(float(c), 6) for c in np.ravel(meta.coef_)],
                         "intercept": round(float(np.ravel(meta.intercept_)[0]), 6)}}
    return Fitted("Stacking", XL.Stack(bases, meta, [REFERENCE, "XGBoost", "EBM"]), settings, list(X.columns))


def tune(name: str, X: pd.DataFrame, y, seed: int, trials: int, mono: list[int], min_leaf: int = 20,
         max_rounds: int = 3000, es: int = 100, base_settings: dict | None = None) -> Fitted:
    y = np.asarray(y, float)
    cols = list(X.columns)
    if name in NEW:
        return _new_tune(name, X, y, seed, trials, mono, min_leaf, base_settings)
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
    if name in NEW:
        from . import extra_learners as XL

        if name == "EBM":
            m = XL.fit_ebm(X, y, seed, outer_bags=int(s.get("outer_bags", 14)), threads=threads())
            return Fitted(name, m, {**s, **m.settings}, cols)
        if name == "LR + interactions":
            m = XL.fit_interaction_spline(X, y, seed, k=int(s["knots"]), C=float(s["C"]))
            return Fitted(name, m, {**s, **m.settings}, cols)
        if name == "LR + multiple imputation":
            m = XL.fit_mi_spline(X, y, seed, int(s["knots"]), float(s["C"]), int(s.get("imputations", XL.MI_IMPUTATIONS)))
            return Fitted(name, m, {**s, **m.settings}, cols)
        if name == "Stacking":
            return _fit_stack(X, y, seed, {k: s[k] for k in ("reference", "xgboost", "ebm")}, mono, min_leaf, cv_trees)
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
    if f.kind in ("spline", "spline_int"):
        return _spline_contributions(f.model, X, f.columns)
    if f.kind == "spline_mi":
        # averaged over the imputations of the patients' missing values
        parts = [_spline_contributions(f.model.inner, Z, f.columns) for Z in f.model._imputed(X)]
        return sum(p[0] for p in parts) / len(parts), parts[0][1]
    if f.kind == "ebm":
        from .extra_learners import ebm_contributions

        return ebm_contributions(f.model, X, f.columns)
    if f.kind == "stack":
        st = f.model
        w = np.ravel(st.meta.coef_)
        total, base = None, float(np.ravel(st.meta.intercept_)[0])
        for wk, b in zip(w, st.bases):
            c, b0 = contributions(b, X)
            total = c * wk if total is None else total + c * wk
            base += wk * b0
        return total, base
    return None


def _spline_contributions(m: SplineLogReg, X: pd.DataFrame, columns: list[str]):
    """Spline terms belong to their variable; a product of two centred inputs is split half and half."""
    Xe = m.expanded(X) if hasattr(m, "expanded") else expand(X[m.columns], m.knots)
    parts, owner, b0 = _linear_parts(m.model, Xe)
    pos = {c: i for i, c in enumerate(columns)}
    out = np.zeros((len(X), len(columns)))
    for j, o in enumerate(owner):
        name = Xe.columns[o]
        if "__x__" in name:
            a, b = name.split("__x__")
            out[:, pos[a]] += parts[:, j] / 2
            out[:, pos[b]] += parts[:, j] / 2
        else:
            out[:, pos[name.split("__rcs")[0] if "__rcs" in name else name]] += parts[:, j]
    return pd.DataFrame(out, columns=columns, index=X.index), b0


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
    if f.kind == "spline_int":
        from .extra_learners import export_interaction_spline

        return {"kind": "spline", "spec": export_interaction_spline(f.model, 0.0), "intercept_recent": round(float(shift), 6)}
    if f.kind == "spline_mi":
        return {"kind": "spline", "spec": export_spline_logreg(f.model.inner, 0.0), "intercept_recent": round(float(shift), 6)}
    if f.kind == "ebm":
        from .extra_learners import export_ebm

        if int(f.model.m.min_samples_leaf) < min_leaf:
            raise ValueError(f"EBM: leaves may summarise fewer than {min_leaf} patients")
        return {"kind": "ebm", "spec": export_ebm(f.model, f.columns, 0.0), "intercept_recent": round(float(shift), 6)}
    if f.kind == "stack":
        st = f.model
        return {"kind": "stack", "bases": [export(b, 0.0, X_train, min_leaf) for b in st.bases], "base_names": st.names,
                "meta": {"coef": [float(c) for c in np.ravel(st.meta.coef_)], "intercept": float(np.ravel(st.meta.intercept_)[0])},
                "intercept_recent": round(float(shift), 6)}
    raise ValueError(f"{f.name} is not published")
