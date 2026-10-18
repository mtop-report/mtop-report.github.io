"""Penalised logistic regression with restricted cubic splines (Harrell) for the continuous
predictors, as a fair "regression" comparator: the linear-only regression in `lr.py` cannot bend,
which handicaps it against the tree models on age, NIHSS, time and glucose.

Each spline variable x with knots t1 < ... < tk at Harrell's recommended percentiles of the
training data (k = 3: 10/50/90; k = 4: 5/35/65/95; k = 5: 5/27.5/50/72.5/95; k chosen by
cross-validated log-loss) is represented by x and k - 2 non-linear terms

    s_j(x) = [(x - t_j)+^3 - (x - t_{k-1})+^3 (t_k - t_j) / (t_k - t_{k-1})
              + (x - t_k)+^3 (t_{k-1} - t_j) / (t_k - t_{k-1})] / (t_k - t_1)^2 ,

which are linear beyond the outer knots. The expanded matrix goes through the same pipeline as
`lr.fit_logreg` (median imputation with missing indicators, scaling, L2 with C chosen by
cross-validated log-loss). The browser can reproduce the arithmetic from `export_spline_logreg`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lr import export_logreg, fit_logreg, fit_logreg_fixed

SPLINE_VARS = ("age", "nihss_adm", "onset_to_puncture", "glucose", "procedure_min", "nihss_24h", "nihss_7d")
KNOT_QUANTILES = {3: (0.10, 0.50, 0.90), 4: (0.05, 0.35, 0.65, 0.95), 5: (0.05, 0.275, 0.50, 0.725, 0.95)}
KNOT_OPTIONS = (3, 4, 5)


def rcs_terms(x: np.ndarray, knots) -> np.ndarray:
    """Non-linear restricted-cubic-spline terms (n x (k-2)); NaN stays NaN."""
    x = np.asarray(x, float)
    t = np.asarray(knots, float)
    k = len(t)
    p3 = lambda v: np.where(np.isnan(v), np.nan, np.maximum(v, 0.0) ** 3)  # noqa: E731
    out = np.empty((len(x), k - 2))
    for j in range(k - 2):
        out[:, j] = (p3(x - t[j]) - p3(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                     + p3(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])) / (t[k - 1] - t[0]) ** 2
    return out


def choose_knots(X: pd.DataFrame, variables=SPLINE_VARS, k: int = 4) -> dict:
    knots = {}
    qs = KNOT_QUANTILES[k]
    for v in variables:
        if v not in X.columns:
            continue
        x = X[v].dropna().to_numpy(float)
        if len(x) < 50:
            continue
        q = np.unique(np.quantile(x, qs))
        if len(q) == len(qs):
            knots[v] = [float(a) for a in q]
    return knots


def expand(X: pd.DataFrame, knots: dict) -> pd.DataFrame:
    """Original columns plus the spline terms of each knotted variable."""
    parts = [X]
    for v, t in knots.items():
        terms = rcs_terms(X[v].to_numpy(float), t)
        parts.append(pd.DataFrame(terms, index=X.index, columns=[f"{v}__rcs{j + 1}" for j in range(terms.shape[1])]))
    return pd.concat(parts, axis=1)


class SplineLogReg:
    def __init__(self, knots: dict, model, columns: list[str]):
        self.knots, self.model, self.columns = knots, model, columns

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(expand(X[self.columns], self.knots))


def fit_spline_logreg(X: pd.DataFrame, y, seed: int, knot_options=KNOT_OPTIONS, Cs=None) -> SplineLogReg:
    """For each number of knots, the penalty is tuned over 50 values (or `Cs`); the pair with the
    lowest cross-validated log-loss (same folds for every pair) is kept."""
    best = None
    for k in knot_options:
        knots = choose_knots(X, k=k)
        model = fit_logreg(expand(X, knots), y, seed, Cs=Cs)
        if best is None or model.settings["cv_logloss"] < best[2].settings["cv_logloss"]:
            best = (k, knots, model)
    k, knots, model = best
    out = SplineLogReg(knots, model, list(X.columns))
    out.settings = {"knots": k, **model.settings}
    return out


def fit_spline_logreg_fixed(X: pd.DataFrame, y, k: int, C: float) -> SplineLogReg:
    """A chosen setting (number of knots, penalty) refitted without search; knots are placed on X."""
    knots = choose_knots(X, k=k)
    return SplineLogReg(knots, fit_logreg_fixed(expand(X, knots), y, C), list(X.columns))


def export_spline_logreg(m: SplineLogReg, shift: float = 0.0) -> dict:
    spec = export_logreg(m.model, list(expand(pd.DataFrame(columns=m.columns, dtype=float), m.knots).columns), shift)
    spec["base_columns"] = m.columns
    spec["knots"] = m.knots
    return spec
