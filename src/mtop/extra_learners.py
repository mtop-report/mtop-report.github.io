"""The learners added by amendment 1 (docs/amendment_1.md) that the calculator can publish.

  EBM                       interpret's ExplainableBoostingClassifier with at least 20 patients per leaf
                            (min_samples_leaf = 20) and every input treated as continuous (the design
                            matrix is already numeric and one-hot coded). Only the cut points where a
                            shape function changes are published.
  LR + interactions         the reference specification (restricted cubic splines, L2, knots and penalty
                            chosen by cross-validation) plus age x NIHSS on admission and onset-to-puncture
                            x age, as products of centred values (the 2026 report's two interactions).
  LR + multiple imputation  the reference specification with the reference's tuned knots and penalty, fitted
                            on 10 chained-equation imputations of the predictors stacked with weight 1/10
                            (outcome not used); a patient's missing values are imputed the same way and the
                            10 predictions averaged.
  Stacking                  a logistic meta-model (L2, C = 1) on the out-of-fold log-odds (the 5 shared folds)
                            of the reference, XGBoost (both with their tuned settings) and EBM (at least 20
                            patients per leaf, 4 outer bags as in the 2026 report).

Explanations are exact on the log-odds scale: EBM is additive by construction (pair terms are split half
and half between their two inputs, exact for purified pairs); an interaction term of centred inputs is
split half and half (exact for a product of centred, independent inputs); the stacking's attribution is
the meta-model's weighted sum of its three base models' attributions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lr import fit_logreg_fixed
from .splines import SplineLogReg, choose_knots, expand, export_spline_logreg, fit_spline_logreg

EBM_MIN_LEAF = 20
# amendment 1a: the published EBM keeps 10 pair terms on 32 bins (the default 3 x inputs pairs on 64 bins
# make every published model 1.5-2 MB); the default EBM is validated for comparison (compare.py)
EBM_PAIRS, EBM_PAIR_BINS = 10, 32
INTERACTIONS = (("age", "nihss_adm"), ("onset_to_puncture", "age"))
MI_IMPUTATIONS = 10
STACK_EBM_BAGS = 4


# ------------------------------------------------------------------------------------------ EBM
class EBMModel:
    """interpret's EBM on numpy arrays (no feature-name checks), with the scikit-learn interface."""

    def __init__(self, m):
        self.m = m

    def predict_proba(self, X) -> np.ndarray:
        return self.m.predict_proba(np.asarray(X, float))

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(self.m.decision_function(np.asarray(X, float)), float)


def fit_ebm(X: pd.DataFrame, y, seed: int, outer_bags: int = 14, threads: int = 2, publishable: bool = True):
    """The publishable EBM (at least 20 patients per leaf, 10 pairs on 32 bins), or with
    `publishable=False` interpret's defaults, as in the 2026 report (comparison only)."""
    from interpret.glassbox import ExplainableBoostingClassifier

    kw = dict(min_samples_leaf=EBM_MIN_LEAF, interactions=EBM_PAIRS, max_interaction_bins=EBM_PAIR_BINS) if publishable else {}
    m = ExplainableBoostingClassifier(feature_types=["continuous"] * X.shape[1], outer_bags=outer_bags,
                                      random_state=seed, n_jobs=threads, **kw)
    m.fit(X.to_numpy(float), np.asarray(y, int))
    w = EBMModel(m)
    w.settings = {"min_samples_leaf": int(m.min_samples_leaf), "outer_bags": int(outer_bags), "interactions": str(m.interactions),
                  "max_bins": int(m.max_bins), "max_interaction_bins": int(m.max_interaction_bins),
                  "terms": len(m.term_features_), "pairs": int(sum(len(f) == 2 for f in m.term_features_))}
    return w


def ebm_contributions(w: EBMModel, X: pd.DataFrame, columns: list[str]):
    """Per model column (pair terms split half and half) and the intercept."""
    m = w.m
    T = m.eval_terms(X.to_numpy(float))
    out = np.zeros((len(X), len(columns)))
    for t, feats in enumerate(m.term_features_):
        for f in feats:
            out[:, f] += T[:, t] / len(feats)
    return pd.DataFrame(out, columns=columns, index=X.index), float(np.ravel(m.intercept_)[0])


def _merge_axis(cuts: np.ndarray, block: np.ndarray, axis: int):
    """Drop the cut points where the scores on both sides are identical (never used by a split).
    `block` holds the bin scores along `axis` (missing and unknown bins excluded)."""
    keep_idx = [0]
    for k in range(1, block.shape[axis]):
        a = np.take(block, k, axis=axis)
        b = np.take(block, keep_idx[-1], axis=axis)
        if not np.array_equal(a, b):
            keep_idx.append(k)
    new_cuts = [float(cuts[k - 1]) for k in keep_idx[1:]]
    return new_cuts, np.take(block, keep_idx, axis=axis)


def export_ebm(w: EBMModel, columns: list[str], shift: float = 0.0, digits: int = 7) -> dict:
    """Terms with only the effective cut points. Main term: {"f": [i], "cuts": [[...]], "miss": score of a
    missing value, "s": [scores per interval]}; pair term: {"f": [i, j], "cuts": [[...], [...]],
    "s": 2-D scores with row/column 0 for a missing value}."""
    m = w.m
    rnd = lambda a: np.round(np.asarray(a, float), digits).tolist()  # noqa: E731
    terms = []
    for feats, sc in zip(m.term_features_, m.term_scores_):
        sc = np.asarray(sc, float)
        if len(feats) == 1:
            i = feats[0]
            cuts = np.asarray(m.bins_[i][0], float)
            body = sc[1:len(cuts) + 2]
            c, s = _merge_axis(cuts, body, 0)
            if len(s) == 1 and float(s[0]) == 0.0 and float(sc[0]) == 0.0:
                continue
            terms.append({"f": [int(i)], "cuts": [c], "miss": float(np.round(sc[0], digits)), "s": rnd(s)})
        else:
            i, j = feats
            ci = np.asarray(m.bins_[i][min(1, len(m.bins_[i]) - 1)], float)
            cj = np.asarray(m.bins_[j][min(1, len(m.bins_[j]) - 1)], float)
            grid = sc[:len(ci) + 2, :len(cj) + 2]            # row/col 0 = missing; drop the unknown bin
            body = grid[1:, :]
            c_i, merged = _merge_axis(ci, body, 0)
            grid2 = np.vstack([grid[:1, :], merged])
            body = grid2[:, 1:]
            c_j, merged = _merge_axis(cj, body, 1)
            grid3 = np.hstack([grid2[:, :1], merged])
            if not np.any(grid3):
                continue
            terms.append({"f": [int(i), int(j)], "cuts": [c_i, c_j], "s": rnd(grid3)})
    return {"columns": list(columns), "intercept": float(np.ravel(m.intercept_)[0]), "terms": terms,
            "shift": float(shift), "min_samples_leaf": int(m.min_samples_leaf)}


def _bin(cuts, v) -> int:
    """Interval of v among the cut points: 1..len(cuts)+1, 0 for a missing value (interpret's rule:
    a value equal to a cut point belongs to the upper interval)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0
    return int(np.searchsorted(np.asarray(cuts, float), v, side="right")) + 1


def ebm_spec_terms(spec: dict, x) -> tuple[list[float], list[list[int]]]:
    """Reference arithmetic of the browser (tests): each term's score for one row."""
    out, owners = [], []
    for t in spec["terms"]:
        if len(t["f"]) == 1:
            k = _bin(t["cuts"][0], x[t["f"][0]])
            out.append(t["miss"] if k == 0 else t["s"][k - 1])
        else:
            a, b = (_bin(t["cuts"][d], x[t["f"][d]]) for d in (0, 1))
            out.append(t["s"][a][b])
        owners.append(t["f"])
    return out, owners


def ebm_spec_logit(spec: dict, X: np.ndarray) -> np.ndarray:
    return np.array([spec["intercept"] + sum(ebm_spec_terms(spec, row)[0]) for row in np.asarray(X, float)])


# ---------------------------------------------------------------------------- LR + interactions
class InteractionSplineLogReg(SplineLogReg):
    """Spline regression plus products of centred inputs (centres = training means)."""

    def __init__(self, knots: dict, model, columns: list[str], products: list[tuple[str, str, float, float]]):
        super().__init__(knots, model, columns)
        self.products = products

    def expanded(self, X: pd.DataFrame) -> pd.DataFrame:
        Xe = expand(X[self.columns], self.knots)
        for a, b, ca, cb in self.products:
            Xe[f"{a}__x__{b}"] = (X[a].astype(float) - ca) * (X[b].astype(float) - cb)
        return Xe

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.expanded(X))


def _products(X: pd.DataFrame) -> list[tuple[str, str, float, float]]:
    return [(a, b, float(X[a].mean()), float(X[b].mean())) for a, b in INTERACTIONS if a in X.columns and b in X.columns]


def fit_interaction_spline(X: pd.DataFrame, y, seed: int, k: int | None = None, C: float | None = None):
    """Knots (3, 4 or 5) and penalty chosen by cross-validated log-loss as for the reference, unless given."""
    from .lr import fit_logreg

    prods = _products(X)
    best = None
    for kk in ([k] if k else (3, 4, 5)):
        knots = choose_knots(X, k=kk)
        probe = InteractionSplineLogReg(knots, None, list(X.columns), prods)
        Xe = probe.expanded(X)
        model = fit_logreg_fixed(Xe, y, C) if C else fit_logreg(Xe, y, seed)
        ll = model.settings["cv_logloss"] if not C else 0.0
        if best is None or ll < best[0]:
            best = (ll, kk, knots, model)
    _, kk, knots, model = best
    out = InteractionSplineLogReg(knots, model, list(X.columns), prods)
    out.settings = {"knots": kk, "C": float(C) if C else model.settings["C"],
                    "interactions": [f"{a} x {b}" for a, b, _, _ in prods],
                    **({"cv_logloss": model.settings["cv_logloss"]} if not C else {})}
    return out


def export_interaction_spline(m: InteractionSplineLogReg, shift: float = 0.0) -> dict:
    spec = export_spline_logreg(m, shift)
    spec["columns"] = list(m.expanded(pd.DataFrame(columns=m.columns, dtype=float)).columns)
    spec["products"] = [[m.columns.index(a), m.columns.index(b), ca, cb] for a, b, ca, cb in m.products]
    return spec


# ------------------------------------------------------------------------ LR + multiple imputation
class MISplineLogReg:
    """One spline regression fitted on stacked imputations; predictions averaged over imputations of
    the patient's missing values."""

    def __init__(self, inner: SplineLogReg, imputers: list, columns: list[str]):
        self.inner, self.imputers, self.columns = inner, imputers, columns

    def _imputed(self, X: pd.DataFrame):
        A = X[self.columns].to_numpy(float, copy=True)      # writable (pandas copy-on-write)
        if not np.isnan(A).any():
            return [X[self.columns]]
        return [pd.DataFrame(imp.transform(A.copy()), columns=self.columns, index=X.index) for imp in self.imputers]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ps = [self.inner.predict_proba(Z)[:, 1] for Z in self._imputed(X)]
        p = np.mean(ps, axis=0)
        return np.column_stack([1 - p, p])


def fit_mi_spline(X: pd.DataFrame, y, seed: int, k: int, C: float, m: int = MI_IMPUTATIONS) -> MISplineLogReg:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y, float)
    cols = list(X.columns)
    A = X.to_numpy(float, copy=True)                 # writable (pandas copy-on-write)
    knots = choose_knots(X, k=k)                 # on the observed values, as the reference
    imps, stacks = [], []
    for j in range(m):
        imp = IterativeImputer(sample_posterior=True, max_iter=10, random_state=seed + j, keep_empty_features=True)
        Z = pd.DataFrame(imp.fit_transform(A.copy()), columns=cols)
        imps.append(imp)
        stacks.append(expand(Z, knots))
    Xs = pd.concat(stacks, ignore_index=True)
    ys = np.tile(y, m)
    w = np.full(len(ys), 1.0 / m)
    pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000))
    pipe.fit(Xs, ys, logisticregression__sample_weight=w)
    inner = SplineLogReg(knots, pipe, cols)
    out = MISplineLogReg(inner, imps, cols)
    out.settings = {"knots": int(k), "C": float(C), "imputations": int(m)}
    return out


# ----------------------------------------------------------------------------------- stacking
class Stack:
    def __init__(self, bases: list, meta, names: list[str]):
        self.bases, self.meta, self.names = bases, meta, names

    def base_logits(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([b.logit(X) for b in self.bases])

    def logit(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_logits(X) @ np.ravel(self.meta.coef_) + float(np.ravel(self.meta.intercept_)[0])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = 1 / (1 + np.exp(-self.logit(X)))
        return np.column_stack([1 - p, p])
