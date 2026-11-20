"""The reference model of MTOP in the specification of version 1.0.0 and of version 1.1.

Penalised logistic regression with restricted cubic splines (splines.py), median imputation with
missing-value indicators, standardisation and an L2 penalty. Plan: docs/registrations/v1_1_plan.md
(with amendment 1). Each version 1.1 change is a switch of `Spec`, so every step can be evaluated on
its own and the code with every switch off reproduces v1.0.0 (plan, section 5, step 2):

  2.1 per_input_indicator  one missing-value indicator per input, not one per model column
  2.2 min_missing          an indicator only when at least this many fitting patients lack the
                           input; below that a missing value is set to the median, with no indicator
  2.3 drop_derived         inputs that are exact functions of other inputs are left out
  2.4 min_category         an occlusion category with fewer fitting patients joins its clinical
                           parent (clean.OCCLUSION_PARENT); a category without a parent keeps its term
  2.5 repeats              knots and penalty chosen by 5-fold cross-validation repeated `repeats`
                           times (mean log-loss); repeat 1 uses the folds of v1.0.0. The slope
                           correction of 2.5 is applied after the bootstrap (`with_slope`).

"Fitting patients" are the patients the model is fitted on (in cross-validation and the bootstrap:
the training part), so every rule is part of the pipeline that the internal validation repeats.

Solver. v1.0.0 used scikit-learn's L-BFGS (tolerance 1e-4). Version 1.1 minimises the same penalised
likelihood, C * sum(log-loss) + |beta|^2 / 2 with the intercept unpenalised, by Newton-Raphson to
convergence, so the choice between two neighbouring penalties never depends on where L-BFGS stopped.
`solver="lbfgs"` (only with every change off) is the v1.0.0 code path, used by the pipeline check.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit

from .lr import C_GRID
from .splines import KNOT_OPTIONS, choose_knots, expand

# change 2.3: exact functions of other inputs (the 2026 FNO analysis' derived inputs, and the clock
# hour of onset, whose circular scale has no linear meaning; night onset stays)
DERIVED_DROP = ("pulse_pressure", "ivt_to_puncture", "nihss_change_24h", "hyperglycemia", "onset_hour", "eni_24h")
EPS = np.finfo(float).eps


@dataclass(frozen=True)
class Spec:
    label: str = "1.0.0"
    per_input_indicator: bool = False     # 2.1
    min_missing: int = 1                  # 2.2 (1: any missing value gets an indicator, as v1.0.0)
    drop_derived: bool = False            # 2.3
    min_category: int = 0                 # 2.4 (0: no merging beyond the registry-level rule)
    repeats: int = 1                      # 2.5
    slope_rule: bool = False              # 2.5 (applied in deployment, see with_slope)
    solver: str = "newton"

    def replace(self, **kw) -> "Spec":
        return dataclasses.replace(self, **kw)

    @property
    def is_v100(self) -> bool:
        return (not self.per_input_indicator and self.min_missing <= 1 and not self.drop_derived
                and self.min_category <= 0 and self.repeats == 1)


V100 = Spec(label="1.0.0", solver="lbfgs")                      # the published code path
V100_NEWTON = Spec(label="1.0.0 specification")                 # the same model, exact solver
SIMPLIFIED = Spec(label="changes 2.1-2.4", per_input_indicator=True, min_missing=20, drop_derived=True,
                  min_category=20)
V110 = SIMPLIFIED.replace(label="1.1", repeats=5, slope_rule=True)
# change 3.1: the update model (previous stage's log-odds, current NIHSS with a 3-knot spline, the
# stage's events), tuned as in 2.5; its missing event inputs follow 2.1-2.2
UPDATE = Spec(label="update model", per_input_indicator=True, min_missing=20, repeats=5, slope_rule=True)


def input_of(col: str) -> str:
    """The input a model column belongs to: occlusion__M1 -> occlusion, age__rcs1 -> age."""
    return col.split("__")[0]


# --------------------------------------------------------------------------- layout (fitted on X)
@dataclass
class Layout:
    stage_columns: list          # the columns the model receives (a stage's design matrix)
    base: list                   # base columns the model uses
    sources: list                # for each base column, the stage columns summed into it (merges)
    knots: dict
    columns: list                # expanded columns (base, then spline terms)
    medians: np.ndarray
    ind: list                    # expanded columns that carry a missing-value indicator
    mean: np.ndarray
    scale: np.ndarray

    def base_matrix(self, A: np.ndarray) -> np.ndarray:
        out = np.empty((A.shape[0], len(self.sources)))
        for j, src in enumerate(self.sources):
            out[:, j] = A[:, src[0]] if len(src) == 1 else A[:, src].sum(axis=1)   # NaN stays NaN
        return out

    def expanded(self, X: pd.DataFrame) -> np.ndarray:
        B = pd.DataFrame(self.base_matrix(X[self.stage_columns].to_numpy(float)), columns=self.base, index=X.index)
        return expand(B, self.knots).to_numpy(float)

    def raw(self, E: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(E), self.medians[None, :], E)
        return np.hstack([filled, np.isnan(E[:, self.ind]).astype(float)]) if self.ind else filled

    def Z(self, X: pd.DataFrame) -> np.ndarray:
        return (self.raw(self.expanded(X)) - self.mean) / self.scale

    @property
    def owners(self) -> list:
        """Base-column index of every coefficient (expanded columns, then indicators)."""
        pos = {c: i for i, c in enumerate(self.base)}
        own = [pos[c.split("__rcs")[0]] if "__rcs" in c else pos[c] for c in self.columns]
        return own + [own[j] for j in self.ind]


def _merge_categories(X: pd.DataFrame, base: list, sources: dict, min_n: int) -> None:
    """Change 2.4: an occlusion category with fewer than `min_n` patients joins its parent."""
    from .clean import OCCLUSION_PARENT

    A = X.to_numpy(float)
    cols = list(X.columns)
    while True:
        moved = False
        for c in list(base):
            key, _, cat = c.partition("__")
            parent = f"{key}__{OCCLUSION_PARENT.get(cat, '')}"
            if key != "occlusion" or cat not in OCCLUSION_PARENT or parent not in sources:
                continue
            n = int(np.nansum(A[:, [cols.index(s) for s in sources[c]]].sum(axis=1) == 1))
            if n < min_n:
                sources[parent] = sources[parent] + sources.pop(c)
                base.remove(c)
                moved = True
        if not moved:
            return


def fit_layout(X: pd.DataFrame, spec: Spec, k: int) -> Layout:
    """Columns, merges, knots, medians, indicators and scaling learned on the fitting rows of X
    (bootstrap copies included, as in v1.0.0)."""
    cols = list(X.columns)
    base = [c for c in cols if not (spec.drop_derived and input_of(c) in DERIVED_DROP)]
    sources = {c: [c] for c in base}
    if spec.min_category > 0:
        _merge_categories(X, base, sources, spec.min_category)
    src_idx = [[cols.index(s) for s in sources[c]] for c in base]
    lay = Layout(cols, base, src_idx, {}, [], np.zeros(0), [], np.zeros(0), np.zeros(0))
    B = pd.DataFrame(lay.base_matrix(X.to_numpy(float)), columns=base, index=X.index)
    lay.knots = choose_knots(B, k=k)
    Ed = expand(B, lay.knots)
    lay.columns = list(Ed.columns)
    E = Ed.to_numpy(float)
    miss = np.isnan(E)
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(miss, np.nan, E), axis=0) if len(E) else np.zeros(E.shape[1])
    lay.medians = np.where(np.isnan(med), 0.0, med)          # an empty column: 0 (keep_empty_features)
    floor = max(1, int(spec.min_missing))
    nmiss = miss.sum(axis=0)
    if spec.per_input_indicator:
        first = {}
        for j, c in enumerate(lay.columns):
            first.setdefault(input_of(c), j)                 # every column of an input is missing together
        lay.ind = sorted(j for j in first.values() if nmiss[j] >= floor)
    else:
        lay.ind = [j for j in range(len(lay.columns)) if nmiss[j] >= floor]
    R = lay.raw(E)
    lay.mean = R.mean(axis=0)
    sd = R.std(axis=0)
    lay.scale = np.where(sd < 10 * EPS, 1.0, sd)
    return lay


# --------------------------------------------------------------------------- Newton solver
def _objective(A, y, w, C, beta, pen):
    eta = A @ beta
    return C * float(np.sum(w * (np.logaddexp(0.0, eta) - y * eta))) + 0.5 * float(np.sum(pen * beta * beta))


def newton(A: np.ndarray, y, w, C: float, beta=None, tol: float = 1e-10, max_iter: int = 500):
    """argmin C * sum_i w_i * logloss_i + |coef|^2 / 2 (last column of A: the unpenalised intercept).

    Newton-Raphson with the exact gradient at every step. The Hessian is refreshed only while the
    steps shrink by less than half (a chord step with a slightly old Hessian still converges to the
    same minimum, and costs a fraction of a Hessian); far from the minimum, steps are halved until
    the objective decreases. Stops when the largest step is below `tol` (relative)."""
    n, p = A.shape
    pen = np.ones(p)
    pen[-1] = 0.0
    beta = np.zeros(p) if beta is None else beta.copy()
    f = None
    Hinv, refresh, prev = None, True, np.inf
    eye = np.eye(p)
    for _ in range(max_iter):
        q = expit(A @ beta)
        g = C * (A.T @ (w * (q - y))) + pen * beta
        if refresh or Hinv is None:
            H = (A.T * (C * w * q * (1 - q))) @ A
            H[np.diag_indices(p)] += pen + 1e-12
            Hinv = cho_solve(cho_factor(H, check_finite=False), eye, check_finite=False)
        step = Hinv @ g
        s = float(np.max(np.abs(step)))
        refresh = s > 0.5 * prev
        t = 1.0
        if s > 1e-2:                     # far from the minimum: make sure the objective decreases
            f = _objective(A, y, w, C, beta, pen) if f is None else f
            while True:
                fn = _objective(A, y, w, C, beta - t * step, pen)
                if fn <= f + 1e-13 * abs(f) or t < 1e-8:
                    break
                t *= 0.5
                refresh = True
            f = fn
        else:
            f = None
        beta = beta - t * step
        prev = s * t
        if s * t <= tol * max(1.0, float(np.max(np.abs(beta)))):
            break
    return beta


def _with_ones(Z):
    return np.hstack([Z, np.ones((Z.shape[0], 1))])


def path_scores(Z, y, w, folds, Cs=C_GRID) -> np.ndarray:
    """Validation log-loss (count-weighted mean) for every fold x penalty, warm-started along Cs."""
    A = _with_ones(Z)
    S = np.empty((len(folds), len(Cs)))
    for f, (tr, va) in enumerate(folds):
        beta = None
        Atr, ytr, wtr = A[tr], y[tr], w[tr]
        for j, C in enumerate(Cs):
            beta = newton(Atr, ytr, wtr, float(C), beta)
            p = np.clip(expit(A[va] @ beta), EPS, 1 - EPS)
            ll = -(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p))
            S[f, j] = float(np.sum(w[va] * ll) / np.sum(w[va]))
    return S


# --------------------------------------------------------------------------- folds and copies
def make_folds(y, seed: int, repeats: int, groups=None) -> list:
    """Repeat r uses random_state seed + r; repeat 1 is v1.0.0's (tuning.shared_folds, or the
    bootstrap's stratified group folds of optimism.grouped_folds)."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    y = np.asarray(y)
    out = []
    for r in range(repeats):
        if groups is None:
            out += list(StratifiedKFold(5, shuffle=True, random_state=seed + r).split(np.zeros(len(y)), y))
        else:
            out += list(StratifiedGroupKFold(5, shuffle=True, random_state=seed + r).split(np.zeros(len(y)), y, np.asarray(groups)))
    return out


def _compress(groups, n: int):
    """Rows -> unique patients: (first row of each patient, row -> unique index, counts)."""
    if groups is None:
        return np.arange(n), np.arange(n), np.ones(n)
    g = np.asarray(groups)
    uniq, first, inv, counts = np.unique(g, return_index=True, return_inverse=True, return_counts=True)
    return first, inv, counts.astype(float)


def _unique_folds(folds, inv, n_unique: int) -> list:
    out = []
    for tr, va in folds:
        m = np.zeros(n_unique, bool)
        m[inv[va]] = True
        out.append((np.flatnonzero(~m), np.flatnonzero(m)))
    return out


# --------------------------------------------------------------------------- the fitted model
class Reference:
    """A fitted reference model. `coef`/`intercept` act on the standardised design (Layout.Z)."""

    kind = "spline"

    def __init__(self, layout: Layout, beta: np.ndarray, settings: dict, spec: Spec):
        self.layout, self.spec, self.settings = layout, spec, dict(settings)
        self.coef, self.intercept = np.asarray(beta[:-1], float), float(beta[-1])
        self.columns = list(layout.stage_columns)

    def logit(self, X: pd.DataFrame) -> np.ndarray:
        return self.layout.Z(X) @ self.coef + self.intercept

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = expit(self.logit(X))
        return np.column_stack([1 - p, p])

    @property
    def n_coef(self) -> int:
        return int(len(self.coef))

    def contributions(self, X: pd.DataFrame):
        """Per stage column, coefficient x standardised value summed over the column's terms
        (spline terms and indicator with their input; a merged category with its parent)."""
        parts = self.layout.Z(X) * self.coef[None, :]
        out = np.zeros((len(X), len(self.columns)))
        for j, o in enumerate(self.layout.owners):
            out[:, self.layout.sources[o][0]] += parts[:, j]
        return pd.DataFrame(out, columns=self.columns, index=X.index), self.intercept

    def export(self, shift: float = 0.0) -> dict:
        lay = self.layout
        spec = {"columns": list(lay.columns), "medians": [float(v) for v in lay.medians],
                "indicators": [int(i) for i in lay.ind], "mean": [float(v) for v in lay.mean],
                "scale": [float(v) for v in lay.scale], "coef": [float(v) for v in self.coef],
                "intercept": float(self.intercept), "shift": float(shift), "base_columns": list(lay.base),
                "knots": lay.knots}
        if lay.base != lay.stage_columns:
            # base column j = sum of these stage-row positions (web/model.js splineExpand)
            spec["from"] = [[int(i) for i in s] for s in lay.sources]
        return spec


def with_slope(m: Reference, X: pd.DataFrame, y, factor: float, w=None) -> Reference:
    """Change 2.5: the linear predictor times `factor`, the intercept re-estimated on X, y (uniform
    shrinkage, van Houwelingen and le Cessie 1990)."""
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    lp = factor * (m.layout.Z(X) @ m.coef)
    a = m.intercept
    for _ in range(100):
        q = expit(lp + a)
        h = float(np.sum(w * q * (1 - q)))
        step = float(np.sum(w * (y - q))) / max(h, 1e-12)
        a += step
        if abs(step) < 1e-12:
            break
    out = Reference(m.layout, np.r_[m.coef * factor, a], {**m.settings, "slope_factor": float(factor)}, m.spec)
    out.coef_uncorrected = m.coef
    return out


# --------------------------------------------------------------------------- tuning and fitting
def tune_scores(X: pd.DataFrame, y, seed: int, spec: Spec, groups=None, repeats: int | None = None,
                knot_options=KNOT_OPTIONS, row_folds=None):
    """{knots: (layout, Z of unique patients, fold x penalty log-loss)} and the unique patients'
    (y, counts). With bootstrap copies (`groups`) the layout uses every copy and the folds keep a
    patient's copies together; the solver works on unique patients weighted by their copies."""
    y = np.asarray(y, float)
    reps = spec.repeats if repeats is None else repeats
    first, inv, counts = _compress(groups, len(y))
    rf = row_folds[: 5 * reps] if row_folds is not None else make_folds(y, seed, reps, groups)
    folds = _unique_folds(rf, inv, len(first))
    yu = y[first]
    out = {}
    for k in knot_options:
        lay = fit_layout(X, spec, k)
        Zu = lay.Z(X.iloc[first])
        out[k] = (lay, Zu, path_scores(Zu, yu, counts, folds))
    return out, yu, counts


def choose(scores: dict, n_folds: int | None = None) -> tuple[int, int, float]:
    """(knots, index of C, mean log-loss) with the lowest mean over the first `n_folds` folds;
    ties go to the smaller penalty strength C and to fewer knots (as LogisticRegressionCV)."""
    best = None
    for k, (_, _, S) in scores.items():
        m = S[: n_folds or len(S)].mean(axis=0)
        j = int(np.argmin(m))
        if best is None or m[j] < best[2]:
            best = (k, j, float(m[j]))
    return best


def fit_chosen(scores: dict, yu, counts, k: int, j: int, cv_ll: float, spec: Spec, n_folds: int) -> Reference:
    lay, Zu, _ = scores[k]
    C = float(C_GRID[j])
    beta = newton(_with_ones(Zu), yu, counts, C)
    st = {"knots": int(k), "C": C, "cv_logloss": cv_ll, "candidates": len(C_GRID), "cv_folds": int(n_folds),
          "C_at_grid_end": bool(j in (0, len(C_GRID) - 1))}
    return Reference(lay, beta, st, spec)


def fit(X: pd.DataFrame, y, seed: int, spec: Spec, groups=None, settings: dict | None = None,
        knot_options=KNOT_OPTIONS, row_folds=None):
    """Tune knots and penalty on X, y (or, with `settings`, refit those) and fit the model."""
    y = np.asarray(y, float)
    if spec.solver == "lbfgs":
        if not spec.is_v100 or groups is not None:
            raise ValueError("the L-BFGS path is the v1.0.0 specification only")
        from .splines import fit_spline_logreg, fit_spline_logreg_fixed
        if settings is None:
            return V100Model(fit_spline_logreg(X, y, seed))
        m = fit_spline_logreg_fixed(X, y, int(settings["knots"]), float(settings["C"]))
        m.settings = dict(settings)
        return V100Model(m)
    if settings is not None:
        first, inv, counts = _compress(groups, len(y))
        lay = fit_layout(X, spec, int(settings["knots"]))
        Zu = lay.Z(X.iloc[first])
        beta = newton(_with_ones(Zu), y[first], counts, float(settings["C"]))
        st = {k: settings[k] for k in settings if k != "slope_factor"}
        return Reference(lay, beta, st, spec)
    scores, yu, counts = tune_scores(X, y, seed, spec, groups, knot_options=knot_options, row_folds=row_folds)
    nf = 5 * spec.repeats
    k, j, ll = choose(scores, nf)
    return fit_chosen(scores, yu, counts, k, j, ll, spec, nf)


class V100Model:
    """The v1.0.0 model (splines.SplineLogReg) behind the interface of `Reference`."""

    kind = "spline"

    def __init__(self, m):
        self.m, self.settings, self.columns = m, dict(m.settings), list(m.columns)

    def predict_proba(self, X):
        return self.m.predict_proba(X)

    def logit(self, X):
        p = np.clip(self.m.predict_proba(X)[:, 1], 1e-15, 1 - 1e-15)
        return np.log(p / (1 - p))

    @property
    def n_coef(self) -> int:
        return int(np.ravel(self.m.model.steps[2][1].coef_).size)

    def export(self, shift: float = 0.0) -> dict:
        from .splines import export_spline_logreg
        return export_spline_logreg(self.m, shift)
