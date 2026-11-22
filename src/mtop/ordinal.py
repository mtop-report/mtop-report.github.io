"""MTOP ordinal model: a penalised proportional-odds (cumulative logit) regression on the 90-day mRS.

Plan: docs/registrations/ordinal_plan.md (committed 22 November 2026 before any of this code).

Model (plan, section 4): logit P(mRS <= k | x) = theta_k - x'beta, k = 0..K-2, one coefficient vector
shared by all cut-points. Objective C * sum_i w_i * (-log pi_i) + |beta|^2 / 2, the thresholds not
penalised, minimised by Newton-Raphson to convergence (as the v1.1 reference, reference.newton). The
inputs, layout and tuning grid are those of the v1.1 reference (reference.V110, reference.fit_layout);
only the outcome and the likelihood change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit

from . import reference as R
from .lr import C_GRID

EPS = np.finfo(float).eps
UW_MRS = np.array([1.0, 0.91, 0.76, 0.65, 0.33, 0.0, 0.0])   # utility weights, Chaisinanunkul et al. 2015
TINY = 1e-300


# --------------------------------------------------------------------------- likelihood pieces
def _cdf_pdf(x):
    """sigma(x), sigma'(x) and sigma''(x); +-inf give (1|0, 0, 0)."""
    F = expit(x)
    f = F * (1.0 - F)
    fp = f * (1.0 - 2.0 * F)
    return F, f, fp


def _bounds(theta, eta, y):
    """Upper and lower cut-point arguments a = theta_y - eta, b = theta_{y-1} - eta (+-inf at the ends)."""
    K = len(theta) + 1
    th = np.concatenate([[-np.inf], np.asarray(theta, float), [np.inf]])
    a = th[y + 1] - eta
    b = th[y] - eta
    return a, b, K


def _pi(a, b):
    """P(b < latent <= a) computed without cancellation: sigma(a) - sigma(b), or sigma(-b) - sigma(-a)
    when both are in the upper tail."""
    up = b > 0
    pi = np.where(up, expit(-b) - expit(-a), expit(a) - expit(b))
    return pi


def nll(theta, beta, Z, y, w, offset=0.0):
    eta = Z @ beta + offset
    if np.ndim(eta) == 0:
        eta = np.full(len(y), float(eta))
    if USE_NUMBA:
        return float(_nll_numba(np.ascontiguousarray(eta, float), np.ascontiguousarray(theta, float),
                                np.ascontiguousarray(y, np.int64), np.ascontiguousarray(w, float), len(theta) + 1))
    a, b, _ = _bounds(theta, eta, y)
    pi = _pi(a, b)
    if np.any(~(pi > 0)):
        return np.inf
    return float(-np.sum(w * np.log(pi)))


def objective(params, Z, y, w, C, K, offset=0.0, pen=1.0):
    beta, theta = params[: Z.shape[1]], params[Z.shape[1]:]
    if np.any(np.diff(theta) <= 0):
        return np.inf
    v = nll(theta, beta, Z, y, w, offset)
    return C * v + 0.5 * pen * float(beta @ beta)


def _terms_numpy(eta, theta, y, w, K):
    """Per-patient derivatives of the negative log-likelihood: d/d eta and d2/d eta2 (n), the
    threshold gradient (K-1), the threshold Hessian (K-1 x K-1) and the eta-threshold cross terms
    (n x K-1, column k holding the patients whose upper or lower threshold is theta_k)."""
    n = len(eta)
    a, b, _ = _bounds(theta, eta, y)
    pi = np.maximum(_pi(a, b), TINY)
    _, fa, fpa = _cdf_pdf(a)
    _, fb, fpb = _cdf_pdf(b)
    fa = np.where(np.isfinite(a), fa, 0.0)
    fpa = np.where(np.isfinite(a), fpa, 0.0)
    fb = np.where(np.isfinite(b), fb, 0.0)
    fpb = np.where(np.isfinite(b), fpb, 0.0)
    d = fa - fb
    g_eta = w * d / pi
    h_eta = -w * ((fpa - fpb) / pi - d * d / pi ** 2)
    g_up = -w * fa / pi                       # d/d theta_y        (y < K-1)
    g_lo = w * fb / pi                        # d/d theta_{y-1}    (y > 0)
    h_upup = -w * (fpa / pi - fa * fa / pi ** 2)
    h_lolo = -w * (-fpb / pi - fb * fb / pi ** 2)
    h_uplo = -w * (fa * fb / pi ** 2)
    h_eta_up = -w * (-fpa / pi + fa * d / pi ** 2)
    h_eta_lo = -w * (fpb / pi - fb * d / pi ** 2)
    m = K - 1
    up = y < K - 1          # has an upper threshold theta_y
    lo = y > 0              # has a lower threshold theta_{y-1}
    iu, il = y[up], y[lo] - 1
    g_th = np.bincount(iu, weights=g_up[up], minlength=m) + np.bincount(il, weights=g_lo[lo], minlength=m)
    Mt = np.zeros((n, m))
    Mt[np.flatnonzero(up), iu] += h_eta_up[up]
    Mt[np.flatnonzero(lo), il] += h_eta_lo[lo]
    Tt = np.zeros((m, m))
    Tt[np.diag_indices(m)] += np.bincount(iu, weights=h_upup[up], minlength=m) + np.bincount(il, weights=h_lolo[lo], minlength=m)
    both = up & lo                         # patients with both thresholds: theta_y and theta_{y-1}
    off = np.bincount(y[both] - 1, weights=h_uplo[both], minlength=m)   # entry (y-1, y)
    for k in range(m - 1):
        Tt[k, k + 1] += off[k]
        Tt[k + 1, k] += off[k]
    return g_eta, h_eta, g_th, Tt, Mt


try:                                           # the same arithmetic, fused (an optional speed-up)
    import numba as _nb

    @_nb.njit(cache=True)
    def _sig(x):
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        e = np.exp(x)
        return e / (1.0 + e)

    @_nb.njit(cache=True)
    def _terms_numba(eta, theta, y, w, K):
        n, m = len(eta), K - 1
        g_eta, h_eta = np.zeros(n), np.zeros(n)
        g_th, Tt, Mt = np.zeros(m), np.zeros((m, m)), np.zeros((n, m))
        for i in range(n):
            yi = y[i]
            has_a, has_b = yi < K - 1, yi > 0
            fa = fpa = fb = fpb = 0.0
            if has_a:
                a = theta[yi] - eta[i]
                Fa = _sig(a)
                fa = Fa * (1.0 - Fa)
                fpa = fa * (1.0 - 2.0 * Fa)
            if has_b:
                b = theta[yi - 1] - eta[i]
                Fb = _sig(b)
                fb = Fb * (1.0 - Fb)
                fpb = fb * (1.0 - 2.0 * Fb)
            if has_b and b > 0:
                pi = _sig(-b) - (_sig(-a) if has_a else 0.0)
            else:
                pi = (_sig(a) if has_a else 1.0) - (_sig(b) if has_b else 0.0)
            pi = max(pi, 1e-300)
            wi, d = w[i], fa - fb
            g_eta[i] = wi * d / pi
            h_eta[i] = -wi * ((fpa - fpb) / pi - d * d / (pi * pi))
            if has_a:
                g_th[yi] += -wi * fa / pi
                Tt[yi, yi] += -wi * (fpa / pi - fa * fa / (pi * pi))
                Mt[i, yi] += -wi * (-fpa / pi + fa * d / (pi * pi))
            if has_b:
                g_th[yi - 1] += wi * fb / pi
                Tt[yi - 1, yi - 1] += -wi * (-fpb / pi - fb * fb / (pi * pi))
                Mt[i, yi - 1] += -wi * (fpb / pi - fb * d / (pi * pi))
            if has_a and has_b:
                h = -wi * (fa * fb / (pi * pi))
                Tt[yi - 1, yi] += h
                Tt[yi, yi - 1] += h
        return g_eta, h_eta, g_th, Tt, Mt

    @_nb.njit(cache=True)
    def _nll_numba(eta, theta, y, w, K):
        tot = 0.0
        for i in range(len(eta)):
            yi = y[i]
            has_a, has_b = yi < K - 1, yi > 0
            a = theta[yi] - eta[i] if has_a else np.inf
            b = theta[yi - 1] - eta[i] if has_b else -np.inf
            if has_b and b > 0:
                pi = _sig(-b) - (_sig(-a) if has_a else 0.0)
            else:
                pi = (_sig(a) if has_a else 1.0) - (_sig(b) if has_b else 0.0)
            if not pi > 0:
                return np.inf
            tot -= w[i] * np.log(pi)
        return tot

    HAVE_NUMBA = True
except ImportError:                            # pragma: no cover
    HAVE_NUMBA = False

USE_NUMBA = HAVE_NUMBA


def _terms(eta, theta, y, w, K):
    if USE_NUMBA:
        return _terms_numba(np.ascontiguousarray(eta, float), np.ascontiguousarray(theta, float),
                            np.ascontiguousarray(y, np.int64), np.ascontiguousarray(w, float), K)
    return _terms_numpy(eta, theta, y, w, K)


def grad_hess(params, Z, y, w, C, K, offset=0.0, pen=1.0):
    """Gradient and Hessian of the penalised objective (beta first, then the K-1 thresholds).
    `offset` is added to the linear predictor; `pen` scales the penalty (0: unpenalised)."""
    n, p = Z.shape
    beta, theta = params[:p], params[p:]
    eta = Z @ beta + offset
    if np.ndim(eta) == 0:
        eta = np.full(n, float(eta))
    g_eta, h_eta, g_th, Tt, Mt = _terms(eta, theta, y, w, K)
    m = K - 1
    g = np.empty(p + m)
    g[:p] = Z.T @ g_eta
    g[p:] = g_th
    H = np.empty((p + m, p + m))
    H[:p, :p] = (Z.T * h_eta) @ Z
    Bt = Z.T @ Mt
    H[:p, p:] = Bt
    H[p:, :p] = Bt.T
    H[p:, p:] = Tt
    g *= C
    H *= C
    g[:p] += pen * beta
    H[np.arange(p), np.arange(p)] += pen
    return g, H


def start_thresholds(y, w, K):
    cum = np.array([np.sum(w[y <= k]) for k in range(K - 1)]) / np.sum(w)
    cum = np.clip(cum, 1e-4, 1 - 1e-4)
    th = np.log(cum / (1 - cum))
    for k in range(1, len(th)):                # strictly increasing
        th[k] = max(th[k], th[k - 1] + 1e-3)
    return th


def newton(Z, y, w, C: float, K: int, params=None, tol: float = 1e-10, max_iter: int = 500,
           offset=0.0, pen: float = 1.0):
    """argmin C * sum_i w_i * (-log pi_i) + pen * |beta|^2 / 2 over (beta, theta), the linear
    predictor Z beta + offset. Full Newton steps, halved until the objective decreases (which also
    keeps the thresholds ordered); stops when the largest step is below `tol` relative to the largest
    parameter. A design with no columns fits the thresholds alone."""
    Z = np.asarray(Z, float)
    y = np.asarray(y, int)
    w = np.asarray(w, float)
    n, p = Z.shape
    if params is None:
        th0 = start_thresholds(y, w, K)
        if np.ndim(offset) and len(np.atleast_1d(offset)) == n:
            th0 = th0 + float(np.average(offset, weights=w))
        params = np.concatenate([np.zeros(p), th0])
    else:
        params = np.asarray(params, float).copy()
    f = objective(params, Z, y, w, C, K, offset, pen)
    for _ in range(max_iter):
        g, H = grad_hess(params, Z, y, w, C, K, offset, pen)
        H[np.diag_indices(len(params))] += 1e-12
        step = cho_solve(cho_factor(H, check_finite=False), g, check_finite=False)
        t = 1.0
        while True:
            cand = params - t * step
            fn = objective(cand, Z, y, w, C, K, offset, pen)
            if fn <= f + 1e-13 * abs(f) or t < 1e-10:
                break
            t *= 0.5
        if not np.isfinite(fn):
            break
        params, f = cand, fn
        s = float(np.max(np.abs(t * step)))
        if s <= tol * max(1.0, float(np.max(np.abs(params)))):
            break
    return params


def probs(theta, eta):
    """n x K grade probabilities from thresholds and linear predictors."""
    th = np.asarray(theta, float)
    cum = expit(th[None, :] - np.asarray(eta, float)[:, None])
    n = cum.shape[0]
    cum = np.hstack([np.zeros((n, 1)), cum, np.ones((n, 1))])
    return np.clip(np.diff(cum, axis=1), 0.0, 1.0)


# --------------------------------------------------------------------------- tuning (v1.1 grid)
def path_scores(Z, y, w, K, folds, Cs=C_GRID) -> np.ndarray:
    """Validation ordinal log-loss (count-weighted mean) for every fold x penalty, warm-started."""
    S = np.empty((len(folds), len(Cs)))
    for fi, (tr, va) in enumerate(folds):
        par = None
        Ztr, ytr, wtr = Z[tr], y[tr], w[tr]
        for j, C in enumerate(Cs):
            par = newton(Ztr, ytr, wtr, float(C), K, par)
            p = Z.shape[1]
            P = probs(par[p:], Z[va] @ par[:p])
            ll = -np.log(np.clip(P[np.arange(len(va)), y[va]], EPS, None))
            S[fi, j] = float(np.sum(w[va] * ll) / np.sum(w[va]))
    return S


class Ordinal:
    """A fitted ordinal model. Coefficients act on the standardised design of `layout`."""

    kind = "ordinal"

    def __init__(self, layout: R.Layout, params: np.ndarray, K: int, settings: dict):
        p = len(layout.mean)
        self.layout, self.K, self.settings = layout, K, dict(settings)
        self.coef, self.theta = np.asarray(params[:p], float), np.asarray(params[p:], float)
        self.columns = list(layout.stage_columns)

    def lp(self, X: pd.DataFrame) -> np.ndarray:
        return self.layout.Z(X) @ self.coef

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return probs(self.theta, self.lp(X))

    @property
    def n_coef(self) -> int:
        return int(len(self.coef))

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Exact additive explanation of the linear predictor per stage column (coefficient x
        standardised value, spline terms and indicator with their input, a merged category with its
        parent); the same for every cut-point because the coefficients are shared. Rows sum to lp."""
        parts = self.layout.Z(X) * self.coef[None, :]
        out = np.zeros((len(X), len(self.columns)))
        for j, o in enumerate(self.layout.owners):
            out[:, self.layout.sources[o][0]] += parts[:, j]
        return pd.DataFrame(out, columns=self.columns, index=X.index)

    def export(self, shift: float = 0.0) -> dict:
        """The numbers the browser needs (web/model.js, kind "ordinal"): the spline-regression layout
        of reference.Reference.export with intercept 0, the thresholds and the recent-era shift."""
        lay = self.layout
        spec = {"columns": list(lay.columns), "medians": [float(v) for v in lay.medians],
                "indicators": [int(i) for i in lay.ind], "mean": [float(v) for v in lay.mean],
                "scale": [float(v) for v in lay.scale], "coef": [float(v) for v in self.coef], "intercept": 0.0,
                "shift": 0.0, "base_columns": list(lay.base), "knots": lay.knots,
                "theta": [float(v) for v in self.theta], "theta_shift": float(shift)}
        if lay.base != lay.stage_columns:
            spec["from"] = [[int(i) for i in src] for src in lay.sources]
        return spec


def eval_export(spec: dict, X: pd.DataFrame, recent: bool = True):
    """Linear predictor, grade probabilities and per-column contributions from an exported model
    (the arithmetic of web/model.js), for parity tests and the validation tool."""
    from .splines import rcs_terms
    A = X.to_numpy(float)
    if "from" in spec:
        B = np.column_stack([A[:, src].sum(axis=1) for src in spec["from"]])
        own0 = [src[0] for src in spec["from"]]
    else:
        B, own0 = A, list(range(A.shape[1]))
    cols, owner = [B], list(own0)
    for v, knots in spec["knots"].items():
        i = spec["base_columns"].index(v)
        T = rcs_terms(B[:, i], knots)
        cols.append(T)
        owner += [own0[i]] * T.shape[1]
    E = np.hstack(cols)
    med = np.asarray(spec["medians"], float)
    F = np.where(np.isnan(E), med[None, :], E)
    ind = spec["indicators"]
    Zr = np.hstack([F, np.isnan(E[:, ind]).astype(float)]) if ind else F
    owner += [owner[j] for j in ind]
    terms = (Zr - np.asarray(spec["mean"])) / np.asarray(spec["scale"]) * np.asarray(spec["coef"])
    lp = terms.sum(axis=1)
    phi = np.zeros((len(X), A.shape[1]))
    for j, o in enumerate(owner):
        phi[:, o] += terms[:, j]
    th = np.asarray(spec["theta"], float) + (spec.get("theta_shift", 0.0) if recent else 0.0)
    return lp, probs(th, lp), phi


def tune_scores(X, y, K, seed, spec=R.V110, groups=None, repeats=5, knot_options=R.KNOT_OPTIONS, row_folds=None):
    """{knots: (layout, Z of unique patients, fold x penalty ordinal log-loss)}, with folds stratified
    by grade (repeat 1 with random_state seed, as reference.make_folds)."""
    y = np.asarray(y, int)
    first, inv, counts = R._compress(groups, len(y))
    rf = row_folds[: 5 * repeats] if row_folds is not None else R.make_folds(y, seed, repeats, groups)
    folds = R._unique_folds(rf, inv, len(first))
    yu = y[first]
    out = {}
    for k in knot_options:
        lay = R.fit_layout(X, spec, k)
        Zu = lay.Z(X.iloc[first])
        out[k] = (lay, Zu, path_scores(Zu, yu, counts, K, folds))
    return out, yu, counts


def fit_chosen(scores, yu, counts, K, k, j, cv_ll, n_folds) -> Ordinal:
    lay, Zu, _ = scores[k]
    C = float(C_GRID[j])
    par = newton(Zu, yu, counts, C, K)
    st = {"knots": int(k), "C": C, "cv_logloss": float(cv_ll), "cv_folds": int(n_folds),
          "C_at_grid_end": bool(j in (0, len(C_GRID) - 1))}
    return Ordinal(lay, par, K, st)


def fit(X, y, K, seed, groups=None, repeats=5, row_folds=None) -> Ordinal:
    sc, yu, cn = tune_scores(X, y, K, seed, groups=groups, repeats=repeats, row_folds=row_folds)
    k, j, ll = R.choose(sc, 5 * repeats)
    return fit_chosen(sc, yu, cn, K, k, j, ll, 5 * repeats)


def fit_fixed(X, y, K, settings: dict, groups=None) -> Ordinal:
    """Refit with fixed knots and penalty (the layout from X, copies weighted as in tune_scores)."""
    y = np.asarray(y, int)
    first, inv, counts = R._compress(groups, len(y))
    lay = R.fit_layout(X, R.V110, int(settings["knots"]))
    par = newton(lay.Z(X.iloc[first]), y[first], counts, float(settings["C"]), K)
    return Ordinal(lay, par, K, {k: settings[k] for k in settings if k != "slope_factor"})


# --------------------------------------------------------------------------- recalibration pieces
def fit_thresholds(eta, y, K, w=None, theta0=None) -> np.ndarray:
    """Maximum-likelihood thresholds for a fixed linear predictor `eta`."""
    eta = np.asarray(eta, float)
    w = np.ones(len(eta)) if w is None else np.asarray(w, float)
    par = None if theta0 is None else np.asarray(theta0, float)
    return newton(np.zeros((len(eta), 0)), y, w, 1.0, K, par, offset=eta, pen=0.0)


def calibration_slope(eta, y, K, w=None) -> tuple[float, np.ndarray]:
    """Ordinal calibration slope (plan, section 6): the coefficient of the linear predictor, and the
    thresholds, of an unpenalised proportional-odds model refitted on it."""
    eta = np.asarray(eta, float)
    w = np.ones(len(eta)) if w is None else np.asarray(w, float)
    par = newton(eta[:, None], y, w, 1.0, K, pen=0.0)
    return float(par[0]), par[1:]


def threshold_shift(eta, theta, y, K, w=None, tol: float = 1e-12) -> float:
    """Common shift d of all thresholds (P(mRS <= k) = expit(theta_k + d - eta)) by maximum
    likelihood, the linear predictor and the spacing of the thresholds fixed."""
    eta = np.asarray(eta, float)
    w = np.ones(len(eta)) if w is None else np.asarray(w, float)
    one = np.ones((len(eta), 1))
    th = np.asarray(theta, float)
    b = 0.0                                   # logit = theta - eta - b, so d = -b
    for _ in range(100):
        g, H = grad_hess(np.r_[b, th], one, y, w, 1.0, K, eta, 0.0)
        step = g[0] / H[0, 0]
        b -= step
        if abs(step) < tol:
            break
    return float(-b)


def with_slope(m: Ordinal, X, y, factor: float, w=None) -> Ordinal:
    """Rule O3: the linear predictor times `factor`, the thresholds re-estimated on X, y (as
    reference.with_slope does for the intercept)."""
    eta = factor * m.lp(X)
    th = fit_thresholds(eta, y, m.K, w, m.theta)
    out = Ordinal(m.layout, np.r_[m.coef * factor, th], m.K, {**m.settings, "slope_factor": float(factor)})
    out.coef_uncorrected = m.coef
    return out
