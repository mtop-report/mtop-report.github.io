"""Discrimination, calibration and clinical-utility metrics (TRIPOD+AI set)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-np.clip(np.asarray(z, float), -500, 500)))


def _fit_logistic(x, y, offset=None, slope=True):
    """Tiny, dependency-free logistic fit: y ~ a + b*x (+ offset)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    off = np.zeros_like(x) if offset is None else np.asarray(offset, float)

    def nll(w):
        a = w[0]
        b = w[1] if slope else 0.0
        z = a + b * x + off
        return np.sum(np.logaddexp(0, z) - y * z)

    w0 = np.array([0.0, 1.0]) if slope else np.array([0.0])
    res = minimize(nll, w0, method="BFGS")
    return res.x


def calibration_intercept_slope(y, p):
    """Calibration-in-the-large (intercept with slope fixed at 1) and calibration slope."""
    lp = logit(p)
    citl = _fit_logistic(np.zeros_like(lp), y, offset=lp, slope=False)[0]
    slope = _fit_logistic(lp, y)[1]
    return float(citl), float(slope)


def fast_citl_slope(y, lp, iters: int = 25):
    """The same two quantities by Newton-Raphson (used inside the bootstrap, where BFGS is slow)."""
    y = np.asarray(y, float)
    lp = np.asarray(lp, float)
    a = 0.0
    for _ in range(iters):
        q = expit(lp + a)
        g, hss = np.sum(y - q), np.sum(q * (1 - q))
        if hss <= 1e-12:
            break
        step = g / hss
        a += step
        if abs(step) < 1e-9:
            break
    w = np.zeros(2)
    X = np.column_stack([np.ones_like(lp), lp])
    for _ in range(iters):
        q = expit(X @ w)
        H = (X * (q * (1 - q))[:, None]).T @ X
        try:
            step = np.linalg.solve(H + 1e-9 * np.eye(2), X.T @ (y - q))
        except np.linalg.LinAlgError:
            break
        w += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(a), float(w[1])


def ici(y, p, frac=0.75):
    """Integrated calibration index (Austin & Steyerberg 2019) using LOWESS.

    Falls back to a binned estimate for scores with few distinct values (e.g. SPAN-100).
    """
    import warnings

    from statsmodels.nonparametric.smoothers_lowess import lowess

    p = np.asarray(p, float)
    y = np.asarray(y, float)
    if len(np.unique(np.round(p, 6))) >= 20:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm = lowess(y, p, frac=frac, it=0, return_sorted=False)
        if np.all(np.isfinite(sm)):
            return float(np.mean(np.abs(sm - p)))
    vals, inv = np.unique(np.round(p, 6), return_inverse=True)
    obs = np.array([y[inv == k].mean() for k in range(len(vals))])
    return float(np.mean(np.abs(obs[inv] - p)))


def calibration_bins(y, p, n_bins=10, min_n=10):
    """Quantile bins of predicted risk with observed proportion (only bins with >= min_n)."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    out = []
    for b in bins:
        if len(b) < min_n:
            continue
        out.append({"pred": float(p[b].mean()), "obs": float(y[b].mean()), "n": int(len(b)),
                    "lo": float(p[b].min()), "hi": float(p[b].max())})
    return out


def roc_points(y, p, grid=None):
    """ROC curve resampled on a fixed FPR grid (no individual operating points exposed)."""
    fpr, tpr, _ = roc_curve(y, p)
    grid = np.linspace(0, 1, 51) if grid is None else grid
    t = np.interp(grid, fpr, tpr)
    t[0] = 0.0
    return [{"fpr": float(a), "tpr": float(b)} for a, b in zip(grid, t)]


def decision_curve(y, p, thresholds=None):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    n = len(y)
    th = np.round(np.arange(0.05, 0.81, 0.05), 2) if thresholds is None else thresholds
    prev = y.mean()
    out = []
    for t in th:
        pos = p >= t
        tp = np.sum(pos & (y == 1))
        fp = np.sum(pos & (y == 0))
        nb = tp / n - fp / n * t / (1 - t)
        nb_all = prev - (1 - prev) * t / (1 - t)
        out.append({"t": float(t), "model": float(nb), "all": float(nb_all), "none": 0.0})
    return out


def summary(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    citl, slope = calibration_intercept_slope(y, p)
    brier = brier_score_loss(y, p)
    brier_ref = y.mean() * (1 - y.mean())
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "prevalence": float(y.mean()),
        "mean_pred": float(p.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier),
        "brier_scaled": float(1 - brier / brier_ref),
        "citl": citl,
        "slope": slope,
        "ici": ici(y, p),
    }


def bootstrap_ci(y, preds: dict, n_boot=2000, seed=0, ref=None, calibration=True):
    """Percentile 95% CIs for AUROC, Brier, calibration-in-the-large and slope of each model, plus
    the paired AUROC difference against `ref` (model minus reference: positive = better than the
    reference), with a two-sided bootstrap p-value."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    names = list(preds)
    P = {k: np.asarray(preds[k], float) for k in names}
    LP = {k: logit(P[k]) for k in names}
    stats = {k: {"auroc": [], "brier": [], "citl": [], "slope": []} for k in names}
    deltas = {k: [] for k in names if ref and k != ref and ref in preds}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        aucs = {}
        for k in names:
            pb = P[k][idx]
            aucs[k] = roc_auc_score(yb, pb)
            stats[k]["auroc"].append(aucs[k])
            if np.all((pb > 0) & (pb < 1)):
                stats[k]["brier"].append(brier_score_loss(yb, pb))
            if calibration:
                a, b = fast_citl_slope(yb, LP[k][idx])
                if np.isfinite(a) and np.isfinite(b) and abs(b) < 50:
                    stats[k]["citl"].append(a)
                    stats[k]["slope"].append(b)
        for k in deltas:
            deltas[k].append(aucs[k] - aucs[ref])
    out = {}
    for k in names:
        out[k] = {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                  for m, v in stats[k].items() if len(v)}
    for k, v in deltas.items():
        v = np.asarray(v)
        out[k]["delta_auroc_vs_ref"] = {
            "ref": ref,
            "mean": float(v.mean()),
            "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))),
        }
    return out


def oe_ratio(y, p) -> float:
    """Observed / expected events (1 = calibrated in the large; < 1 = the model over-predicts)."""
    e = float(np.sum(p))
    return float(np.sum(y) / e) if e > 0 else float("nan")
