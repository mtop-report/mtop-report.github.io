"""Penalised logistic regression trained by federated Newton steps.

Each Newton step needs only X'(y - p), X'WX and the log-likelihood summed over all patients; the
centres return these sums (secure aggregation), the coordinator solves the small linear system.
The result equals pooled penalised regression up to the fixed-point rounding of the sums
(about 1e-7). Used for two things:

  spline   the report's model family: restricted cubic splines for the continuous predictors
           with pre-agreed knots, missing values set to a public default plus a missing-indicator,
           columns standardised with network-wide means and SDs (one extra round);
  recal    calibration of any shared model on a group of centres: logit(P) = a + b * logit(p)
           (slope), or a + logit(p) with b fixed to 1 (calibration-in-the-large).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import stage_features
from ..features import design_matrix
from ..splines import rcs_terms
from .bins import fed_settings

CLIP = 1e-12


def spline_spec(stage: str) -> dict:
    lr = fed_settings()["logreg"]
    keys = {f.key for f in stage_features(stage)}
    return {"kind": "spline", "stage": stage, "knots": {k: v for k, v in lr["knots"].items() if k in keys}}


def spline_design(df: pd.DataFrame, spec: dict) -> tuple[np.ndarray, list[str]]:
    """Unstandardised design matrix (no intercept). Identical columns at every centre."""
    stage = spec["stage"]
    X = design_matrix(df, stage)
    cols, names = [], []
    for f in stage_features(stage):
        sub = X[f.columns].to_numpy(float)
        miss = np.isnan(sub).any(axis=1)
        if f.type == "categorical":
            fill = np.where(np.isnan(sub), 0.0, sub)
        elif f.type == "derived":
            fill = np.where(np.isnan(sub), 0.0, sub)
        else:
            default = float(f.spec.get("default", 0.0))
            fill = np.where(np.isnan(sub), default, sub)
        for j, c in enumerate(f.columns):
            cols.append(fill[:, j])
            names.append(c)
        if f.key in spec["knots"]:
            terms = rcs_terms(fill[:, 0], spec["knots"][f.key])
            for j in range(terms.shape[1]):
                cols.append(terms[:, j])
                names.append(f"{f.key}__rcs{j + 1}")
        cols.append(miss.astype(float))
        names.append(f"{f.key}__missing")
    return np.column_stack(cols), names


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def newton_terms(X: np.ndarray, y: np.ndarray, beta: np.ndarray, offset=None) -> np.ndarray:
    """[X'(y-p), upper triangle of X'WX, log-likelihood] as one float vector."""
    eta = X @ beta + (0.0 if offset is None else offset)
    p = 1.0 / (1.0 + np.exp(-eta))
    grad = X.T @ (y - p)
    W = p * (1.0 - p)
    hess = (X * W[:, None]).T @ X
    ll = float(np.sum(y * np.log(np.clip(p, CLIP, 1)) + (1 - y) * np.log(np.clip(1 - p, CLIP, 1))))
    iu = np.triu_indices(X.shape[1])
    return np.concatenate([grad, hess[iu], [ll]])


def unpack_terms(vec: np.ndarray, d: int):
    grad = vec[:d]
    iu = np.triu_indices(d)
    h = np.zeros((d, d))
    h[iu] = vec[d:d + len(iu[0])]
    h = h + h.T - np.diag(np.diag(h))
    return grad, h, float(vec[-1])


def newton(step, d: int, lam: float, penalise: np.ndarray, iterations: int = 25, tol: float = 1e-8,
           beta0=None) -> tuple[np.ndarray, list[float], int]:
    """Maximise loglik - lam/2 * ||beta[penalise]||^2 given step(beta) -> (grad, hess, loglik)."""
    beta = np.zeros(d) if beta0 is None else np.asarray(beta0, float).copy()
    P = np.diag(penalise.astype(float)) * lam
    trace = []
    it = 0
    for it in range(1, iterations + 1):
        g, H, ll = step(beta)
        trace.append(ll - 0.5 * lam * float(np.sum((beta * penalise) ** 2)))
        delta = np.linalg.solve(H + P, g - P @ beta)
        beta = beta + delta
        if float(np.max(np.abs(delta))) < tol:
            break
    return beta, trace, it


def predict_spline(model: dict, df: pd.DataFrame) -> np.ndarray:
    X, _ = spline_design(df, model["design"])
    Z = (X - np.asarray(model["mean"])) / np.asarray(model["sd"])
    eta = model["beta"][0] + Z @ np.asarray(model["beta"][1:])
    return 1.0 / (1.0 + np.exp(-eta))
