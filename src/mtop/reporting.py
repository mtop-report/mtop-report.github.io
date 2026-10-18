"""Numbers for the reporting-standards page (artifacts/public/reporting.json): sample size.

Development (TRIPOD+AI item 10; Riley et al., BMJ 2020; pmsampsize for a binary outcome), for the
reference model (regression with splines) of every stage and outcome:
  (i)   expected shrinkage >= 0.9:  n = p / ((S - 1) ln(1 - R2cs / S)), S = 0.9
  (ii)  optimism in Nagelkerke R2 <= 0.05:  S_ii = R2cs / (R2cs + 0.05 R2max), n as in (i) with S_ii
  (iii) overall risk within +-0.05:  n = (1.96 / 0.05)^2 phi (1 - phi)
  p = the model's parameters (spline terms, one-hot columns and missing-value indicators),
  R2cs = Cox-Snell R2 of out-of-fold predictions in the development patients; also the expected
  (van Houwelingen) shrinkage at the available n: 1 + p / (n ln(1 - R2cs)).
Validation (Riley et al., Stat Med 2021; pmvalsampsize for a binary outcome), for each temporal
split with the reference model's observed performance:
  O/E with 95% CI width 0.2, calibration slope with CI width 0.2, C-statistic with CI width 0.1.
Aggregates only.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import learners as Lr
from . import metrics as M
from .cohort import flow, split_temporal, splits, stage_cohort
from .config import settings, stage_ids, targets
from .features import design_matrix, monotone_vector
from .paths import PRIVATE, PUBLIC
from .train import load_deployed


def r2_cs(y, p) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    phi = y.mean()
    ll0 = len(y) * (phi * np.log(phi) + (1 - phi) * np.log(1 - phi))
    return float(1 - np.exp(2 * (ll0 - ll) / len(y)))


def development_size(p: int, phi: float, r2: float, n_avail: int, shrink: float = 0.9, delta: float = 0.05) -> dict:
    r2max = 1 - (phi ** phi * (1 - phi) ** (1 - phi)) ** 2
    n_i = p / ((shrink - 1) * np.log(1 - r2 / shrink))
    s_ii = r2 / (r2 + delta * r2max)
    n_ii = p / ((s_ii - 1) * np.log(1 - r2 / s_ii))
    n_iii = (1.96 / 0.05) ** 2 * phi * (1 - phi)
    need = int(np.ceil(max(n_i, n_ii, n_iii)))
    return {"p": int(p), "prevalence": float(phi), "r2_cs": float(r2), "r2_max": float(r2max),
            "n_i": int(np.ceil(n_i)), "n_ii": int(np.ceil(n_ii)), "n_iii": int(np.ceil(n_iii)), "n_required": need,
            "events_required": int(np.ceil(need * phi)), "epp_required": float(need * phi / p),
            "n_available": int(n_avail), "events_available": int(round(n_avail * phi)), "epp_available": float(n_avail * phi / p),
            "shrinkage_expected": float(1 + p / (n_avail * np.log(1 - r2))), "met": bool(n_avail >= need)}


def validation_size(y, p, width_oe=0.2, width_slope=0.2, width_c=0.1) -> dict:
    """Riley et al. 2021 criteria with the observed outcome rate, C-statistic and linear-predictor
    distribution of the reference model in the validation patients."""
    y = np.asarray(y, float)
    phi = float(y.mean())
    lp = M.logit(p)
    c = float(M.roc_auc_score(y, p))
    # O/E: SE(ln O/E) ~ sqrt((1 - phi) / (n phi)); CI for O/E = 1 of width ~ 2 * 1.96 * SE
    se_oe = width_oe / (2 * 1.96)
    n_oe = (1 - phi) / (phi * se_oe ** 2)
    # calibration slope: information per patient E[w (lp - m)^2] under perfect calibration
    q = M.expit(lp)
    w = q * (1 - q)
    m = np.sum(w * lp) / np.sum(w)
    a = np.mean(w)
    b = np.mean(w * (lp - m))
    cc = np.mean(w * (lp - m) ** 2)
    info = cc - b ** 2 / a
    se_slope = width_slope / (2 * 1.96)
    n_slope = 1 / (info * se_slope ** 2)
    # C-statistic (Newcombe): solve SE(C; n) = width / (2 * 1.96)
    se_c = width_c / (2 * 1.96)

    def se_of(n):
        n1, n0 = n * phi, n * (1 - phi)
        return np.sqrt(c * (1 - c) * (1 + (n1 - 1) * (1 - c) / (2 - c) + (n0 - 1) * c / (1 + c)) / (n1 * n0))

    n_c = brentq(lambda n: se_of(n) - se_c, 20, 1e7)
    need = int(np.ceil(max(n_oe, n_slope, n_c)))
    return {"prevalence": phi, "c": c, "n_oe": int(np.ceil(n_oe)), "n_slope": int(np.ceil(n_slope)), "n_c": int(np.ceil(n_c)),
            "n_required": need, "events_required": int(np.ceil(need * phi)), "n_available": int(len(y)),
            "events_available": int(y.sum()), "met": bool(len(y) >= need)}


def build_reporting(clean: pd.DataFrame) -> dict:
    cfg = settings()
    seed = int(cfg["seed"])
    min_leaf = int(cfg["privacy"]["min_leaf"])
    base, _ = flow(clean)
    ref = Lr.REFERENCE
    primary = json.loads((PRIVATE / "train_primary.json").read_text())
    out = {"reference": ref, "development": {}, "validation": {}}
    for stage in stage_ids():
        d = stage_cohort(base, stage)
        dev, _ = split_temporal(d)
        X = design_matrix(d, stage).loc[dev.index]
        out["development"][stage] = {}
        for target in targets():
            y = dev[target].to_numpy(float)
            st = primary["stages"][stage]["targets"][target]["settings"][ref]
            oof = Lr.oof(ref, st, X, y, seed + 3, monotone_vector(stage, target), min_leaf=min_leaf)
            pk = load_deployed(PRIVATE, stage, target)
            p_par = int(np.ravel(pk["models"][ref]["fitted"].model.model.steps[2][1].coef_).size) + 1
            out["development"][stage][target] = development_size(p_par, float(y.mean()), r2_cs(y, oof), len(y))
    preds = {}
    for label in splits():
        f = PRIVATE / f"predictions_{label}.parquet"
        if f.exists():
            preds[label] = pd.read_parquet(f)
    for label, pr in preds.items():
        out["validation"][label] = {}
        for stage in stage_ids():
            out["validation"][label][stage] = {}
            for target in targets():
                r = pr[(pr["stage"] == stage) & (pr["target"] == target) & (pr["model"] == ref)]
                if len(r) and 0 < r["y"].sum() < len(r):
                    out["validation"][label][stage][target] = validation_size(r["y"].to_numpy(float), r["p"].to_numpy(float))
    with open(PUBLIC / "reporting.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    a = out["development"]["admission"]["good_outcome"]
    print(f"reporting: admission mRS 0-2 needs n={a['n_required']} (p={a['p']}), has {a['n_available']}")
    return out
