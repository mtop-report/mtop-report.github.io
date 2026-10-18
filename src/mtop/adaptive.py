"""Adaptive updating: how should MTOP be kept up to date as practice and outcomes change?

Outputs aggregate results only (artifacts/public/adaptive.json):

Every publishable learner is followed (regression with splines, linear regression, LightGBM,
XGBoost), each with settings fixed in advance (below), so the strategies can be compared within and
across learners.

1. Rolling-origin ("prequential") comparison of update strategies. A base model is trained on
   patients treated up to BASE_END. Each test year Y (FIRST_TEST..LAST_TEST) is predicted using
   only data from earlier years:
     frozen       base model, never updated
     recal_int    base model, intercept re-estimated on the previous RECAL_YEARS years
     recal_log    base model, intercept and slope re-estimated on the previous RECAL_YEARS years
     refit        refitted on all earlier years (expanding window)
     refit_recal  refitted on all earlier years, then intercept shifted to the previous
                  RECAL_YEARS years (shift estimated out-of-sample); the MTOP v1.0 policy
     sliding      refitted on the previous WINDOW years only
   Settings are fixed in advance for every learner: the trees use generic hyperparameters with the
   number of trees chosen by 5-fold cross-validation inside each training set; the regressions
   choose their penalty from 9 values by 5-fold cross-validation inside each training set (4 spline
   knots placed on the training data). Nothing from a test year leaks into a model.
2. Drift monitor: observed vs expected outcome rate per year for each frozen base model with
   95 % and 3-sigma control limits, plus the cumulative standardised difference.
3. Internal-external validation across treatment eras (leave one era out), pooled with a
   DerSimonian-Laird random-effects model on the logit-AUROC scale.
4. Data availability: share of treated adults with each predictor recorded, per year.

Per-year discrimination is reported only when a year has >= 10 events and >= 10 non-events.
"""
from __future__ import annotations

import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import metrics as M
from .cohort import flow, stage_cohort
from .config import settings, stage_ids, targets
from .lr import fit_logreg
from .splines import fit_spline_logreg
from .tuning import fit_xgb_fixed, threads
from .features import design_matrix, monotone_vector
from .paths import PUBLIC

BASE_END = 2015
FIRST_TEST = 2018
LAST_TEST = 2025
RECAL_YEARS = 2
WINDOW = 5
MIN_EVENTS = 10
N_BOOT = 500

STRATEGIES = ["frozen", "recal_int", "recal_log", "refit", "refit_recal", "sliding"]
STRATEGY_LABELS = {
    "frozen": {"en": "Frozen (never updated)", "cs": "Zmrazený (bez aktualizace)"},
    "recal_int": {"en": "Recalibrate intercept", "cs": "Rekalibrace interceptu"},
    "recal_log": {"en": "Recalibrate intercept + slope", "cs": "Rekalibrace interceptu a sklonu"},
    "refit": {"en": "Refit, all earlier years", "cs": "Přetrénování, všechny dřívější roky"},
    "refit_recal": {"en": "Refit + recalibrate (MTOP v1.0)", "cs": "Přetrénování + rekalibrace (MTOP v1.0)"},
    "sliding": {"en": f"Refit, last {WINDOW} years only", "cs": f"Přetrénování, jen posledních {WINDOW} let"},
}

ERAS = [(0, 2014, "2005–2014"), (2015, 2017, "2015–2017"), (2018, 2020, "2018–2020"), (2021, 3000, "2021–2025")]

LEARNERS = ("Logistic regression (splines)", "Logistic regression", "LightGBM", "XGBoost")
C_GRID_ADAPTIVE = np.logspace(-3, 1, 9)     # regression penalties tried in every training window
GENERIC_XGB = {"learning_rate": 0.03, "max_depth": 3, "min_child_weight": 5.0, "subsample": 0.8,
               "colsample_bytree": 0.8, "lambda": 1.0, "gamma": 0.0}
GENERIC = dict(objective="binary", learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=25,
               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
               min_sum_hessian_in_leaf=5.0, verbosity=-1, deterministic=True, force_row_wise=True,
               num_threads=2, monotone_constraints_method="advanced")

AVAILABILITY = [
    ("aspects", {"en": "ASPECTS", "cs": "ASPECTS"}),
    ("glucose", {"en": "Glucose", "cs": "Glykémie"}),
    ("sbp", {"en": "Systolic BP", "cs": "Systolický tlak"}),
    ("onset_to_puncture", {"en": "Onset to puncture", "cs": "Od vzniku do punkce"}),
    ("anesthesia", {"en": "Anaesthesia", "cs": "Anestezie"}),
    ("smoking", {"en": "Smoking", "cs": "Kouření"}),
    ("tici", {"en": "Final mTICI", "cs": "Výsledné mTICI"}),
    ("procedure_min", {"en": "Procedure duration", "cs": "Délka výkonu"}),
    ("nihss_24h_status", {"en": "NIHSS at 24 h", "cs": "NIHSS za 24 h"}),
    ("nihss_7d_status", {"en": "NIHSS day 7", "cs": "NIHSS 7. den"}),
    ("mrs90", {"en": "90-day mRS", "cs": "mRS za 90 dní"}),
]


def _params(stage, target, seed):
    p = dict(GENERIC)
    p["seed"] = seed
    p["num_threads"] = threads()
    p["monotone_constraints"] = monotone_vector(stage, target)
    return p


def fit_cv(X, y, params, seed):
    """Fit with the number of trees chosen by 5-fold CV on the training data only."""
    y = np.asarray(y, float)
    rounds = 100
    if y.sum() >= 10 and (len(y) - y.sum()) >= 10:
        folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
        cv = lgb.cv(params, lgb.Dataset(X, y, params={"feature_pre_filter": False}), num_boost_round=2000,
                    folds=folds, callbacks=[lgb.early_stopping(100, verbose=False)])
        rounds = len(cv["valid binary_logloss-mean"])
    return lgb.train(params, lgb.Dataset(X, y, params={"feature_pre_filter": False}), num_boost_round=rounds)


def fit_learner(name: str, X: pd.DataFrame, y, stage: str, target: str, seed: int):
    """A model of the given learner with the fixed settings above; returns predict(X) -> p."""
    y = np.asarray(y, float)
    if name == "LightGBM":
        b = fit_cv(X, y, _params(stage, target, seed), seed)
        return lambda Z: b.predict(Z)
    if name == "XGBoost":
        m = fit_xgb_fixed(X, y, GENERIC_XGB, seed, trees=100, cv_trees=bool(y.sum() >= 10 and len(y) - y.sum() >= 10))
        return lambda Z: m.predict_proba(Z)[:, 1]
    if name == "Logistic regression":
        m = fit_logreg(X, y, seed, Cs=C_GRID_ADAPTIVE)
        return lambda Z: m.predict_proba(Z)[:, 1]
    if name == "Logistic regression (splines)":
        m = fit_spline_logreg(X, y, seed, knot_options=(4,), Cs=C_GRID_ADAPTIVE)
        return lambda Z: m.predict_proba(Z)[:, 1]
    raise KeyError(name)


def _citl(y, p):
    return float(M._fit_logistic(np.zeros(len(y)), np.asarray(y, float), offset=M.logit(p), slope=False)[0])


def _recal_log(y, p):
    a, b = M._fit_logistic(M.logit(p), np.asarray(y, float))
    return float(a), float(b)


def _enough(y):
    y = np.asarray(y, float)
    return y.sum() >= MIN_EVENTS and (len(y) - y.sum()) >= MIN_EVENTS


def _year_metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    out = {"exp": float(p.mean()), "oe": float(y.mean() / p.mean()), "citl": _citl(y, p),
           "brier": float(brier_score_loss(y, p)), "auroc": None}
    if _enough(y):
        out["auroc"] = float(roc_auc_score(y, p))
    return out


def _boot_auc(y, p, seed, n=N_BOOT):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    v = np.asarray(vals)
    lv = np.log(v / (1 - v))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], float(lv.std(ddof=1))


def rolling(d: pd.DataFrame, stage: str, target: str, seed: int, learner: str = "LightGBM") -> dict:
    X = design_matrix(d, stage)
    y = d[target].to_numpy(float)
    yr = d["year"].to_numpy()
    cache = {}

    def model_until(end_year, start_year=None):
        key = (start_year, end_year)
        if key not in cache:
            m = (yr <= end_year) if start_year is None else ((yr >= start_year) & (yr <= end_year))
            cache[key] = fit_learner(learner, X[m], y[m], stage, target, seed)
        return cache[key]

    base = model_until(BASE_END)
    preds = {s: np.full(len(y), np.nan) for s in STRATEGIES}
    for Y in range(FIRST_TEST, LAST_TEST + 1):
        te = yr == Y
        rc = (yr >= Y - RECAL_YEARS) & (yr < Y)
        if te.sum() == 0:
            continue
        pf_te, pf_rc = base(X[te]), base(X[rc])
        preds["frozen"][te] = pf_te
        preds["recal_int"][te] = M.expit(M.logit(pf_te) + _citl(y[rc], pf_rc))
        a, b = _recal_log(y[rc], pf_rc)
        preds["recal_log"][te] = M.expit(a + b * M.logit(pf_te))
        p_refit = model_until(Y - 1)(X[te])
        preds["refit"][te] = p_refit
        shift = _citl(y[rc], model_until(Y - 1 - RECAL_YEARS)(X[rc]))
        preds["refit_recal"][te] = M.expit(M.logit(p_refit) + shift)
        preds["sliding"][te] = model_until(Y - 1, Y - WINDOW)(X[te])

    years = []
    for Y in range(FIRST_TEST, LAST_TEST + 1):
        te = yr == Y
        if te.sum() < 10:
            continue
        years.append({"year": Y, "n": int(te.sum()), "events": int(y[te].sum()), "obs": float(y[te].mean()),
                      "by": {s: _year_metrics(y[te], preds[s][te]) for s in STRATEGIES}})
    window = (yr >= FIRST_TEST) & (yr <= LAST_TEST)
    pooled = {}
    for s in STRATEGIES:
        p = preds[s][window]
        yy = y[window]
        summ = M.summary(yy, p)
        ci, _ = _boot_auc(yy, p, seed)
        pooled[s] = {"n": summ["n"], "events": summ["events"], "auroc": summ["auroc"], "auroc_ci": ci,
                     "citl": summ["citl"], "slope": summ["slope"], "brier": summ["brier"], "ici": summ["ici"],
                     "mean_abs_citl": float(np.mean([abs(r["by"][s]["citl"]) for r in years]))}

    # drift monitor for the frozen base model, from the first year after BASE_END
    monitor = []
    cum_o = cum_e = cum_v = 0.0
    for Y in range(BASE_END + 1, LAST_TEST + 1):
        m = yr == Y
        if m.sum() < 10:
            continue
        p = base(X[m])
        o, e, v = float(y[m].sum()), float(p.sum()), float((p * (1 - p)).sum())
        n = int(m.sum())
        sd = np.sqrt(v) / n
        z = (o - e) / np.sqrt(v)
        cum_o, cum_e, cum_v = cum_o + o, cum_e + e, cum_v + v
        zc = (cum_o - cum_e) / np.sqrt(cum_v)
        flag = "alarm" if abs(z) > 3 else ("warning" if abs(z) > 1.96 else "ok")
        monitor.append({"year": Y, "n": n, "obs": o / n, "exp": e / n,
                        "lo95": e / n - 1.96 * sd, "hi95": e / n + 1.96 * sd,
                        "lo3": e / n - 3 * sd, "hi3": e / n + 3 * sd,
                        "z": float(z), "zcum": float(zc), "flag": flag})
    return {"years": years, "pooled": pooled, "monitor": monitor}


def random_effects(est, se):
    """DerSimonian-Laird pooling. Returns pooled estimate, CI, prediction interval, tau2, I2."""
    est = np.asarray(est, float)
    se = np.asarray(se, float)
    w = 1 / se**2
    fixed = np.sum(w * est) / np.sum(w)
    q = float(np.sum(w * (est - fixed) ** 2))
    k = len(est)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 else 0.0
    wr = 1 / (se**2 + tau2)
    mu = float(np.sum(wr * est) / np.sum(wr))
    se_mu = float(np.sqrt(1 / np.sum(wr)))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    from scipy.stats import t as tdist

    tq = float(tdist.ppf(0.975, max(1, k - 2)))
    half_pi = tq * np.sqrt(tau2 + se_mu**2)
    return {"mu": mu, "lo": mu - 1.96 * se_mu, "hi": mu + 1.96 * se_mu,
            "pi_lo": mu - half_pi, "pi_hi": mu + half_pi, "tau2": float(tau2), "i2": float(i2), "q": q, "k": k}


def iecv(d: pd.DataFrame, stage: str, target: str, seed: int, learner: str = "LightGBM") -> dict:
    X = design_matrix(d, stage)
    y = d[target].to_numpy(float)
    yr = d["year"].to_numpy()
    rows = []
    for lo, hi, label in ERAS:
        te = (yr >= lo) & (yr <= hi)
        if te.sum() < 30 or not _enough(y[te]):
            rows.append({"label": label, "suppressed": True})
            continue
        p = fit_learner(learner, X[~te], y[~te], stage, target, seed)(X[te])
        ci, se_l = _boot_auc(y[te], p, seed)
        auc = float(roc_auc_score(y[te], p))
        rows.append({"label": label, "n": int(te.sum()), "events": int(y[te].sum()), "auroc": auc, "ci": ci,
                     "logit": float(np.log(auc / (1 - auc))), "se_logit": se_l, "citl": _citl(y[te], p),
                     "slope": _recal_log(y[te], p)[1], "oe": float(y[te].mean() / p.mean())})
    ok = [r for r in rows if not r.get("suppressed")]
    pooled = None
    if len(ok) >= 2:
        re = random_effects([r["logit"] for r in ok], [r["se_logit"] for r in ok])
        ex = lambda v: float(1 / (1 + np.exp(-v)))  # noqa: E731
        pooled = {"auroc": ex(re["mu"]), "ci": [ex(re["lo"]), ex(re["hi"])], "pi": [ex(re["pi_lo"]), ex(re["pi_hi"])],
                  "i2": re["i2"], "tau2": re["tau2"], "k": re["k"]}
    return {"eras": rows, "pooled": pooled}


def availability(clean: pd.DataFrame) -> dict:
    d = clean[~(clean["age"] < 18)].copy()
    d["yr"] = d["year"].clip(lower=2007)
    years, labels, counts = [], [], []
    for yr, g in d.groupby("yr"):
        if len(g) >= 10:
            years.append(int(yr))
            labels.append("2005–2007" if yr == 2007 else str(int(yr)))
            counts.append(int(len(g)))
    rows = []
    for key, label in AVAILABILITY:
        vals = []
        for yr in years:
            g = d[d["yr"] == yr][key]
            rec = (g != "missing") & g.notna() if key.endswith("_status") else g.notna()
            vals.append(round(100 * float(rec.mean()), 1))
        rows.append({"key": key, "label": label, "values": vals})
    return {"years": labels, "n": counts, "variables": rows}


def _task(args):
    """One stage x outcome x learner: rolling-origin strategies and leave-one-era-out validation."""
    base, (stage, target, learner), seed = args
    t0 = time.time()
    d = stage_cohort(base, stage)
    res = rolling(d, stage, target, seed, learner)
    ie = iecv(d, stage, target, seed, learner)
    pr = res["pooled"]
    print(f"[adaptive {stage}/{target} {learner}] pooled AUROC " +
          " ".join(f"{s}={pr[s]['auroc']:.3f}/citl {pr[s]['citl']:+.2f}" for s in STRATEGIES) +
          f" ({time.time() - t0:.0f}s)", flush=True)
    return res, ie


def build_adaptive(clean: pd.DataFrame, stages=None, targets_=None, learners=LEARNERS) -> dict:
    """MTOP_JOBS > 1 runs the stage x outcome x learner tasks in parallel processes."""
    seed = int(settings()["seed"])
    base, _ = flow(clean)
    out = {
        "settings": {"base_end": BASE_END, "first_test": FIRST_TEST, "last_test": LAST_TEST,
                     "recal_years": RECAL_YEARS, "window": WINDOW, "min_events": MIN_EVENTS,
                     "params": {k: v for k, v in GENERIC.items() if k not in ("verbosity", "num_threads")},
                     "xgb": GENERIC_XGB, "regression": {"penalty": "L2, chosen by 5-fold CV in each training set",
                                                        "candidates": [float(c) for c in C_GRID_ADAPTIVE], "spline_knots": 4}},
        "learners": list(learners),
        "strategies": [{"key": s, "label": STRATEGY_LABELS[s]} for s in STRATEGIES],
        "results": {}, "iecv": {}, "availability": availability(clean),
    }
    tasks = [(stage, target, learner) for stage in (stages or stage_ids()) for target in (targets_ or list(targets()))
             for learner in learners]
    for stage, target, _ in tasks:
        out["results"].setdefault(stage, {}).setdefault(target, {})
        out["iecv"].setdefault(stage, {}).setdefault(target, {})
    jobs = int(os.environ.get("MTOP_JOBS", "1"))
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for (stage, target, learner), (res, ie) in zip(tasks, ex.map(_task, [(base, t, seed) for t in tasks])):
                out["results"][stage][target][learner], out["iecv"][stage][target][learner] = res, ie
    else:
        for t in tasks:
            stage, target, learner = t
            out["results"][stage][target][learner], out["iecv"][stage][target][learner] = _task((base, t, seed))
    PUBLIC.mkdir(parents=True, exist_ok=True)
    with open(PUBLIC / "adaptive.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    return out
