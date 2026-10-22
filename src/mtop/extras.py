"""Deployment extras for the calculator (artifacts/public/extras.json, aggregates only).

1. Comparators for the "other models" panel, fitted on all years and shifted to the recent era:
   published point scores mapped to probabilities, and published or bedside predictor sets refitted
   on the registry. (The other learners are in model_bundle.json.)
2. Full 90-day mRS distribution: a cumulative-logit (proportional odds) layer on the two chosen
   models' scores (mRS 0-2 and death), fitted on out-of-fold scores. The browser uses it only to
   split the independent (0-2) and dependent (3-5) bands into single grades, so the headline
   probabilities stay those of the binary models.
3. Prediction uncertainty: bootstrap refits of the chosen models (Riley & Collins 2023); 95 %
   ranges summarised by predicted-risk bin.
4. A synthetic demo cohort in the upload template format for the in-browser validation tool.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import learners as Lr
from . import metrics as M
from .clean import TICI_LABEL, clean_registry, load_raw
from .cohort import flow, split_temporal, stage_cohort
from .config import features, settings, stage_ids
from .export import clean_json
from .features import design_matrix, monotone_vector
from .lr import export_logreg, fit_logreg, fit_logreg_fixed
from .paths import PRIVATE, PUBLIC, SYNTHETIC
from .scores import PANEL, POINT_SCORES, REFIT_TERMS, apply_points_map, design, fit_points_map
from .train import load_deployed

UW_MRS = [1.0, 0.91, 0.76, 0.65, 0.33, 0.0, 0.0]  # utility-weighted mRS (Chaisinanunkul et al., Stroke 2015)
N_BOOT = 100
RISK_BINS = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]


# ------------------------------------------------------------------------------ ordinal layer
def _unpack(w, fixed_beta=None):
    if fixed_beta is None:
        bg, bd, t0, *inc = w
    else:
        (bg, bd), (t0, *inc) = fixed_beta, w
    th = t0 + np.concatenate([[0.0], np.cumsum(np.exp(inc))])
    return bg, bd, th


def ordinal_probs(zg, zd, bg, bd, th):
    eta = bg * np.asarray(zg) + bd * np.asarray(zd)
    cum = M.expit(np.asarray(th)[None, :] - eta[:, None])
    n = len(eta)
    cum = np.hstack([np.zeros((n, 1)), cum, np.ones((n, 1))])
    return np.diff(cum, axis=1)  # n x 7


def fit_ordinal(zg, zd, y, fixed_beta=None, init=None):
    y = np.asarray(y, int)
    n = len(y)
    marg = np.array([(y <= k).mean() for k in range(6)]).clip(0.01, 0.99)
    th0 = np.log(marg / (1 - marg))
    w_th = [th0[0]] + list(np.log(np.maximum(np.diff(th0), 0.05)))
    w0 = (w_th if fixed_beta is not None else [-1.0, 1.0] + w_th) if init is None else init

    def nll(w):
        bg, bd, th = _unpack(w, fixed_beta)
        p = ordinal_probs(zg, zd, bg, bd, th)
        return -np.sum(np.log(np.clip(p[np.arange(n), y], 1e-12, None)))

    res = minimize(nll, np.asarray(w0, float), method="BFGS")
    bg, bd, th = _unpack(res.x, fixed_beta)
    return float(bg), float(bd), [float(v) for v in th]


def full_distribution(p_good, p_death, q):
    """Split the binary-model bands into grades using the ordinal layer's within-band shares."""
    p_good = np.asarray(p_good, float).copy()
    p_death = np.asarray(p_death, float).copy()
    over = p_good + p_death > 0.98
    k = np.where(over, 0.98 / (p_good + p_death), 1.0)
    p_good, p_death = p_good * k, p_death * k
    p_dep = 1 - p_good - p_death
    q = np.asarray(q, float)
    good_share = q[:, :3] / q[:, :3].sum(1, keepdims=True)
    dep_share = q[:, 3:6] / q[:, 3:6].sum(1, keepdims=True)
    return np.hstack([good_share * p_good[:, None], dep_share * p_dep[:, None], p_death[:, None]])


def generalized_c(score_better, mrs):
    """Probability that, of two patients with different mRS, the one predicted better did better."""
    s = np.asarray(score_better, float)
    y = np.asarray(mrs, float)
    ds = s[:, None] - s[None, :]
    dy = y[None, :] - y[:, None]   # positive when i has lower (better) mRS than j
    mask = dy > 0
    if mask.sum() == 0:
        return None
    conc = (ds[mask] > 0).sum() + 0.5 * (ds[mask] == 0).sum()
    return float(conc / mask.sum())


# ------------------------------------------------------------------------------ helpers
def _oof_logreg(Xd, y, seed, C):
    p = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xd, y):
        p[va] = fit_logreg_fixed(Xd.iloc[tr], y[tr], C).predict_proba(Xd.iloc[va])[:, 1]
    return p


def _citl(y, p):
    return float(M._fit_logistic(np.zeros(len(y)), np.asarray(y, float), offset=M.logit(p), slope=False)[0])


def _bins(p0, lo_off, hi_off, min_n):
    out = []
    for a, b in zip(RISK_BINS[:-1], RISK_BINS[1:]):
        m = (p0 >= a) & (p0 < b)
        if m.sum() < min_n:
            continue
        pm = float(np.median(p0[m]))
        lo, hi = float(np.median(lo_off[m])), float(np.median(hi_off[m]))
        out.append({"lo": float(a), "hi": float(min(b, 1.0)), "n": int(m.sum()), "off_lo": lo, "off_hi": hi,
                    "p": pm, "p_lo": float(M.expit(M.logit(pm) + lo)), "p_hi": float(M.expit(M.logit(pm) + hi))})
    return out


# ------------------------------------------------------------------------------ comparators
def stage_comparators(stage: str, d: pd.DataFrame, yg: np.ndarray, recent: np.ndarray, seed: int) -> dict:
    """Scores and published predictor sets for the calculator's "other models" panel (mRS 0-2),
    each shifted to the recent era."""
    comp = {}
    for name in PANEL[stage]:
        if name in POINT_SCORES:
            sc = POINT_SCORES[name](d)
            mp = fit_points_map(sc, yg)
            mp["shift"] = _citl(yg[recent], apply_points_map(mp, sc[recent]))
            comp[name] = {"kind": "points", "map": mp}
        else:
            terms = REFIT_TERMS[name]
            Xd = design(d, terms)
            mdl = fit_logreg(Xd, yg, seed)
            sh = _citl(yg[recent], _oof_logreg(Xd, yg, seed, mdl.settings["C"])[recent])
            comp[name] = {"kind": "refit", "terms": [list(t) for t in terms],
                          "spec": export_logreg(mdl, list(Xd.columns), sh)}
    return comp


def update_comparators(clean: pd.DataFrame) -> dict:
    """Rebuild only the comparator panel in extras.json (the slow parts stay as they are)."""
    cfg = settings()
    seed = int(cfg["seed"])
    recent_from = int(cfg["split"]["recent_era_from_year"])
    base, _ = flow(clean)
    path = PUBLIC / "extras.json"
    out = json.loads(path.read_text(encoding="utf-8"))
    out["panel"] = PANEL
    for stage in stage_ids():
        d = stage_cohort(base, stage)
        recent = (d["year"] >= recent_from).to_numpy()
        out["comparators"][stage] = stage_comparators(stage, d, d["good_outcome"].to_numpy(float), recent, seed)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean_json(out), fh, ensure_ascii=False, separators=(",", ":"))
    return out


# ------------------------------------------------------------------------------ main
def build_extras(clean: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    cfg = settings()
    seed = int(cfg["seed"])
    recent_from = int(cfg["split"]["recent_era_from_year"])
    min_cell = int(cfg["privacy"]["min_cell"])
    min_leaf = int(cfg["privacy"]["min_leaf"])
    primary = json.loads((PRIVATE / "train_primary.json").read_text())
    test_preds = pd.read_parquet(PRIVATE / "predictions_primary.parquet")
    base, _ = flow(clean)
    out = {"uw_mrs": UW_MRS, "panel": PANEL, "comparators": {}, "ordinal": {}, "ordinal_validation": {},
           "uncertainty": {}, "stability": {}}

    for stage in stage_ids():
        t0 = time.time()
        d = stage_cohort(base, stage)
        X = design_matrix(d, stage)
        recent = (d["year"] >= recent_from).to_numpy()
        y = {t: d[t].to_numpy(float) for t in ("good_outcome", "death90")}
        dep = {t: load_deployed(PRIVATE, stage, t) for t in y}
        chosen = {t: dep[t]["chosen"] for t in y}
        mono = {t: monotone_vector(stage, t) for t in y}

        # 1. comparators (primary outcome)
        out["comparators"][stage] = stage_comparators(stage, d, y["good_outcome"], recent, seed)

        # 2. ordinal layer on out-of-fold scores of the chosen deployed models
        z = {}
        for t in y:
            f = dep[t]["models"][chosen[t]]["fitted"]
            z[t] = M.logit(Lr.oof(chosen[t], f.settings, X, y[t], seed + 1, mono[t], min_leaf=min_leaf))
        mrs = d["mrs90"].to_numpy(int)
        bg, bd, _ = fit_ordinal(z["good_outcome"], z["death90"], mrs)
        _, _, th = fit_ordinal(z["good_outcome"][recent], z["death90"][recent], mrs[recent], fixed_beta=(bg, bd))
        out["ordinal"][stage] = {"bg": bg, "bd": bd, "theta": th, "models": chosen}

        # 2b. temporal validation of the full distribution (development models of the primary split)
        dev, test = split_temporal(d)
        Xdv = X.loc[dev.index]
        zd_dev, p_test = {}, {}
        for t in y:
            st = primary["stages"][stage]["targets"][t]["settings"][chosen[t]]
            zd_dev[t] = M.logit(Lr.oof(chosen[t], st, Xdv, dev[t].to_numpy(float), seed + 2, mono[t], min_leaf=min_leaf))
            tp = test_preds[(test_preds["stage"] == stage) & (test_preds["target"] == t) & (test_preds["model"] == chosen[t])]
            p_test[t] = tp.set_index("patient_id").loc[test["patient_id"].to_numpy(), "p"].to_numpy(float)
        mrs_dev, mrs_te = dev["mrs90"].to_numpy(int), test["mrs90"].to_numpy(int)
        vbg, vbd, vth = fit_ordinal(zd_dev["good_outcome"], zd_dev["death90"], mrs_dev)
        q = ordinal_probs(M.logit(p_test["good_outcome"]), M.logit(p_test["death90"]), vbg, vbd, vth)
        dist = full_distribution(p_test["good_outcome"], p_test["death90"], q)
        uw = dist @ np.asarray(UW_MRS)
        cum = np.cumsum(dist, axis=1)
        out["ordinal_validation"][stage] = {
            "n": int(len(mrs_te)),
            "observed": [float((mrs_te == k).mean()) for k in range(7)],
            "predicted": [float(v) for v in dist.mean(0)],
            "c_ordinal": generalized_c(uw, mrs_te),
            "cum_auroc": [float(roc_auc_score(mrs_te <= k, cum[:, k])) if 0 < (mrs_te <= k).sum() < len(mrs_te) else None
                          for k in range(6)],
            "uw_obs": float(np.mean(np.asarray(UW_MRS)[mrs_te])), "uw_pred": float(uw.mean()),
        }

        # 3. bootstrap uncertainty of the chosen deployed models
        rng = np.random.default_rng(seed)
        out["uncertainty"][stage], out["stability"][stage] = {}, {}
        for t in y:
            m = dep[t]["models"][chosen[t]]
            f, shift = m["fitted"], m["shift"]
            l0 = f.logit(X) + shift
            p0 = M.expit(l0)
            L = np.empty((n_boot, len(l0)))
            for b in range(n_boot):
                i = rng.integers(0, len(l0), len(l0))
                fb = Lr.refit(chosen[t], f.settings, X.iloc[i], y[t][i], seed, mono[t], min_leaf=min_leaf)
                L[b] = fb.logit(X) + shift
            lo_off = np.percentile(L, 2.5, axis=0) - l0
            hi_off = np.percentile(L, 97.5, axis=0) - l0
            out["uncertainty"][stage][t] = _bins(p0, lo_off, hi_off, min_cell)
            pb = M.expit(L)
            # classification instability at 0.5 (Riley & Collins 2023): the share of bootstrap models that
            # put a patient on the other side of the threshold than the published model
            flip = ((pb >= 0.5) != (p0 >= 0.5)[None, :]).mean(axis=0)
            out["stability"][stage][t] = {
                "model": chosen[t],
                "mape": float(np.mean(np.abs(pb - p0[None, :]))),
                "width_median": float(np.median(np.percentile(pb, 97.5, axis=0) - np.percentile(pb, 2.5, axis=0))),
                "share_wide": float(np.mean((np.percentile(pb, 97.5, axis=0) - np.percentile(pb, 2.5, axis=0)) > 0.2)),
                "share_flip_05": float(np.mean(flip > 0.2)),
                "share_near_05": float(np.mean(np.abs(p0 - 0.5) < 0.05)),
                "n_boot": n_boot,
            }
        ov = out["ordinal_validation"][stage]
        print(f"[extras {stage}] ordinal c {ov['c_ordinal']:.3f}; stability MAPE "
              f"{out['stability'][stage]['good_outcome']['mape']:.3f} ({time.time() - t0:.0f}s)", flush=True)

    PUBLIC.mkdir(parents=True, exist_ok=True)
    with open(PUBLIC / "extras.json", "w", encoding="utf-8") as fh:
        json.dump(clean_json(out), fh, ensure_ascii=False, separators=(",", ":"))
    write_demo_csv()
    return out


def template_columns() -> list[str]:
    return [f.key for f in features() if f.type != "derived"] + ["mrs90"]


def write_demo_csv() -> None:
    """Synthetic cohort (not real patients) in the upload template, for the in-browser demo."""
    src = SYNTHETIC / "registry_synthetic.xlsx"
    if not src.exists():
        return
    c, _ = clean_registry(load_raw(src))
    c = c[(c["age"] >= 18)].copy()
    c["tici"] = c["tici"].map(lambda v: TICI_LABEL.get(v, "") if pd.notna(v) else "").str.replace(" (not subgraded)", "")
    cols = template_columns()
    df = c[cols].copy()
    for col in cols:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda v: "" if pd.isna(v) else (str(int(v)) if float(v).is_integer() else f"{v:g}"))
    df = df.fillna("")
    df.to_csv(PUBLIC / "demo_synthetic.csv", index=False)
