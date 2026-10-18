"""Model development, validation, model choice and deployment.

For each temporal split (primary: development 2005-2022, validation 2023-2025, as in the 2026 FNO
analysis; sensitivity: development to 2020, validation 2021-2025, about twice the events) and for
every stage x target:

  1. All five learners (tuning.py, learners.py) are tuned on the development patients with the same
     budget on the same folds: regression with splines (the reference), linear regression,
     LightGBM, XGBoost and an MLP.
  2. Temporal validation of every learner and every published or bedside score: AUROC, Brier,
     calibration-in-the-large (CITL), slope and ICI with bootstrap 95% CIs, the paired AUROC
     difference against the reference, ROC, calibration bins, decision curves, classification at
     fixed thresholds, subgroups, and an updating analysis (recalibration on the first validation
     year, evaluated on the later years).

Model choice (primary split, settings.selection): the reference stays unless a candidate meets the
pre-specified rule on the temporal test or, for the admission stage, in nested cross-validation
(nested.py, run first).

Deployment: every publishable learner is refitted on all years with its development settings (the
number of trees re-estimated by cross-validation) and its intercept re-estimated on the pooled
last three years from out-of-fold predictions (temporal recalibration).

Outputs (artifacts/private, never published): train_results.json, temporal_predictions.parquet and
the deployed models (deployed_<stage>_<target>.pkl).
"""
from __future__ import annotations

import json
import pickle
import time
import warnings

import numpy as np
import pandas as pd

from . import learners as Lr
from . import metrics as M
from .clean import POSTERIOR_DETAIL, POSTERIOR_OCCLUSIONS
from .cohort import split_temporal, splits, stage_cohort
from .config import settings, stage_ids, targets
from .features import column_to_feature, design_matrix, monotone_vector
from .scores import benchmark_predictions

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


# ------------------------------------------------------------------------------- helpers
def citl(y, p) -> float:
    return float(M._fit_logistic(np.zeros(len(y)), np.asarray(y, float), offset=M.logit(p), slope=False)[0])


def threshold_table(y, p, min_cell: int) -> list[dict]:
    """Sensitivity, specificity, PPV, NPV, accuracy at fixed thresholds (positive = estimate >= t).
    A threshold is published only if every cell of its 2x2 table holds >= min_cell patients."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    rows = []
    for t in THRESHOLDS:
        pos = p >= t
        tp, fp = int(((y == 1) & pos).sum()), int(((y == 0) & pos).sum())
        fn, tn = int(((y == 1) & ~pos).sum()), int(((y == 0) & ~pos).sum())
        if min(tp, fp, fn, tn) < min_cell:
            rows.append({"t": t, "suppressed": True})
            continue
        rows.append({"t": t, "sens": round(tp / (tp + fn), 3), "spec": round(tn / (tn + fp), 3),
                     "ppv": round(tp / (tp + fp), 3), "npv": round(tn / (tn + fn), 3),
                     "acc": round((tp + tn) / len(y), 3), "pos": round((tp + fp) / len(y), 3)})
    return rows


def _occ_detail(d: pd.DataFrame) -> pd.Series:
    """Detailed occlusion code (circulation of sites the models do not use, e.g. ACA, stays known)."""
    return d["occlusion_detail"] if "occlusion_detail" in d else d["occlusion"]


SUBGROUPS = [
    ("circulation", lambda d: np.where(_occ_detail(d).isin(POSTERIOR_DETAIL + POSTERIOR_OCCLUSIONS), "posterior",
                                       np.where(_occ_detail(d).notna(), "anterior", None))),
    ("occlusion", lambda d: d["occlusion"].map({"M1": "M1", "ICA": "ICA", "tandem": "tandem", "cervical_ICA": "cervical_ICA",
                                                "M2": "M2", "basilar": "posterior", "basilar_vertebral": "posterior",
                                                "other_posterior": "posterior"})),
    ("age", lambda d: np.where(d["age"] >= 80, "80+", np.where(d["age"].notna(), "<80", None))),
    ("sex", lambda d: d["sex_male"].map({1.0: "men", 0.0: "women"})),
    ("ivt", lambda d: d["ivt"].map({1.0: "ivt", 0.0: "no_ivt"})),
    ("route", lambda d: d["transfer"].map({1.0: "transfer", 0.0: "direct"})),
    ("nihss", lambda d: pd.cut(d["nihss_adm"], [-1, 9, 19, 99], labels=["<10", "10-19", "20+"]).astype(object)),
    ("otp", lambda d: np.where(d["onset_to_puncture"] > 360, ">6h", np.where(d["onset_to_puncture"].notna(), "<=6h", None))),
    ("premrs", lambda d: np.where(d["mrs_pre"] >= 1, "1+", np.where(d["mrs_pre"].notna(), "0", None))),
    ("procedure", lambda d: d["technique"].map({"aspiration": "mt", "stent_retriever": "mt", "solumbra": "mt", "combined": "mt",
                                                "none": "other", "diagnostic": "diagnostic"})),
]


def subgroup_table(d: pd.DataFrame, y, p, min_cell: int) -> list[dict]:
    """Observed and predicted outcome, O/E, CITL and AUROC per subgroup (aggregates only).
    Groups under min_cell patients are suppressed; AUROC and CITL need >= min_cell events and
    non-events."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    rows = []
    for dim, fn in SUBGROUPS:
        g = pd.Series(np.asarray(fn(d), dtype=object), index=d.index)
        for level in [v for v in pd.unique(g.dropna())]:
            m = (g == level).to_numpy()
            n = int(m.sum())
            if n < min_cell:
                rows.append({"dim": dim, "group": str(level), "suppressed": True})
                continue
            ev = int(y[m].sum())
            r = {"dim": dim, "group": str(level), "n": n, "observed": float(y[m].mean()), "predicted": float(p[m].mean()),
                 "oe": M.oe_ratio(y[m], p[m])}
            if ev >= min_cell and n - ev >= min_cell:
                r["auroc"] = float(M.roc_auc_score(y[m], p[m]))
                r["citl"] = citl(y[m], p[m])
            else:
                r["events_small"] = True
            rows.append(r)
    return rows


def subgroup_auroc_ci(base: pd.DataFrame, preds: pd.DataFrame, min_cell: int, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap 95% CIs of the subgroup AUROCs (TRIPOD+AI 23a), computed afterwards from the saved
    test predictions (predictions_<split>.parquet) so the training run need not repeat. Keys:
    (stage, target, model) -> {(dim, group): [lo, hi]}; only groups with >= min_cell patients with and
    without the outcome, as in subgroup_table."""
    from scipy.stats import rankdata

    rng = np.random.default_rng(seed)
    d0 = base.set_index("patient_id")
    out = {}
    for (stage, target, model), g in preds.groupby(["stage", "target", "model"], sort=False):
        if model not in Lr.ORDER:
            continue
        d = d0.loc[g["patient_id"].to_numpy()].reset_index()
        y, p = g["y"].to_numpy(float), g["p"].to_numpy(float)
        res = {}
        for dim, fn in SUBGROUPS:
            lv = pd.Series(np.asarray(fn(d), dtype=object))
            for level in pd.unique(lv.dropna()):
                m = (lv == level).to_numpy()
                yy, pp = y[m], p[m]
                n1 = int(yy.sum())
                if n1 < min_cell or len(yy) - n1 < min_cell:
                    continue
                idx = rng.integers(0, len(yy), (n_boot, len(yy)))
                yb, pb = yy[idx], pp[idx]
                n1b = yb.sum(axis=1)
                n0b = yb.shape[1] - n1b
                ok = (n1b > 0) & (n0b > 0)
                r = rankdata(pb, axis=1)
                auc = ((r * yb).sum(axis=1) - n1b * (n1b + 1) / 2) / np.where(ok, n1b * n0b, 1)
                a = auc[ok]
                res[(dim, str(level))] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        out[(stage, target, model)] = res
    return out


def updating(y, p, years, first_year) -> dict | None:
    """Recalibrate on the first validation year, evaluate on the later years (the 2026 analysis's
    updating table): original, intercept updated, intercept and slope updated."""
    years = np.asarray(years)
    a_m, b_m = years == first_year, years > first_year
    if a_m.sum() < 40 or b_m.sum() < 40 or len(np.unique(y[a_m])) < 2 or len(np.unique(y[b_m])) < 2:
        return None
    lp = M.logit(p)
    shift = citl(y[a_m], p[a_m])
    a, b = M._fit_logistic(lp[a_m], y[a_m])
    variants = {"original": p[b_m], "intercept": M.expit(lp[b_m] + shift), "logistic": M.expit(a + b * lp[b_m])}
    out = {"recal_year": int(first_year), "n_recal": int(a_m.sum()), "n_eval": int(b_m.sum())}
    for k, v in variants.items():
        s = M.summary(y[b_m], v)
        out[k] = {m: s[m] for m in ("auroc", "brier", "citl", "slope")}
    return out


def explanation_summary(f: Lr.Fitted, X: pd.DataFrame, stage: str, min_cell: int) -> dict | None:
    """Mean |contribution| per predictor and binned dependence (aggregates only)."""
    res = Lr.contributions(f, X)
    if res is None:
        return None
    phi, base = res
    c2f = column_to_feature(stage)
    grouped = phi.T.groupby(lambda c: c2f[c]).sum().T
    importance = grouped.abs().mean().sort_values(ascending=False)
    dependence = {}
    for key in importance.index:
        if key not in X.columns:
            continue
        x = X[key]
        ok = x.notna()
        if x[ok].nunique() <= 8:
            groups = x[ok].astype(float)
        else:
            try:
                groups = pd.qcut(x[ok], q=10, duplicates="drop")
            except ValueError:
                continue
        rows = []
        for _, idx in groups.groupby(groups, observed=True).groups.items():
            if len(idx) < min_cell:
                continue
            rows.append({"x": float(x.loc[idx].median()), "shap": float(grouped.loc[idx, key].mean()), "n": int(len(idx))})
        dependence[key] = rows
    for feat in {c2f[c] for c in X.columns if "__" in c}:
        rows = []
        for c in [c for c in X.columns if c2f[c] == feat]:
            idx = X.index[X[c] == 1]
            if len(idx) >= min_cell:
                rows.append({"category": c.split("__", 1)[1], "shap": float(grouped.loc[idx, feat].mean()), "n": int(len(idx))})
        dependence[feat] = rows
    return {"importance": {k: float(v) for k, v in importance.items()}, "dependence": dependence, "base_value": float(base)}


# ------------------------------------------------------------------------------- validation
def validate_split(base: pd.DataFrame, test_from: int, seed: int, trials: int, n_boot: int, min_cell: int,
                   min_leaf: int, label: str):
    """Tune every learner on the development years and validate on the later years."""
    out, rows, fitted = {}, [], {}
    for stage in stage_ids():
        d = stage_cohort(base, stage)
        dev, test = split_temporal(d, test_from)
        X_all = design_matrix(d, stage)
        X_dev, X_test = X_all.loc[dev.index], X_all.loc[test.index]
        out[stage] = {"n_all": int(len(d)), "n_dev": int(len(dev)), "n_test": int(len(test)),
                      "columns": list(X_all.columns), "targets": {}}
        for target in targets():
            t0 = time.time()
            y_dev, y_test = dev[target].to_numpy(float), test[target].to_numpy(float)
            mono = monotone_vector(stage, target)
            fits = {}
            for name in Lr.ORDER:
                ts = time.time()
                fits[name] = Lr.tune(name, X_dev, y_dev, seed, trials, mono, min_leaf=min_leaf)
                print(f"    [{label} {stage}/{target}] {name} tuned ({time.time() - ts:.0f}s)", flush=True)
            fitted[(stage, target)] = fits
            preds = {name: f.proba(X_test) for name, f in fits.items()}
            scores = benchmark_predictions(dev, test, stage, target, seed)
            allp = {**preds, **scores}
            temporal = {k: M.summary(y_test, v) for k, v in allp.items()}
            ci = M.bootstrap_ci(y_test, allp, n_boot=n_boot, seed=seed, ref=Lr.REFERENCE)
            for k in temporal:
                temporal[k]["ci"] = ci[k]
                temporal[k]["oe"] = M.oe_ratio(y_test, allp[k])
                temporal[k]["kind"] = "learner" if k in preds else "score"
            nb = 8 if len(y_test) < 250 else 10
            curves = {
                "roc": {k: M.roc_points(y_test, v) for k, v in allp.items()},
                "calibration": {k: M.calibration_bins(y_test, v, n_bins=nb, min_n=min_cell) for k, v in allp.items()},
                "dca": {k: [{"t": r["t"], "nb": r["model"]} for r in M.decision_curve(y_test, v)] for k, v in allp.items()},
                "dca_all": [{"t": r["t"], "nb": r["all"]} for r in M.decision_curve(y_test, preds[Lr.REFERENCE])],
            }
            res = {
                "n_dev": int(len(dev)), "n_test": int(len(test)), "events_dev": int(y_dev.sum()), "events_test": int(y_test.sum()),
                "prevalence_dev": float(y_dev.mean()), "prevalence_test": float(y_test.mean()),
                "models": temporal, "curves": curves,
                "thresholds": {k: threshold_table(y_test, preds[k], min_cell) for k in preds},
                "subgroups": {k: subgroup_table(test, y_test, preds[k], min_cell) for k in preds},
                "updating": {k: updating(y_test, preds[k], test["year"].to_numpy(), test_from) for k in preds},
                "settings": {k: f.settings for k, f in fits.items()},
            }
            out[stage]["targets"][target] = res
            for k, v in allp.items():
                rows.append(pd.DataFrame({"patient_id": test["patient_id"].to_numpy(), "split": label, "stage": stage,
                                          "target": target, "model": k, "y": y_test, "p": v}))
            ref = temporal[Lr.REFERENCE]
            best = max(preds, key=lambda k: temporal[k]["auroc"])
            print(f"[{label} {stage}/{target}] n_dev={len(dev)} n_test={len(test)} ref AUROC={ref['auroc']:.3f} "
                  f"best {best} {temporal[best]['auroc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return out, rows, fitted


# ------------------------------------------------------------------------------- model choice
def select_models(primary: dict, nested: dict | None) -> dict:
    """Apply the pre-specified rule (settings.selection) per stage x target."""
    sel = settings()["selection"]
    ref = sel["reference"]
    out = {}
    for stage, sres in primary.items():
        out[stage] = {}
        for target, tres in sres["targets"].items():
            models = tres["models"]
            nst = ((nested or {}).get(stage) or {}).get(target) if nested else None
            cands = {}
            for name in sel["deployable"]:
                if name == ref or name not in models:
                    continue
                dl = models[name]["ci"].get("delta_auroc_vs_ref")
                a = bool(dl and dl["ci"][0] > 0)
                b = False
                calib_ok = True
                info = {"delta_test": dl, "test_rule": a}
                if nst and name in nst["models"] and ref in nst["models"]:
                    cm, rm = nst["models"][name], nst["models"][ref]
                    b = bool(cm["delta_mean"] >= sel["min_cv_gain"] and cm["share_better"] >= sel["min_cv_share"])
                    calib_ok = bool(cm["abs_citl_mean"] - rm["abs_citl_mean"] <= sel["max_citl_worse"])
                    info.update({"cv_delta_mean": cm["delta_mean"], "cv_share_better": cm["share_better"], "cv_rule": b,
                                 "cv_abs_citl": cm["abs_citl_mean"], "ref_cv_abs_citl": rm["abs_citl_mean"]})
                else:
                    calib_ok = bool(abs(models[name]["citl"]) - abs(models[ref]["citl"]) <= sel["max_citl_worse"])
                info["calibration_ok"] = calib_ok
                info["qualifies"] = bool((a or b) and calib_ok)
                cands[name] = info
            q = [k for k, v in cands.items() if v["qualifies"]]
            chosen = max(q, key=lambda k: models[k]["auroc"]) if q else ref
            out[stage][target] = {"chosen": chosen, "reference": ref, "candidates": cands,
                                  "nested_cv": bool(nst)}
    return out


# ------------------------------------------------------------------------------- deployment
def deploy(base: pd.DataFrame, primary: dict, selection: dict, out_private, seed: int, min_cell: int,
           min_leaf: int) -> dict:
    """Refit every publishable learner on all years with its development settings (from the primary
    split) and recalibrate it to the recent era."""
    recent_from = int(settings()["split"]["recent_era_from_year"])
    out = {}
    for stage in stage_ids():
        d = stage_cohort(base, stage)
        X_all = design_matrix(d, stage)
        recent = (d["year"] >= recent_from).to_numpy()
        out[stage] = {}
        for target in targets():
            t0 = time.time()
            y_all = d[target].to_numpy(float)
            mono = monotone_vector(stage, target)
            chosen = selection[stage][target]["chosen"]
            models, info = {}, {}
            for name in Lr.DEPLOYABLE:
                dev_settings = primary[stage]["targets"][target]["settings"][name]
                f = Lr.refit(name, dev_settings, X_all, y_all, seed, mono, min_leaf=min_leaf, cv_trees=True)
                oof_p = Lr.oof(name, f.settings, X_all, y_all, seed + 1, mono, min_leaf=min_leaf)
                shift = citl(y_all[recent], oof_p[recent])
                p_recent = M.expit(M.logit(oof_p[recent]) + shift)
                info[name] = {
                    "settings": f.settings, "intercept_recent": shift, "n": int(len(y_all)),
                    "n_recent": int(recent.sum()), "events_recent": int(y_all[recent].sum()),
                    "oof_all": M.summary(y_all, oof_p),
                    "oof_recent_uncalibrated": M.summary(y_all[recent], oof_p[recent]),
                    "oof_recent_recalibrated": M.summary(y_all[recent], p_recent),
                    "explain": explanation_summary(f, X_all, stage, min_cell),
                }
                if name == chosen:
                    info[name]["reality_bins"] = M.calibration_bins(
                        y_all[recent], p_recent, n_bins=int(max(4, min(10, recent.sum() // 25))), min_n=min_cell)
                models[name] = {"fitted": f, "shift": shift}
            with open(out_private / f"deployed_{stage}_{target}.pkl", "wb") as fh:
                pickle.dump({"chosen": chosen, "models": models, "columns": list(X_all.columns)}, fh)
            out[stage][target] = {"chosen": chosen, "prevalence_recent": float(y_all[recent].mean()),
                                  "prevalence_all": float(y_all.mean()), "learners": info}
            print(f"[deploy {stage}/{target}] chosen {chosen}; recent shift "
                  f"{info[chosen]['intercept_recent']:+.2f} ({time.time() - t0:.0f}s)", flush=True)
    return out


def load_deployed(out_private, stage: str, target: str) -> dict:
    with open(out_private / f"deployed_{stage}_{target}.pkl", "rb") as fh:
        return pickle.load(fh)


# ------------------------------------------------------------------------------- main routine
def run(base: pd.DataFrame, out_private, quick: bool = False, only: str | None = None) -> dict:
    """Run both splits (or `only` one). Each split writes its own files, so the two can run in
    parallel: train_<split>.json and predictions_<split>.parquet. Model choice and deployment
    follow in choose_and_deploy (python -m mtop deploy)."""
    cfg = settings()
    seed = int(cfg["seed"])
    trials = 6 if quick else int(cfg["tuning"]["trials"])
    n_boot = 200 if quick else int(cfg["bootstrap"]["n"])
    min_leaf = int(cfg["privacy"]["min_leaf"])
    min_cell = int(cfg["privacy"]["min_cell"])
    meta = {"seed": seed, "trials": trials, "bootstrap": n_boot, "quick": quick, "splits": splits(),
            "recent_era_from_year": int(cfg["split"]["recent_era_from_year"]), "reference": Lr.REFERENCE}
    results = {}
    for label, test_from in splits().items():
        if only and label != only:
            continue
        t0 = time.time()
        print(f"== split {label}: validation from {test_from}", flush=True)
        res, rows, fitted = validate_split(base, test_from, seed, trials, n_boot, min_cell, min_leaf, label)
        out = {"meta": meta, "label": label, "test_from_year": test_from, "stages": res}
        out["seconds"] = round(time.time() - t0)
        (out_private / f"train_{label}.json").write_text(json.dumps(out, indent=1))
        pd.concat(rows).to_parquet(out_private / f"predictions_{label}.parquet", index=False)
        results[label] = out
    return results


def choose_and_deploy(base: pd.DataFrame, out_private) -> dict:
    """Model choice by the pre-specified rule (primary split + nested CV if available), then
    deployment of every publishable learner. Writes deployment.json and the deployed models."""
    cfg = settings()
    seed = int(cfg["seed"])
    primary = json.loads((out_private / "train_primary.json").read_text())
    nested_path = out_private / "nested_cv.json"
    nested = json.loads(nested_path.read_text()) if nested_path.exists() else None
    selection = select_models(primary["stages"], nested)
    deployed = deploy(base, primary["stages"], selection, out_private, seed, int(cfg["privacy"]["min_cell"]),
                      int(cfg["privacy"]["min_leaf"]))
    out = {"selection": selection, "deployed": deployed, "nested_cv_used": bool(nested)}
    (out_private / "deployment.json").write_text(json.dumps(out, indent=1))
    return out


def load_results(out_private) -> dict:
    """Both splits' results: {"primary": {...}, "sensitivity": {...}} (whichever exist)."""
    out = {}
    for label in splits():
        p = out_private / f"train_{label}.json"
        if p.exists():
            out[label] = json.loads(p.read_text())
    return out
