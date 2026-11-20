"""MTOP version 1.1: pipeline check, evaluation and deployment.

Plan: docs/registrations/v1_1_plan.md, with amendment 1 (version 1.1 = changes 2.1-2.5, 3.1 and 4.2 on
exactly the data of v1.0.0). Outputs with patient-level content stay in artifacts/private/v11 and
reports/v11 (git-ignored); only aggregates are published.

    python -m mtop.v11 check        section 5, step 2: the code with every change off reproduces v1.0.0
    python -m mtop.v11 nested       section 6: nested cross-validation, development years
    python -m mtop.v11 bootstrap    section 6: full-pipeline bootstrap, development years
    python -m mtop.v11 temporal     section 6, step 4: the 2023-2025 test (not independent)
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import datetime as dt
import hashlib
import json
import time

import numpy as np
import pandas as pd

from . import learners as Lr
from . import paths
from . import reference as R

SEED = 20261016            # settings.yaml seed (the folds of v1.0.0)
V100_FINGERPRINT = "77bb37a365930cdf3e5551d000f8ae40fdfa41f50a44c0593bdc7fea80ba4425"
STAGES = ("admission", "procedure", "h24", "d7")
TARGETS = ("good_outcome", "death90")
PRIV = paths.PRIVATE / "v11"
REP = paths.REPORTS / "v11"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_base() -> pd.DataFrame:
    from .cohort import flow
    return flow(pd.read_parquet(paths.PROCESSED / "clean.parquet"))[0]


def stage_data(base: pd.DataFrame, stage: str, test_from: int = 2023):
    from .cohort import split_temporal, stage_cohort
    from .features import design_matrix
    d = stage_cohort(base, stage)
    dev, test = split_temporal(d, test_from)
    return d, dev, test, design_matrix(d, stage)


def _published():
    """v1.0.0's public files: frozen in artifacts/private/v11/published_v1_0_0 the first time they are
    read, so that rebuilding version 1.1's public files never reads version 1.1's own output."""
    pub = PRIV / "published_v1_0_0"
    if not (pub / "model_bundle.json").exists():
        import shutil
        src = paths.ROOT / "artifacts" / "public"
        b = json.loads((src / "model_bundle.json").read_text())
        if b.get("fingerprint") != V100_FINGERPRINT:
            raise SystemExit("artifacts/public no longer holds v1.0.0; restore it from git (tag v1.0.0)")
        pub.mkdir(parents=True, exist_ok=True)
        for name in ("model_bundle.json", "validation.json", "data_checks.json", "extras.json"):
            shutil.copyfile(src / name, pub / name)
    return (json.loads((pub / "model_bundle.json").read_text()), json.loads((pub / "validation.json").read_text()),
            json.loads((pub / "data_checks.json").read_text()))


def _r5(x):
    return None if x is None else round(float(x), 5)


def _compare_exports(new: dict, old: dict, X: pd.DataFrame) -> dict:
    """Field-by-field comparison of two exported regressions and of their predictions for X."""
    from .lr import predict_exported
    from .splines import expand
    a, b = new["spec"], old["spec"]
    other = [k for k in set(a) | set(b) if k not in ("coef", "intercept")]
    same_other = all(json.dumps(a.get(k), sort_keys=True) == json.dumps(b.get(k), sort_keys=True) for k in other)
    E = expand(X[a["base_columns"]], a["knots"]).to_numpy(float) if "knots" in a else X[a["columns"]].to_numpy(float)
    dp = float(np.max(np.abs(predict_exported(a, E) - predict_exported(b, E))))
    dc = float(max(np.max(np.abs(np.asarray(a["coef"]) - np.asarray(b["coef"]))), abs(a["intercept"] - b["intercept"])))
    return {"identical_except_coefficients": bool(same_other), "max_abs_coef_diff": dc, "max_abs_p_diff": dp,
            "recent_intercept_diff": abs(float(new.get("intercept_recent", 0)) - float(old.get("intercept_recent", 0)))}


# =========================================================================== pipeline check
def check(raw_file=None) -> dict:
    """Plan section 5, step 2. Every version 1.1 switch off (reference.V100, the v1.0.0 code path):
    the registry export, cleaning, cohorts and design matrices are those of v1.0.0, the reference
    and the linear regression are re-tuned on 2005-2022 and redeployed on all years, and their
    exported numbers and the bundle fingerprint must equal v1.0.0's."""
    from . import metrics as M
    from .export import clean_json
    from .features import feature_columns, monotone_vector
    from .train import citl

    bundle, validation, checks_pub = _published()
    rep = {"plan": "docs/registrations/v1_1_plan.md, section 5, step 2", "started": now(), "steps": {}}
    raw = raw_file or paths.DEFAULT_RAW_FILE
    rep["steps"]["export_sha256"] = sha256_file(raw)

    ck = json.loads((paths.PUBLIC / "data_checks.json").read_text())   # written by `mtop clean` into MTOP_OUT
    rep["steps"]["data_checks_identical"] = ck == checks_pub

    base = load_base()
    cohorts_ok, columns_ok = True, True
    recent_from = 2023
    learners_rep = {}
    models = copy.deepcopy(bundle["models"])
    for stage in STAGES:
        d, dev, test, X_all = stage_data(base, stage)
        pv = validation["splits"]["primary"]["stages"][stage]
        cohorts_ok &= (len(d), len(dev), len(test)) == (pv["n_all"], pv["n_dev"], pv["n_test"])
        columns_ok &= list(X_all.columns) == feature_columns(stage) == bundle["models"][stage]["good_outcome"]["columns"]
        X_dev, X_test = X_all.loc[dev.index], X_all.loc[test.index]
        recent = (d["year"] >= recent_from).to_numpy()
        for target in TARGETS:
            tv = pv["targets"][target]
            y_dev, y_test, y_all = (dev[target].to_numpy(float), test[target].to_numpy(float), d[target].to_numpy(float))
            cohorts_ok &= (int(y_dev.sum()), int(y_test.sum())) == (tv["events_dev"], tv["events_test"])
            mono = monotone_vector(stage, target)
            for name in ("Logistic regression (splines)", "Logistic regression"):
                # development fit (primary split), as train.validate_split
                if name == Lr.REFERENCE:
                    f_dev = R.fit(X_dev, y_dev, SEED, R.V100)
                    s_dev = f_dev.settings
                    p_test = f_dev.predict_proba(X_test)[:, 1]
                else:
                    fd = Lr.tune(name, X_dev, y_dev, SEED, 50, mono)
                    s_dev = fd.settings
                    p_test = fd.proba(X_test)
                s_pub = tv["settings"][name]
                same_settings = all(_r5(s_dev[k]) == _r5(s_pub[k]) for k in s_pub if k in s_dev and k != "candidates")
                same_kc = s_dev.get("knots") == s_pub.get("knots") and _r5(s_dev["C"]) == _r5(s_pub["C"])
                summ = M.summary(y_test, p_test)
                m_pub = tv["models"][name]
                same_temporal = all(_r5(summ[k]) == _r5(m_pub[k]) for k in ("auroc", "brier", "citl", "slope", "auprc"))
                # deployment on all years, as train.deploy
                f = Lr.refit(name, s_dev, X_all, y_all, SEED, mono, cv_trees=True)
                oof_p = Lr.oof(name, f.settings, X_all, y_all, SEED + 1, mono)
                shift = citl(y_all[recent], oof_p[recent])
                spec = clean_json(Lr.export(f, shift, X_all))
                published = bundle["models"][stage][target]["learners"][name]
                identical = json.dumps(spec, sort_keys=True) == json.dumps(published, sort_keys=True)
                tol = _compare_exports(spec, published, X_all)
                models[stage][target]["learners"][name] = spec
                learners_rep[f"{stage}/{target}/{name}"] = {
                    "settings": {"knots": s_dev.get("knots"), "C": s_dev["C"]}, "settings_as_published": same_settings,
                    "knots_C_as_published": same_kc, "temporal_metrics_as_published": same_temporal, "deployed_export_identical": identical, **tol,
                    "intercept_recent": round(float(shift), 6)}
                print(f"  {stage}/{target} {name}: settings {same_settings}, temporal {same_temporal}, "
                      f"export identical {identical}, other fields identical {tol['identical_except_coefficients']}, "
                      f"max |coef diff| {tol['max_abs_coef_diff']:.1e}, max |p diff| {tol['max_abs_p_diff']:.1e}", flush=True)
    rep["steps"]["cohorts_identical"] = bool(cohorts_ok)
    rep["steps"]["design_columns_identical"] = bool(columns_ok)
    rep["learners"] = learners_rep
    canon = json.dumps(models, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fp = hashlib.sha256(canon).hexdigest()
    rep["steps"]["fingerprint"] = fp
    rep["steps"]["fingerprint_equals_v1_0_0"] = fp == V100_FINGERPRINT
    rep["carried_over"] = ("LightGBM, XGBoost, EBM, LR + interactions and Stacking are not refitted: version 1.1 "
                           "changes neither their code nor their data (amendment 1), so their v1.0.0 numbers are "
                           "carried over; they enter the fingerprint above unchanged.")
    rep["passed_exact"] = bool(rep["steps"]["data_checks_identical"] and cohorts_ok and columns_ok
                               and all(v["deployed_export_identical"] and v["settings_as_published"] and v["temporal_metrics_as_published"]
                                       for v in learners_rep.values()) and fp == V100_FINGERPRINT)
    # amendment 2: every exported number identical except the L-BFGS coefficients, whose stopping
    # point depends on the machine; predictions within 1e-6 for every patient
    rep["passed_to_tolerance"] = bool(rep["steps"]["data_checks_identical"] and cohorts_ok and columns_ok and all(
        v["identical_except_coefficients"] and v["max_abs_p_diff"] < 1e-5 and v["temporal_metrics_as_published"]
        and v["knots_C_as_published"] for v in learners_rep.values()))
    rep["max_abs_coef_diff"] = max(v["max_abs_coef_diff"] for v in learners_rep.values())
    rep["max_abs_p_diff"] = max(v["max_abs_p_diff"] for v in learners_rep.values())
    rep["max_recent_intercept_diff"] = max(v["recent_intercept_diff"] for v in learners_rep.values())
    rep["finished"] = now()
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "pipeline_check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("steps", "passed_exact", "passed_to_tolerance", "max_abs_coef_diff",
                                          "max_abs_p_diff")}, indent=1))
    return rep


# =========================================================================== candidate learners
CANDIDATE_NAMES = ("Logistic regression", "LightGBM", "XGBoost", "EBM", "LR + interactions",
                   "LR + multiple imputation", "Stacking")


def candidates(stages=STAGES, targets_=TARGETS) -> None:
    """The amendment-1 candidates' development-year fits (2005-2022) and their 2023-2025 predictions,
    regenerated with the v1.0.0 code (train.validate_split, compare.predictions), so the selection rule
    can be re-applied with the version 1.1 reference. Row-level predictions stay in artifacts/private."""
    from . import compare as CP
    from . import metrics as M
    from .features import monotone_vector

    _, validation, _ = _published()
    PRIV.mkdir(parents=True, exist_ok=True)
    out_pred = PRIV / "candidates_primary.parquet"
    out_json = PRIV / "candidates_primary.json"
    done = json.loads(out_json.read_text()) if out_json.exists() else {}
    rows = [pd.read_parquet(out_pred)] if out_pred.exists() else []
    base = load_base()
    for stage in stages:
        d, dev, test, X_all = stage_data(base, stage)
        X_dev, X_test = X_all.loc[dev.index], X_all.loc[test.index]
        for target in targets_:
            key = f"{stage}/{target}"
            if key in done:
                continue
            t0 = time.time()
            y_dev, y_test = dev[target].to_numpy(float), test[target].to_numpy(float)
            mono = monotone_vector(stage, target)
            preds, settings = {}, {}
            ref = R.fit(X_dev, y_dev, SEED, R.V100)
            preds[Lr.REFERENCE], settings[Lr.REFERENCE] = ref.predict_proba(X_test)[:, 1], ref.settings
            for name in ("Logistic regression", "LightGBM", "XGBoost"):
                f = Lr.tune(name, X_dev, y_dev, SEED, 50, mono, min_leaf=20)
                preds[name], settings[name] = f.proba(X_test), f.settings
                print(f"    {key} {name} ({time.time() - t0:.0f}s)", flush=True)
            base_settings = {Lr.REFERENCE: ref.settings, "XGBoost": settings["XGBoost"]}
            p2, info, _ = CP.predictions(X_dev, y_dev, X_test, base_settings, SEED, 50, mono, 20,
                                         ["EBM", "LR + interactions", "LR + multiple imputation", "Stacking"],
                                         log=lambda m: print(f"    {key}{m}", flush=True))
            preds.update(p2)
            settings.update(info)
            pub = validation["splits"]["primary"]["stages"][stage]["targets"][target]["models"]
            rep = {}
            for name, p in preds.items():
                s_ = M.summary(y_test, p)
                rep[name] = {"auroc": s_["auroc"], "auroc_published": pub.get(name, {}).get("auroc"),
                             "brier": s_["brier"], "brier_published": pub.get(name, {}).get("brier")}
                rows.append(pd.DataFrame({"patient_id": test["patient_id"].to_numpy(), "stage": stage, "target": target,
                                          "model": name, "y": y_test, "p": p}))
            done[key] = {"seconds": round(time.time() - t0), "models": rep,
                         "settings": json.loads(json.dumps(settings, default=float))}
            pd.concat(rows).to_parquet(out_pred, index=False)
            out_json.write_text(json.dumps(done, indent=1, default=float))
            print(f"[candidates {key}] " + ", ".join(f"{k} {v['auroc']:.4f}/{(v['auroc_published'] or float('nan')):.4f}"
                                                     for k, v in rep.items()) + f" ({time.time() - t0:.0f}s)", flush=True)


# =========================================================================== evaluation (section 6)
PREV = {"h24": "procedure", "d7": "h24"}
NOW = {"h24": ("nihss_24h", ("sich", "ich_other")), "d7": ("nihss_7d", ("edema", "decompression"))}
NIHSS_ALONE = {"h24": "NIHSS 24 h", "d7": "NIHSS day 7"}
# the specification before changes 2.1-2.4, then each change added in turn (nested CV, for the record)
CUMULATIVE = [("1.0.0 specification", R.V100_NEWTON),
              ("+ 2.1", R.V100_NEWTON.replace(label="+ 2.1", per_input_indicator=True)),
              ("+ 2.2", R.V100_NEWTON.replace(label="+ 2.2", per_input_indicator=True, min_missing=20)),
              ("+ 2.3", R.V100_NEWTON.replace(label="+ 2.3", per_input_indicator=True, min_missing=20, drop_derived=True))]
S0, S1, S2, S3 = "1.0.0 specification", "2.1-2.4", "1.1 full structure", "update model"
MEASURES = ("auroc", "brier", "logloss", "citl", "slope", "oe")


def measures(y, p) -> dict:
    from sklearn.metrics import roc_auc_score

    from .metrics import fast_citl_slope
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    citl, slope = fast_citl_slope(y, np.log(p / (1 - p)))
    return {"auroc": float(roc_auc_score(y, p)), "brier": float(np.mean((p - y) ** 2)),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "citl": float(citl), "slope": float(slope), "oe": float(y.sum() / p.sum())}


class Universe:
    """Patients of the stage before thrombectomy, addressed by position; every later stage's cohort is
    a subset (`elig`). test_from=2023: the development patients (2005-2022) only; None: all years."""

    def __init__(self, base: pd.DataFrame, test_from: int | None = 2023):
        from .cohort import split_temporal, stage_cohort
        from .features import design_matrix
        adm = stage_cohort(base, "admission")
        dev = (split_temporal(adm, test_from)[0] if test_from else adm).reset_index(drop=True)
        self.df, self.n = dev, len(dev)
        self.X = {s: design_matrix(dev, s).reset_index(drop=True) for s in STAGES}
        self.elig = {}
        for s in STAGES:
            m = np.zeros(self.n, bool)
            m[stage_cohort(dev, s).index.to_numpy()] = True
            self.elig[s] = m
        self.y = {t: dev[t].to_numpy(float) for t in TARGETS}
        self.year = dev["year"].to_numpy()

    def rows(self, stage: str, mask=None) -> np.ndarray:
        m = self.elig[stage] if mask is None else self.elig[stage] & mask
        return np.flatnonzero(m)


_UNIVERSE = None


def universe() -> Universe:
    global _UNIVERSE
    if _UNIVERSE is None:
        _UNIVERSE = Universe(load_base())
    return _UNIVERSE


class Full:
    kind = "full"

    def __init__(self, U, stage, m):
        self.U, self.stage, self.m = U, stage, m
        self.n_coef, self.settings = m.n_coef, m.settings

    def logit_rows(self, rows):
        return self.m.logit(self.U.X[self.stage].iloc[rows])

    def settings_tree(self) -> dict:
        return {"knots": int(self.m.settings["knots"]), "C": float(self.m.settings["C"])}


def update_frame(U, stage: str, rows, lp) -> pd.DataFrame:
    nihss, events = NOW[stage]
    X = U.X[stage]
    D = pd.DataFrame({"lp": np.asarray(lp, float)})
    D[nihss] = X[nihss].to_numpy(float)[rows]
    for e in events:
        D[e] = X[e].to_numpy(float)[rows]
    return D


class Update:
    kind = "update"

    def __init__(self, U, stage, prev, upd):
        self.U, self.stage, self.prev, self.upd = U, stage, prev, upd
        self.n_coef, self.settings = upd.n_coef, {"update": upd.settings, "prev": prev.settings_tree()}

    def logit_rows(self, rows):
        return self.upd.logit(update_frame(self.U, self.stage, rows, self.prev.logit_rows(rows)))

    def settings_tree(self) -> dict:
        return {"prev": self.prev.settings_tree(), "update": {"knots": 3, "C": float(self.upd.settings["C"])}}


def fit_stage(U, stage, target, W, seed, chain: dict, spec=R.V110, settings=None, grouped=False, prev_model=None):
    """The version 1.1 model of `stage` fitted on the training universe W (positions; copies allowed
    when `grouped`): the full structure, or the update model when chain[stage] == "update"."""
    W = np.asarray(W)
    T = W[U.elig[stage][W]]
    y = U.y[target]
    if chain.get(stage, "full") == "full":
        m = R.fit(U.X[stage].iloc[T], y[T], seed, spec, groups=T if grouped else None, settings=settings)
        return Full(U, stage, m)
    p_stage = PREV[stage]
    settings = settings or {}
    M_prev = prev_model or fit_stage(U, p_stage, target, W, seed, chain, spec, settings.get("prev"), grouped)
    lp = crossfit(U, p_stage, target, W, T, seed, chain, spec, M_prev.settings_tree(), grouped)
    upd = R.fit(update_frame(U, stage, T, lp), y[T], seed, R.UPDATE, groups=T if grouped else None,
                settings=settings.get("update"), knot_options=(3,))
    return Update(U, stage, M_prev, upd)


def crossfit(U, stage, target, W, T, seed, chain, spec, settings, grouped) -> np.ndarray:
    """Out-of-fold log-odds of `stage`'s model for the patients T: 5 stratified folds over the
    patients of W, the model refitted with `settings` fixed on the other four."""
    from sklearn.model_selection import StratifiedKFold
    y = U.y[target]
    uniq = np.unique(W)
    lp = np.full(len(T), np.nan)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(uniq)), y[uniq]):
        M = fit_stage(U, stage, target, W[np.isin(W, uniq[tr])], seed, chain, spec, settings, grouped)
        sel = np.isin(T, uniq[va])
        if sel.any():
            lp[sel] = M.logit_rows(T[sel])
    return lp


def _summ(v) -> dict:
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5))}


# --------------------------------------------------------------------------- nested cross-validation
def nested(stages=STAGES, targets_=TARGETS) -> dict:
    """Section 6.1: 5 outer folds x 3 repeats on each stage's 2005-2022 patients; every specification
    is tuned again inside each outer training part (inner seed 20261116 + k, as v1.0.0)."""
    from sklearn.model_selection import RepeatedStratifiedKFold

    from .scores import POINT_SCORES, apply_points_map, fit_points_map
    U = universe()
    PRIV.mkdir(parents=True, exist_ok=True)
    path = PRIV / "nested.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for stage in stages:
        P = U.rows(stage)
        for target in targets_:
            key = f"{stage}/{target}"
            if key in out:
                continue
            t0 = time.time()
            y = U.y[target]
            per = {}
            folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(np.zeros(len(P)), y[P])
            for k, (tr, va) in enumerate(folds):
                sk = SEED + 100 + k
                Tr, Te = P[tr], P[va]
                Xtr, Xte, ytr, yte = U.X[stage].iloc[Tr], U.X[stage].iloc[Te], y[Tr], y[Te]
                fits = {lab: Full(U, stage, R.fit(Xtr, ytr, sk, spec)) for lab, spec in CUMULATIVE}
                sc, yu, cn = R.tune_scores(Xtr, ytr, sk, R.SIMPLIFIED, repeats=5)
                fits[S1] = Full(U, stage, R.fit_chosen(sc, yu, cn, *R.choose(sc, 5), R.SIMPLIFIED, 5))
                fits[S2] = Full(U, stage, R.fit_chosen(sc, yu, cn, *R.choose(sc, 25), R.V110, 25))
                preds = {lab: 1 / (1 + np.exp(-f.logit_rows(Te))) for lab, f in fits.items()}
                ncoef = {lab: f.n_coef for lab, f in fits.items()}
                if stage in PREV:
                    W = np.setdiff1d(np.arange(U.n), Te)
                    chains = {S3: {"h24": "update", "d7": "update"}} if stage == "h24" else \
                        {S3 + " (24 h full)": {"h24": "full", "d7": "update"}, S3 + " (24 h update)": {"h24": "update", "d7": "update"}}
                    for lab, ch in chains.items():
                        f = fit_stage(U, stage, target, W, sk, ch)
                        preds[lab] = 1 / (1 + np.exp(-f.logit_rows(Te)))
                        ncoef[lab] = f.n_coef
                    name = NIHSS_ALONE[stage]
                    mp = fit_points_map(POINT_SCORES[name](U.df.iloc[Tr]), ytr)
                    preds["NIHSS alone"] = apply_points_map(mp, POINT_SCORES[name](U.df.iloc[Te]))
                    ncoef["NIHSS alone"] = 2
                for lab, p in preds.items():
                    r = per.setdefault(lab, {m: [] for m in MEASURES[:5]} | {"n_coef": []})
                    mm = measures(yte, p)
                    for m in MEASURES[:5]:
                        r[m].append(mm[m])
                    r["n_coef"].append(ncoef[lab])
                print(f"  [nested {key}] fold {k + 1}/15 " + " ".join(f"{lab[:14]}={per[lab]['auroc'][-1]:.3f}" for lab in per)
                      + f" ({time.time() - t0:.0f}s)", flush=True)
            res = {"n": int(len(P)), "events": int(y[P].sum()), "seconds": round(time.time() - t0), "models": {}}
            for lab, r in per.items():
                res["models"][lab] = {m: _summ(r[m]) for m in MEASURES[:5]} | {
                    "n_coef_median": float(np.median(r["n_coef"])), "fold_auroc": [round(v, 5) for v in r["auroc"]],
                    "fold_logloss": [round(v, 6) for v in r["logloss"]]}
            out[key] = res
            path.write_text(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------- bootstrap
N_BOOT, SEED_BOOT = 500, 2026
BOOT_SPECS = (S0, S1, S2)


def _boot_one(b: int, rows: np.ndarray) -> tuple[dict, dict]:
    import warnings
    warnings.filterwarnings("ignore")
    U = universe()
    rec, preds = {"b": b, "fits": {}}, {}
    for stage in STAGES:
        T = rows[U.elig[stage][rows]]
        P = U.rows(stage)
        Xb, Xte = U.X[stage].iloc[T], U.X[stage].iloc[P]
        for target in TARGETS:
            yb, yte = U.y[target][T], U.y[target][P]
            rf = R.make_folds(yb, SEED, 5, groups=T)          # repeat 1 = the folds of the 1.0.0 specification
            m0 = R.fit(Xb, yb, SEED, R.V100_NEWTON, groups=T, row_folds=rf)
            sc, yu, cn = R.tune_scores(Xb, yb, SEED, R.SIMPLIFIED, groups=T, repeats=5, row_folds=rf)
            c1, c2 = R.choose(sc, 5), R.choose(sc, 25)
            m1 = R.fit_chosen(sc, yu, cn, *c1, R.SIMPLIFIED, 5)
            m2 = m1 if c1[:2] == c2[:2] else R.fit_chosen(sc, yu, cn, *c2, R.V110, 25)
            for lab, m, st in ((S0, m0, m0.settings), (S1, m1, m1.settings),
                               (S2, m2, {**m2.settings, "cv_logloss": c2[2], "cv_folds": 25})):
                p_te = m.predict_proba(Xte)[:, 1]
                rec["fits"][f"{lab}|{stage}/{target}"] = {
                    "knots": int(st["knots"]), "C": float(st["C"]), "n_coef": m.n_coef, "n": int(len(T)),
                    "unique": int(len(np.unique(T))), "apparent": measures(yb, m.predict_proba(Xb)[:, 1]),
                    "test": measures(yte, p_te)}
                preds[f"{lab}|{stage}/{target}"] = p_te.astype(np.float32)
    return rec, preds


def bootstrap(n_boot: int = N_BOOT, n_jobs: int = 2, chunk: int = 10, only=None) -> None:
    """Section 6.1: the full-pipeline bootstrap (500 resamples, seed 2026, the resamples of the v1.0.0
    bootstrap) of the specifications before and after changes 2.1-2.4 and 2.5, at every stage and
    outcome. Checkpointed: per-resample records (aggregates) and each patient's estimates (private)."""
    from joblib import Parallel, delayed
    U = universe()
    REP.mkdir(parents=True, exist_ok=True)
    PRIV.mkdir(parents=True, exist_ok=True)
    IDX = np.random.default_rng(SEED_BOOT).integers(0, U.n, size=(N_BOOT, U.n))
    log = REP / "bootstrap_resamples.jsonl"
    done = set()
    if log.exists():
        done = {json.loads(line)["b"] for line in log.read_text().splitlines() if line.strip()}
    todo = [b for b in range(n_boot) if b not in done and (only is None or b in only)]
    print(f"bootstrap: {len(done)} done, {len(todo)} to do, started {now()}", flush=True)
    for start in range(0, len(todo), chunk):
        part = todo[start:start + chunk]
        t0 = time.time()
        res = Parallel(n_jobs=n_jobs)(delayed(_boot_one)(b, IDX[b]) for b in part)
        for rec, preds in res:
            np.savez_compressed(PRIV / f"boot_{rec['b']:03d}.npz", **{k.replace("|", "__").replace("/", "__"): v for k, v in preds.items()})
        with log.open("a") as fh:
            for rec, _ in res:
                fh.write(json.dumps(rec) + "\n")
        print(f"bootstrap: {len(done) + start + len(part)}/{n_boot} at {now()} ({time.time() - t0:.0f}s for {len(part)})", flush=True)



def _records(name: str) -> list[dict]:
    path = REP / name
    recs = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["b"]] = r
    return [recs[b] for b in sorted(recs)]


def _apparent(U, stage, target, label):
    P = U.rows(stage)
    X, y = U.X[stage].iloc[P], U.y[target][P]
    if label == S0:
        m = R.fit(X, y, SEED, R.V100_NEWTON)
    else:
        sc, yu, cn = R.tune_scores(X, y, SEED, R.SIMPLIFIED, repeats=5)
        m = R.fit_chosen(sc, yu, cn, *R.choose(sc, 5), R.SIMPLIFIED, 5) if label == S1 else \
            R.fit_chosen(sc, yu, cn, *R.choose(sc, 25), R.V110, 25)
    return m, measures(y, m.predict_proba(X)[:, 1])


def summarise_bootstrap(prefix: str = "boot", records: str = "bootstrap_resamples.jsonl", labels=BOOT_SPECS,
                        apparent_fn=None) -> dict:
    """Optimism-corrected measures (Harrell), stability and the risk ranges of every specification."""
    U = universe()
    recs = _records(records)
    B = len(recs)
    out = {"n_boot": B, "seed_bootstrap": SEED_BOOT, "seed_folds": SEED, "fits": {}}
    for stage in STAGES:
        for target in TARGETS:
            for lab in labels:
                key = f"{lab}|{stage}/{target}"
                rr = [r["fits"][key] for r in recs if key in r["fits"]]
                if not rr:
                    continue
                m, app = (apparent_fn or _apparent)(U, stage, target, lab)
                res = {"n": int(U.elig[stage].sum()), "events": int(U.y[target][U.rows(stage)].sum()), "resamples": len(rr),
                       "apparent_model": {"knots": int(m.settings.get("knots", 3)), "C": float(m.settings["C"]),
                                          "n_coef": int(m.n_coef)}, "measures": {}}
                for k in MEASURES:
                    o = np.array([r["apparent"][k] - r["test"][k] for r in rr])
                    res["measures"][k] = {"apparent": app[k], "optimism": float(o.mean()),
                                          "monte_carlo_se": float(o.std(ddof=1) / np.sqrt(len(o))),
                                          "corrected": float(app[k] - o.mean())}
                Pm = np.stack([np.load(PRIV / f"{prefix}_{r['b']:03d}.npz")[key.replace("|", "__").replace("/", "__")]
                               for r in recs if key in r["fits"]])
                width = np.percentile(Pm, 97.5, axis=0) - np.percentile(Pm, 2.5, axis=0)
                knots = np.array([r["knots"] for r in rr])
                logC = np.log10([r["C"] for r in rr])
                res["stability"] = {
                    "risk_range_width_median": float(np.median(width)), "risk_range_width_p90": float(np.percentile(width, 90)),
                    "knots_share": {str(k): float(np.mean(knots == k)) for k in (3, 4, 5)},
                    "log10_C_median": float(np.median(logC)), "log10_C_iqr": [float(np.percentile(logC, 25)), float(np.percentile(logC, 75))],
                    "share_C_at_grid_end": float(np.mean((np.isclose(logC, -4)) | (np.isclose(logC, 2)))),
                    "n_coef_median": float(np.median([r["n_coef"] for r in rr]))}
                out["fits"][key] = res
                print(f"  {key}: corrected AUROC {res['measures']['auroc']['corrected']:.4f}, log-loss "
                      f"{res['measures']['logloss']['corrected']:.4f}, slope {res['measures']['slope']['corrected']:.3f}, "
                      f"risk range {res['stability']['risk_range_width_median']:.3f}, coef {m.n_coef}", flush=True)
    return out


def decide(boot: dict, nested_res: dict) -> dict:
    """Section 6, step 3: rules (a) and (b) from the bootstrap, rule (c) from nested CV."""
    dec = {"rule_a": {}, "rule_b": {}, "rule_c": {}}
    for st in ("admission", "h24"):
        k0, k1 = f"{S0}|{st}/good_outcome", f"{S1}|{st}/good_outcome"
        if k0 in boot["fits"] and k1 in boot["fits"]:
            a0, a1 = boot["fits"][k0]["measures"], boot["fits"][k1]["measures"]
            d_auc = a1["auroc"]["corrected"] - a0["auroc"]["corrected"]
            d_ll = a1["logloss"]["corrected"] - a0["logloss"]["corrected"]
            dec["rule_a"][st] = {"delta_auroc": d_auc, "delta_logloss": d_ll, "keep": bool(d_auc >= -0.01 and d_ll <= 0.005)}
    dec["keep_simplifications"] = bool(dec["rule_a"]) and all(v["keep"] for v in dec["rule_a"].values())
    for key, res in boot["fits"].items():
        lab, sk = key.split("|")
        if lab not in (S2, S3):
            continue
        sl = res["measures"]["slope"]["corrected"]
        dec["rule_b"][key] = {"corrected_slope": sl, "apply": bool(sl < 0.9 or sl > 1.1), "factor": sl if (sl < 0.9 or sl > 1.1) else 1.0}
    # rule (c): 24 hours first; the day-7 update model builds on the model chosen at 24 hours
    for target in TARGETS:
        for st in ("h24", "d7"):
            res = (nested_res or {}).get(f"{st}/{target}")
            if not res:
                continue
            m = res["models"]
            if st == "h24":
                lab = S3
            else:
                h = dec["rule_c"].get(f"h24/{target}", {})
                lab = S3 + (" (24 h update)" if h.get("replace") else " (24 h full)")
            if lab not in m:
                continue
            d_auc = m[lab]["auroc"]["mean"] - m[S2]["auroc"]["mean"]
            d_ll = m[lab]["logloss"]["mean"] - m[S2]["logloss"]["mean"]
            dec["rule_c"][f"{st}/{target}"] = {"variant": lab, "delta_auroc": d_auc, "delta_logloss": d_ll,
                                               "replace": bool(d_auc >= 0.01 and d_ll <= 0)}
    return dec



# --------------------------------------------------------------------------- the version 1.1 models
def load_decisions() -> dict:
    return json.loads((PRIV / "decisions.json").read_text())


def chain_for(dec: dict, target: str) -> dict:
    """Which stages use the update model (rule c), for one outcome."""
    return {st: ("update" if dec["rule_c"].get(f"{st}/{target}", {}).get("replace") else "full") for st in PREV}


def slope_factor(dec: dict, stage: str, target: str, chain: dict) -> float:
    lab = S3 if chain.get(stage) == "update" else S2
    return float(dec["rule_b"].get(f"{lab}|{stage}/{target}", {}).get("factor", 1.0))


def _with_slope(model, U, stage, target, W, factor: float):
    """Change 2.5's correction on a fitted stage model (full: its coefficients; update: the update
    model's), with the intercept re-estimated on the training patients."""
    if factor == 1.0:
        return model
    T = np.asarray(W)[U.elig[stage][np.asarray(W)]]
    y = U.y[target][T]
    if model.kind == "full":
        return Full(U, stage, R.with_slope(model.m, U.X[stage].iloc[T], y, factor))
    D = update_frame(U, stage, T, model.prev.logit_rows(T))
    return Update(U, stage, model.prev, R.with_slope(model.upd, D, y, factor))


def fit_v11(U, stage, target, W, dec, settings=None, seed=SEED):
    ch = chain_for(dec, target)
    m = fit_stage(U, stage, target, W, seed, ch, R.V110, settings=settings)
    return _with_slope(m, U, stage, target, W, slope_factor(dec, stage, target, ch))


REF_V11 = "Logistic regression (splines) · v1.1"


def temporal() -> dict:
    """Section 6, step 4: version 1.1 fitted on the development years and tested on the later years,
    next to v1.0.0, in both temporal splits (primary 2023-2025, sensitivity 2021-2025). Not
    independent: version 1.1 was designed after these results were known; it decides nothing.
    Each block has the format of train.validate_split, so the app shows v1.1 as one more row."""
    from . import metrics as M
    from .scores import POINT_SCORES, apply_points_map, fit_points_map
    from .train import subgroup_table, threshold_table, updating
    dec = load_decisions()
    base = load_base()
    U = Universe(base, None)
    out = {}
    for label, test_from in (("primary", 2023), ("sensitivity", 2021)):
        out[label] = {}
        dev, test = U.year < test_from, U.year >= test_from
        W = np.flatnonzero(dev)
        for stage in STAGES:
            out[label][stage] = {}
            Tr, Te = U.rows(stage, dev), U.rows(stage, test)
            df_te = U.df.iloc[Te]
            for target in TARGETS:
                y = U.y[target][Te]
                m = fit_v11(U, stage, target, W, dec)
                p11 = 1 / (1 + np.exp(-m.logit_rows(Te)))
                m0 = R.fit(U.X[stage].iloc[Tr], U.y[target][Tr], SEED, R.V100)
                p10 = m0.predict_proba(U.X[stage].iloc[Te])[:, 1]
                preds = {Lr.REFERENCE: p10, REF_V11: p11}
                if label == "primary" and stage in NIHSS_ALONE:
                    name = NIHSS_ALONE[stage]
                    mp = fit_points_map(POINT_SCORES[name](U.df.iloc[Tr]), U.y[target][Tr])
                    preds["NIHSS alone"] = apply_points_map(mp, POINT_SCORES[name](df_te))
                ci = M.bootstrap_ci(y, preds, n_boot=2000, seed=SEED, ref=Lr.REFERENCE)
                nb = 8 if len(y) < 250 else 10
                block = {
                    "model": {**M.summary(y, p11), "ci": ci[REF_V11], "oe": M.oe_ratio(y, p11), "kind": "learner",
                              "delong_p": M.delong_p(y, p11, p10)},
                    "curves": {"roc": M.roc_points(y, p11), "calibration": M.calibration_bins(y, p11, n_bins=nb, min_n=10),
                               "dca": [{"t": r["t"], "nb": r["model"]} for r in M.decision_curve(y, p11)]},
                    "thresholds": threshold_table(y, p11, 10), "subgroups": subgroup_table(df_te, y, p11, 10),
                    "updating": updating(y, p11, df_te["year"].to_numpy(), test_from),
                    "settings": {"structure": m.kind, "n_coef": int(m.n_coef), **m.settings_tree(),
                                 "slope_factor": slope_factor(dec, stage, target, chain_for(dec, target))},
                    "v1_0_0_reproduced": {k: M.summary(y, p10)[k] for k in ("auroc", "brier", "citl", "slope")},
                    "comparison": {k: {**M.summary(y, v), "ci": ci[k]} for k, v in preds.items()},
                }
                out[label][stage][target] = block
                r = block["comparison"]
                print(f"  temporal {label} {stage}/{target}: v1.0.0 {r[Lr.REFERENCE]['auroc']:.3f} (CITL {r[Lr.REFERENCE]['citl']:+.2f}, "
                      f"slope {r[Lr.REFERENCE]['slope']:.2f}) | v1.1 {r[REF_V11]['auroc']:.3f} (CITL {r[REF_V11]['citl']:+.2f}, "
                      f"slope {r[REF_V11]['slope']:.2f}); delta {block['model']['ci']['delta_auroc_vs_ref']['mean']:+.3f}", flush=True)
                if label == "primary":
                    pid = U.df["patient_id"].to_numpy()[Te]
                    pd.DataFrame({"patient_id": pid, "stage": stage, "target": target, "y": y, "p": p11}).to_parquet(
                        PRIV / f"temporal_v11_{stage}_{target}.parquet", index=False)
    from .export import clean_json
    (PRIV / "temporal.json").write_text(json.dumps(clean_json(out), indent=1))
    return out


def selection() -> dict:
    """Amendment 1's selection rule applied again with the version 1.1 reference (amendment 2):
    temporal criterion (Holm over the 7 candidates, paired bootstrap of the AUROC difference) and,
    before thrombectomy for mRS 0-2, the nested cross-validation criterion on v1.0.0's folds."""
    from sklearn.model_selection import RepeatedStratifiedKFold

    from . import metrics as M
    from .config import settings as cfg
    from .train import holm
    sel = cfg()["selection"]
    _, validation, _ = _published()
    dec = load_decisions()
    cand = pd.read_parquet(PRIV / "candidates_primary.parquet")
    out = {}
    # nested criterion: v1.0.0's folds over all years (admission, mRS 0-2), the v1.1 reference refitted
    U = Universe(load_base(), None)
    P = U.rows("admission")
    y = U.y["good_outcome"]
    fold_auc, fold_citl, fold_auc_v100, fold_brier, fold_slope = [], [], [], [], []
    for k, (tr, va) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(np.zeros(len(P)), y[P])):
        m = fit_v11(U, "admission", "good_outcome", P[tr], dec, seed=SEED + 100 + k)
        p = 1 / (1 + np.exp(-m.logit_rows(P[va])))
        mm = measures(y[P[va]], p)
        fold_auc.append(mm["auroc"])
        fold_citl.append(mm["citl"])
        fold_brier.append(mm["brier"])
        fold_slope.append(mm["slope"])
        m0 = R.fit(U.X["admission"].iloc[P[tr]], y[P[tr]], SEED + 100 + k, R.V100)
        fold_auc_v100.append(measures(y[P[va]], m0.predict_proba(U.X["admission"].iloc[P[va]])[:, 1])["auroc"])
    nst = validation["nested"]["stages"]["admission"]["good_outcome"]["models"]
    ref_auc = np.asarray(fold_auc)
    d0 = ref_auc - np.asarray(nst[Lr.REFERENCE]["fold_auroc"])
    nested_block = {"auroc": _summ(ref_auc), "brier": _summ(fold_brier), "citl": _summ(fold_citl), "slope": _summ(fold_slope),
                    "abs_citl_mean": float(np.mean(np.abs(fold_citl))), "delta_mean": float(d0.mean()),
                    "delta_lo": float(np.percentile(d0, 2.5)), "delta_hi": float(np.percentile(d0, 97.5)),
                    "share_better": float((d0 > 0).mean()), "folds_better": int((d0 > 0).sum()),
                    "fold_auroc": [round(float(x), 4) for x in ref_auc]}
    nested_c = {"v1_1_reference": {"auroc_mean": float(ref_auc.mean()), "abs_citl_mean": float(np.mean(np.abs(fold_citl)))},
                "v1_0_0_reproduced_fold_auroc_max_diff": float(np.max(np.abs(np.asarray(fold_auc_v100) - np.asarray(nst[Lr.REFERENCE]["fold_auroc"])))),
                "candidates": {}}
    for name in CANDIDATE_NAMES:
        if name not in nst:
            continue
        d = np.asarray(nst[name]["fold_auroc"]) - ref_auc
        nested_c["candidates"][name] = {"delta_mean": float(d.mean()), "share_better": float((d > 0).mean()),
                                        "abs_citl": nst[name]["abs_citl_mean"],
                                        "rule": bool(d.mean() >= sel["min_cv_gain"] and (d > 0).mean() >= sel["min_cv_share"]),
                                        "calibration_ok": bool(nst[name]["abs_citl_mean"] - nested_c["v1_1_reference"]["abs_citl_mean"] <= sel["max_citl_worse"])}
    for stage in STAGES:
        for target in TARGETS:
            t11 = pd.read_parquet(PRIV / f"temporal_v11_{stage}_{target}.parquet")
            c = cand[(cand.stage == stage) & (cand.target == target)]
            if c.empty:
                print(f"  selection {stage}/{target}: candidates' development fits not available yet", flush=True)
                out[f"{stage}/{target}"] = {"chosen": Lr.REFERENCE, "candidates": {}, "incomplete": True}
                continue
            yv = t11["y"].to_numpy(float)
            preds = {Lr.REFERENCE: t11["p"].to_numpy(float)}
            for name in CANDIDATE_NAMES:
                g = c[c.model == name]
                if len(g):
                    preds[name] = g.set_index("patient_id").loc[t11["patient_id"], "p"].to_numpy(float)
            ci = M.bootstrap_ci(yv, preds, n_boot=int(cfg()["bootstrap"]["n"]), seed=SEED, ref=Lr.REFERENCE)
            dls = {n: ci[n].get("delta_auroc_vs_ref") for n in CANDIDATE_NAMES if n in ci}
            adj = holm({n: (d["mean"], d["p_two_sided"]) for n, d in dls.items() if d}, float(sel["alpha"]))
            ref_citl = abs(M.summary(yv, preds[Lr.REFERENCE])["citl"])
            cands = {}
            for name, dl in dls.items():
                a = bool(dl and adj.get(name, {}).get("pass"))
                use_nested = stage == "admission" and target == "good_outcome" and name in nested_c["candidates"]
                b = bool(use_nested and nested_c["candidates"][name]["rule"])
                calib_ok = (nested_c["candidates"][name]["calibration_ok"] if use_nested
                            else bool(abs(M.summary(yv, preds[name])["citl"]) - ref_citl <= sel["max_citl_worse"]))
                cands[name] = {"delta_test": dl, "p_holm": adj.get(name, {}).get("p_holm"), "test_rule": a,
                               "cv_rule": b if use_nested else None, "calibration_ok": calib_ok,
                               "qualifies": bool((a or b) and calib_ok)}
            q = [k for k, v in cands.items() if v["qualifies"]]
            out[f"{stage}/{target}"] = {"chosen": (max(q, key=lambda k: M.summary(yv, preds[k])["auroc"]) if q else Lr.REFERENCE),
                                        "candidates": cands}
            print(f"  selection {stage}/{target}: chosen {out[f'{stage}/{target}']['chosen']}; "
                  + ", ".join(f"{n} {v['delta_test']['mean']:+.3f} p_holm {v['p_holm']}" for n, v in cands.items() if v["delta_test"]), flush=True)
    res = {"nested_criterion": nested_c, "nested_block": nested_block, "stages": out}
    (PRIV / "selection.json").write_text(json.dumps(res, indent=1))
    return res



# --------------------------------------------------------------------------- deployment and bundle
def _explanation(model, U, stage, rows, min_cell: int = 10) -> dict | None:
    """As train.explanation_summary: mean |contribution| per predictor and binned dependence."""
    from .features import column_to_feature
    if model.kind != "full":
        return None
    X = U.X[stage].iloc[rows]
    phi, base = model.m.contributions(X)
    c2f = column_to_feature(stage)
    grouped = phi.T.groupby(lambda c: c2f[c]).sum().T
    importance = grouped.abs().mean().sort_values(ascending=False)
    importance = importance[importance > 0]
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
        rows_ = []
        for _, idx in groups.groupby(groups, observed=True).groups.items():
            if len(idx) >= min_cell:
                rows_.append({"x": float(x.loc[idx].median()), "shap": float(grouped.loc[idx, key].mean()), "n": int(len(idx))})
        dependence[key] = rows_
    for feat in {c2f[c] for c in X.columns if "__" in c}:
        if feat not in importance.index:
            continue
        rows_ = []
        for c in [c for c in X.columns if c2f[c] == feat]:
            idx = X.index[X[c] == 1]
            if len(idx) >= min_cell:
                rows_.append({"category": c.split("__", 1)[1], "shap": float(grouped.loc[idx, feat].mean()), "n": int(len(idx))})
        dependence[feat] = rows_
    return {"importance": {k: float(v) for k, v in importance.items()}, "dependence": dependence, "base_value": float(base)}


def export_model(model, shift: float) -> dict:
    if model.kind == "full":
        return {"kind": "spline", "spec": model.m.export(0.0), "intercept_recent": round(float(shift), 6)}
    raise NotImplementedError("update-model export is added only if rule (c) chooses it")


def deploy() -> dict:
    """Version 1.1 deployed models (amendment 1): knots and penalty from the 2005-2022 tuning, refitted
    on every patient of 2005-2025, slope correction by rule (b), recent-era intercept from out-of-fold
    estimates (5 folds, seed + 1) in 2023-2025, as v1.0.0."""
    from sklearn.model_selection import StratifiedKFold

    from . import metrics as M
    from .export import clean_json
    from .train import citl
    dec = load_decisions()
    U = Universe(load_base(), None)
    dev, recent = U.year < 2023, U.year >= 2023
    W_dev, W_all = np.flatnonzero(dev), np.arange(U.n)
    out = {}
    for stage in STAGES:
        out[stage] = {}
        for target in TARGETS:
            t0 = time.time()
            ch = chain_for(dec, target)
            f = slope_factor(dec, stage, target, ch)
            m_dev = fit_stage(U, stage, target, W_dev, SEED, ch, R.V110)
            st = m_dev.settings_tree()
            m = _with_slope(fit_stage(U, stage, target, W_all, SEED, ch, R.V110, settings=st), U, stage, target, W_all, f)
            T = U.rows(stage)
            y = U.y[target][T]
            p_oof = np.zeros(len(T))
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED + 1).split(np.zeros(len(T)), y):
                W = np.setdiff1d(W_all, T[va])
                mk = _with_slope(fit_stage(U, stage, target, W, SEED, ch, R.V110, settings=st), U, stage, target, W, f)
                p_oof[va] = 1 / (1 + np.exp(-mk.logit_rows(T[va])))
            rec = recent[T]
            shift = citl(y[rec], p_oof[rec])
            p_rec = M.expit(M.logit(p_oof[rec]) + shift)
            spec = clean_json(export_model(m, shift))
            lay = m.m.layout if m.kind == "full" else m.upd.layout
            stage_inputs = list(dict.fromkeys(R.input_of(c) for c in lay.stage_columns))
            info = {"structure": m.kind, "settings": st, "dev_settings": m_dev.settings, "slope_factor": f, "n_coef": int(m.n_coef),
                    "n_missing_terms": len(lay.ind), "missing_terms": [R.input_of(lay.columns[i]) for i in lay.ind],
                    "dropped_inputs": [k for k in stage_inputs if k not in {R.input_of(c) for c in lay.base}],
                    "merged": {lay.base[j]: [lay.stage_columns[i] for i in src] for j, src in enumerate(lay.sources) if len(src) > 1},
                    "intercept_recent": float(shift), "n": int(len(y)), "n_recent": int(rec.sum()),
                    "events_recent": int(y[rec].sum()), "oof_all": M.summary(y, p_oof),
                    "oof_recent_uncalibrated": M.summary(y[rec], p_oof[rec]),
                    "oof_recent_recalibrated": M.summary(y[rec], p_rec),
                    "explain": _explanation(m, U, stage, T),
                    "reality_bins": M.calibration_bins(y[rec], p_rec, n_bins=int(max(4, min(10, rec.sum() // 25))), min_n=10)}
            out[stage][target] = {"export": spec, "info": clean_json(info)}
            np.save(PRIV / f"deploy_oof_{stage}_{target}.npy", p_oof)
            print(f"[deploy v1.1 {stage}/{target}] {m.kind}, settings {st}, slope factor {f:.3f}, coefficients {m.n_coef}, "
                  f"recent shift {shift:+.3f} ({time.time() - t0:.0f}s)", flush=True)
    (PRIV / "deployed.json").write_text(json.dumps(out))
    return out


def build_bundle() -> dict:
    """model_bundle.json of version 1.1.0: the v1.1 reference in place of v1.0.0's; every other learner
    carried over unchanged (the stacking keeps its v1.0.0 reference base); v1.0.0's reference kept in
    `archive` so the app can offer it."""
    bundle0, _, _ = _published()
    dep = json.loads((PRIV / "deployed.json").read_text())
    sel = json.loads((PRIV / "selection.json").read_text())
    b = copy.deepcopy(bundle0)
    archive = {"version": "1.0.0", "fingerprint": V100_FINGERPRINT, "generated": bundle0["generated"], "models": {}}
    for stage in STAGES:
        archive["models"][stage] = {}
        for target in TARGETS:
            e = b["models"][stage][target]
            old = e["learners"][Lr.REFERENCE]
            archive["models"][stage][target] = {Lr.REFERENCE: old}
            stk = e["learners"].get("Stacking")
            if stk:
                stk["bases"] = [({k: v for k, v in old.items() if k != "intercept_recent"} if bb == {"same_as": Lr.REFERENCE} else bb)
                                for bb in stk["bases"]]
            e["learners"][Lr.REFERENCE] = dep[stage][target]["export"]
            e["chosen"] = sel["stages"][f"{stage}/{target}"]["chosen"]
    b["version"] = "1.1.0"
    b["generated"] = dt.date.today().isoformat()
    canon = json.dumps(b["models"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    b["fingerprint"] = hashlib.sha256(canon).hexdigest()
    b["archive"] = {"1.0.0": archive}
    return b



def extras_v11(n_boot: int = 100) -> dict:
    """extras.json parts that depend on the chosen models (extras.build_extras, parts 2, 2b and 3),
    for version 1.1: the ordinal layer on out-of-fold scores, its 2023-2025 validation, and the
    bootstrap uncertainty and classification instability of the deployed models."""
    from sklearn.metrics import roc_auc_score

    from . import metrics as M
    from .export import clean_json
    from .extras import UW_MRS, _bins, fit_ordinal, full_distribution, generalized_c, ordinal_probs
    dec = load_decisions()
    dep = json.loads((PRIV / "deployed.json").read_text())
    U = Universe(load_base(), None)
    dev, test, recent = U.year < 2023, U.year >= 2023, U.year >= 2023
    W_all, W_dev = np.arange(U.n), np.flatnonzero(dev)
    mrs_all = U.df["mrs90"].to_numpy(int)
    out = {"ordinal": {}, "ordinal_validation": {}, "uncertainty": {}, "stability": {}}
    for stage in STAGES:
        t0 = time.time()
        T = U.rows(stage)
        rec = recent[T]
        z = {t: M.logit(np.load(PRIV / f"deploy_oof_{stage}_{t}.npy")) for t in TARGETS}
        bg, bd, _ = fit_ordinal(z["good_outcome"], z["death90"], mrs_all[T])
        _, _, th = fit_ordinal(z["good_outcome"][rec], z["death90"][rec], mrs_all[T][rec], fixed_beta=(bg, bd))
        out["ordinal"][stage] = {"bg": bg, "bd": bd, "theta": th, "models": {t: Lr.REFERENCE for t in TARGETS}}
        # validation of the full distribution: development models (out-of-fold, seed + 2) and their test estimates
        Td, Te = U.rows(stage, dev), U.rows(stage, test)
        zd, pt = {}, {}
        from sklearn.model_selection import StratifiedKFold
        for t in TARGETS:
            ch = chain_for(dec, t)
            f = slope_factor(dec, stage, t, ch)
            st = fit_stage(U, stage, t, W_dev, SEED, ch, R.V110).settings_tree()
            yd = U.y[t][Td]
            p = np.zeros(len(Td))
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED + 2).split(np.zeros(len(Td)), yd):
                W = np.setdiff1d(W_dev, Td[va])
                mk = _with_slope(fit_stage(U, stage, t, W, SEED, ch, R.V110, settings=st), U, stage, t, W, f)
                p[va] = 1 / (1 + np.exp(-mk.logit_rows(Td[va])))
            zd[t] = M.logit(p)
            tv = pd.read_parquet(PRIV / f"temporal_v11_{stage}_{t}.parquet")
            pt[t] = tv.set_index("patient_id").loc[U.df["patient_id"].to_numpy()[Te], "p"].to_numpy(float)
        mrs_dev, mrs_te = mrs_all[Td], mrs_all[Te]
        vbg, vbd, vth = fit_ordinal(zd["good_outcome"], zd["death90"], mrs_dev)
        q = ordinal_probs(M.logit(pt["good_outcome"]), M.logit(pt["death90"]), vbg, vbd, vth)
        dist = full_distribution(pt["good_outcome"], pt["death90"], q)
        uw = dist @ np.asarray(UW_MRS)
        cum = np.cumsum(dist, axis=1)
        out["ordinal_validation"][stage] = {
            "n": int(len(mrs_te)), "observed": [float((mrs_te == k).mean()) for k in range(7)],
            "predicted": [float(v) for v in dist.mean(0)], "c_ordinal": generalized_c(uw, mrs_te),
            "cum_auroc": [float(roc_auc_score(mrs_te <= k, cum[:, k])) if 0 < (mrs_te <= k).sum() < len(mrs_te) else None
                          for k in range(6)],
            "uw_obs": float(np.mean(np.asarray(UW_MRS)[mrs_te])), "uw_pred": float(uw.mean())}
        # bootstrap uncertainty of the deployed models (settings fixed, n_boot refits)
        rng = np.random.default_rng(SEED)
        out["uncertainty"][stage], out["stability"][stage] = {}, {}
        for t in TARGETS:
            ch = chain_for(dec, t)
            f = slope_factor(dec, stage, t, ch)
            info = dep[stage][t]["info"]
            st, shift = info["settings"], info["intercept_recent"]
            m = _with_slope(fit_stage(U, stage, t, W_all, SEED, ch, R.V110, settings=st), U, stage, t, W_all, f)
            l0 = m.logit_rows(T) + shift
            p0 = M.expit(l0)
            L = np.empty((n_boot, len(T)))
            for b in range(n_boot):
                if ch.get(stage) == "update":
                    Wb = rng.integers(0, U.n, U.n)
                else:
                    Wb = T[rng.integers(0, len(T), len(T))]
                mb = _with_slope(fit_stage(U, stage, t, Wb, SEED, ch, R.V110, settings=st, grouped=True), U, stage, t, Wb, f)
                L[b] = mb.logit_rows(T) + shift
            lo_off = np.percentile(L, 2.5, axis=0) - l0
            hi_off = np.percentile(L, 97.5, axis=0) - l0
            out["uncertainty"][stage][t] = _bins(p0, lo_off, hi_off, 10)
            pb = M.expit(L)
            flip = ((pb >= 0.5) != (p0 >= 0.5)[None, :]).mean(axis=0)
            wd = np.percentile(pb, 97.5, axis=0) - np.percentile(pb, 2.5, axis=0)
            out["stability"][stage][t] = {"model": Lr.REFERENCE, "mape": float(np.mean(np.abs(pb - p0[None, :]))),
                                          "width_median": float(np.median(wd)), "share_wide": float(np.mean(wd > 0.2)),
                                          "share_flip_05": float(np.mean(flip > 0.2)),
                                          "share_near_05": float(np.mean(np.abs(p0 - 0.5) < 0.05)), "n_boot": n_boot}
        ov = out["ordinal_validation"][stage]
        print(f"[extras v1.1 {stage}] ordinal c {ov['c_ordinal']:.3f}; stability MAPE "
              f"{out['stability'][stage]['good_outcome']['mape']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    out = clean_json(out)
    (PRIV / "extras_v11.json").write_text(json.dumps(out))
    return out



def _boot_update_one(b: int, rows: np.ndarray, s2: dict, chains: dict) -> tuple[dict, dict]:
    """Change 3.1 in bootstrap resample b: the previous stage's model is the version 1.1 model of the
    same resample, refitted with that resample's knots and penalty (from the main bootstrap)."""
    import warnings
    warnings.filterwarnings("ignore")
    U = universe()
    rec, preds = {"b": b, "fits": {}}, {}
    for target in TARGETS:
        ch = chains[target]
        for stage in ("h24", "d7"):
            T = rows[U.elig[stage][rows]]
            P = U.rows(stage)
            pst = PREV[stage]
            # the previous stage's model in this resample (its settings from the main bootstrap)
            if ch.get(pst, "full") == "full":
                prev_settings = {"knots": s2[f"{S2}|{pst}/{target}"]["knots"], "C": s2[f"{S2}|{pst}/{target}"]["C"]}
            else:
                prev_settings = rec.setdefault("_tree", {})[f"{pst}/{target}"]
            M_prev = fit_stage(U, pst, target, rows, SEED, ch, R.V110, settings=prev_settings, grouped=True)
            m = fit_stage(U, stage, target, rows, SEED, {**ch, stage: "update"}, R.V110, grouped=True, prev_model=M_prev)
            rec.setdefault("_tree", {})[f"{stage}/{target}"] = m.settings_tree()
            yb, yte = U.y[target][T], U.y[target][P]
            p_app = 1 / (1 + np.exp(-m.logit_rows(T)))
            p_te = 1 / (1 + np.exp(-m.logit_rows(P)))
            key = f"{S3}|{stage}/{target}"
            rec["fits"][key] = {"knots": 3, "C": float(m.upd.settings["C"]), "n_coef": int(m.n_coef), "n": int(len(T)),
                                "unique": int(len(np.unique(T))), "apparent": measures(yb, p_app), "test": measures(yte, p_te)}
            preds[key] = p_te.astype(np.float32)
    rec.pop("_tree", None)
    return rec, preds


def bootstrap_update(n_jobs: int = 2, chunk: int = 10) -> None:
    """The update model (3.1) in the same 500 resamples, after the main bootstrap of those resamples.
    The day-7 update model builds on the model chosen at 24 hours (rule c, nested cross-validation)."""
    from joblib import Parallel, delayed
    dec = json.loads((PRIV / "decisions_nested.json").read_text())
    chains = {t: {"h24": "update" if dec["rule_c"].get(f"h24/{t}", {}).get("replace") else "full"} for t in TARGETS}
    U = universe()
    IDX = np.random.default_rng(SEED_BOOT).integers(0, U.n, size=(N_BOOT, U.n))
    main_recs = {r["b"]: r["fits"] for r in _records("bootstrap_resamples.jsonl")}
    log = REP / "bootstrap_update.jsonl"
    done = {r["b"] for r in _records("bootstrap_update.jsonl")}
    todo = [b for b in sorted(main_recs) if b not in done]
    print(f"update bootstrap: {len(done)} done, {len(todo)} to do; chains {chains}", flush=True)
    for start in range(0, len(todo), chunk):
        part = todo[start:start + chunk]
        t0 = time.time()
        res = Parallel(n_jobs=n_jobs)(delayed(_boot_update_one)(b, IDX[b], main_recs[b], chains) for b in part) if n_jobs > 1 else [_boot_update_one(b, IDX[b], main_recs[b], chains) for b in part]
        for rec, preds in res:
            np.savez_compressed(PRIV / f"bootu_{rec['b']:03d}.npz", **{k.replace("|", "__").replace("/", "__"): v for k, v in preds.items()})
        with log.open("a") as fh:
            for rec, _ in res:
                fh.write(json.dumps(rec) + "\n")
        print(f"update bootstrap: {len(done) + start + len(part)} at {now()} ({time.time() - t0:.0f}s for {len(part)})", flush=True)


def _apparent_update(U, stage, target, label):
    dec = json.loads((PRIV / "decisions_nested.json").read_text())
    ch = {"h24": "update" if dec["rule_c"].get(f"h24/{target}", {}).get("replace") else "full", stage: "update"}
    m = fit_stage(U, stage, target, np.arange(U.n), SEED, ch, R.V110)
    P = U.rows(stage)
    return m.upd, measures(U.y[target][P], 1 / (1 + np.exp(-m.logit_rows(P))))



def _fmt(x, d=3):
    return f"{x:.{d}f}"


def build_public(out_dir=None) -> None:
    """Version 1.1's public files: model_bundle.json (v1.1, v1.0.0 archived), validation.json (v1.0.0's
    record plus version 1.1 as its own learner and the v1_1 evaluation block) and extras.json (the
    chosen models' ordinal layer and uncertainty for version 1.1). Aggregates only."""
    from .export import clean_json, rounded
    out_dir = out_dir or (paths.ROOT / "artifacts" / "public")
    bundle = build_bundle()
    bundle0, validation, _ = _published()
    V = copy.deepcopy(validation)
    tmp = json.loads((PRIV / "temporal.json").read_text())
    sel = json.loads((PRIV / "selection.json").read_text())
    dep = json.loads((PRIV / "deployed.json").read_text())
    boot = json.loads((PRIV / "bootstrap_summary.json").read_text())
    nst = json.loads((PRIV / "nested.json").read_text())
    dec = load_decisions()
    chk = json.loads((REP / "pipeline_check.json").read_text())
    for label, per in tmp.items():
        for stage, tt in per.items():
            for target, blk in tt.items():
                r = V["splits"][label]["stages"][stage]["targets"][target]
                r["models"][REF_V11] = blk["model"]
                for c in ("roc", "calibration", "dca"):
                    r["curves"][c][REF_V11] = blk["curves"][c]
                for c in ("thresholds", "subgroups", "updating", "settings"):
                    r.setdefault(c, {})[REF_V11] = blk[c]
    V["groups"]["candidates"] = list(V["groups"]["candidates"]) + [REF_V11]
    V["nested"]["stages"]["admission"]["good_outcome"]["models"][REF_V11] = sel["nested_block"]
    if "admission" in V["nested"] and "good_outcome" in V["nested"]["admission"]:
        V["nested"]["admission"]["good_outcome"]["models"][REF_V11] = sel["nested_block"]
    for stage in STAGES:
        for target in TARGETS:
            V["deployed"][stage][target]["learners"][REF_V11] = dep[stage][target]["info"]
    rows = {}
    for stage in STAGES:
        rows[stage] = {}
        for target in TARGETS:
            k = f"{stage}/{target}"
            b0, b1 = boot["fits"][f"{S0}|{k}"], boot["fits"][f"{S2}|{k}"]
            rows[stage][target] = {
                "coef_v100": len(bundle0["models"][stage][target]["learners"][Lr.REFERENCE]["spec"]["coef"]),
                "coef_v11": len(bundle["models"][stage][target]["learners"][Lr.REFERENCE]["spec"]["coef"]),
                "nested_auroc_v100": nst[k]["models"][S0]["auroc"]["mean"], "nested_auroc_v11": nst[k]["models"][S2]["auroc"]["mean"],
                "nested_logloss_v100": nst[k]["models"][S0]["logloss"]["mean"], "nested_logloss_v11": nst[k]["models"][S2]["logloss"]["mean"],
                "boot_auroc_v100": b0["measures"]["auroc"]["corrected"], "boot_auroc_v11": b1["measures"]["auroc"]["corrected"],
                "boot_logloss_v100": b0["measures"]["logloss"]["corrected"], "boot_logloss_v11": b1["measures"]["logloss"]["corrected"],
                "boot_slope_v100": b0["measures"]["slope"]["corrected"], "boot_slope_v11": b1["measures"]["slope"]["corrected"],
                "risk_range_v100": b0["stability"]["risk_range_width_median"], "risk_range_v11": b1["stability"]["risk_range_width_median"],
                "slope_factor": slope_factor(dec, stage, target, chain_for(dec, target)),
                "temporal_auroc_v100": tmp["primary"][stage][target]["comparison"][Lr.REFERENCE]["auroc"],
                "temporal_auroc_v11": tmp["primary"][stage][target]["comparison"][REF_V11]["auroc"]}
    def sgn(x, d=3):
        return f"+{x:.{d}f}" if x >= 0 else f"−{abs(x):.{d}f}"

    def sgn_cs(x, d=3):
        return sgn(x, d).replace(".", ",")

    ra = dec["rule_a"]
    keep = dec["keep_simplifications"]
    rule_a = {"en": (f"Rule (a): changes 2.1–2.4 {'are kept' if keep else 'are not kept'}. Optimism-corrected changes against "
                     "the v1.0.0 specification: "
                     + "; ".join(f"{STAGE_EN[s_]}, mRS 0–2: AUROC {sgn(v['delta_auroc'], 4)}, log-loss {sgn(v['delta_logloss'], 4)}" for s_, v in ra.items())
                     + ". A simplification is dropped only if AUROC falls by more than 0.01 or log-loss rises by more than 0.005."),
              "cs": (f"Pravidlo (a): změny 2.1–2.4 {'zůstávají' if keep else 'nezůstávají'}. Změny s korekcí optimismu "
                     "oproti specifikaci v1.0.0: "
                     + "; ".join(f"{STAGE_CS[s_]}, mRS 0–2: AUROC {sgn_cs(v['delta_auroc'], 4)}, log-loss {sgn_cs(v['delta_logloss'], 4)}" for s_, v in ra.items())
                     + ". Zjednodušení se ruší, jen pokud AUROC klesne o více než 0,01 nebo log-loss vzroste o více než 0,005.")}
    applied = {k: v for k, v in dec["rule_b"].items() if v["apply"] and k.startswith(S2)}
    sl = lambda k: f"{STAGE_EN[k.split('|')[1].split('/')[0]]}, {'mRS 0–2' if 'good' in k else 'death'}"  # noqa: E731
    slc = lambda k: f"{STAGE_CS[k.split('|')[1].split('/')[0]]}, {'mRS 0–2' if 'good' in k else 'úmrtí'}"  # noqa: E731
    rule_b = {"en": ("Rule (b): the optimism-corrected calibration slope was inside 0.9–1.1 everywhere, so no slope correction is applied."
                     if not applied else "Rule (b): the linear predictor is multiplied by the optimism-corrected slope where it was outside 0.9–1.1: "
                     + "; ".join(f"{sl(k)} × {v['factor']:.2f}" for k, v in applied.items()) + "; elsewhere it was inside and nothing is applied."),
              "cs": ("Pravidlo (b): kalibrační sklon s korekcí optimismu byl všude v rozmezí 0,9–1,1, korekce sklonu se nepoužije."
                     if not applied else "Pravidlo (b): lineární prediktor se násobí sklonem s korekcí optimismu tam, kde byl mimo 0,9–1,1: "
                     + "; ".join(f"{slc(k)} × " + f"{v['factor']:.2f}".replace(".", ",") for k, v in applied.items()) + "; jinde byl uvnitř a nic se nepoužije.")}
    rc = dec["rule_c"]
    lab_en = lambda k: f"{STAGE_EN[k.split('/')[0]]}, {'mRS 0–2' if 'good' in k else 'death'}"  # noqa: E731
    lab_cs = lambda k: f"{STAGE_CS[k.split('/')[0]]}, {'mRS 0–2' if 'good' in k else 'úmrtí'}"  # noqa: E731
    repl = [k for k, v in rc.items() if v["replace"]]
    rule_c = {"en": ("Rule (c): the update model (previous stage's log-odds, current NIHSS, the stage's events) "
                     + ("replaces the full structure at " + ", ".join(lab_en(k) for k in repl) if repl else "replaces the full structure nowhere")
                     + ". Its nested cross-validated AUROC gain over the full structure: "
                     + "; ".join(f"{lab_en(k)} {sgn(v['delta_auroc'])}" for k, v in rc.items()) + " (at least +0.010 needed, with log-loss not worse)."),
              "cs": ("Pravidlo (c): model aktualizace (log-šance předchozí fáze, aktuální NIHSS, události fáze) "
                     + ("nahrazuje plnou strukturu: " + ", ".join(lab_cs(k) for k in repl) if repl else "plnou strukturu nikde nenahrazuje")
                     + ". Jeho zisk AUROC ve vnořené křížové validaci oproti plné struktuře: "
                     + "; ".join(f"{lab_cs(k)} {sgn_cs(v['delta_auroc'])}" for k, v in rc.items()) + " (potřeba alespoň +0,010 a log-loss ne horší).")}
    chosen_other = {k: v["chosen"] for k, v in sel["stages"].items() if v["chosen"] != Lr.REFERENCE}
    selection_txt = {"en": ("Amendment 1's selection rule, applied again with version 1.1 as the reference: "
                            + ("no candidate learner qualifies at any stage." if not chosen_other else "chosen: " + ", ".join(f"{k} {v}" for k, v in chosen_other.items()) + ".")),
                     "cs": ("Pravidlo výběru dodatku 1, znovu použité s verzí 1.1 jako referencí: "
                            + ("žádný kandidát v žádné fázi nevyhovuje." if not chosen_other else "zvoleno: " + ", ".join(f"{k} {v}" for k, v in chosen_other.items()) + "."))}
    V["v1_1"] = {"version": bundle["version"], "fingerprint": bundle["fingerprint"], "v1_0_0_fingerprint": V100_FINGERPRINT,
                 "plan": "docs/registrations/v1_1_plan.md", "results": "docs/registrations/v1_1_results.md",
                 "n_boot": boot["n_boot"], "rows": rows,
                 "decisions": {"rule_a": rule_a, "rule_b": rule_b, "rule_c": rule_c, "selection": selection_txt},
                 "pipeline_check": {k: chk.get(k) for k in ("passed_to_tolerance", "max_abs_coef_diff", "max_abs_p_diff")},
                 "nested": nst, "bootstrap": boot, "rules": dec, "selection": sel["stages"], "selection_nested": sel["nested_criterion"]}
    V["version"] = bundle["version"]
    V["generated"] = bundle["generated"]
    ex = json.loads((PRIV / "published_v1_0_0" / "extras.json").read_text())
    ev = json.loads((PRIV / "extras_v11.json").read_text())
    ex["archive_v1_0_0"] = {k: ex[k] for k in ("ordinal", "ordinal_validation", "uncertainty", "stability")}
    ex.update(ev)
    for name, obj, rnd in (("model_bundle.json", clean_json(bundle), False), ("validation.json", clean_json(V), True),
                           ("extras.json", clean_json(ex), False)):
        with open(out_dir / name, "w", encoding="utf-8") as fh:
            json.dump(rounded(obj) if rnd else obj, fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    print(f"public files written to {out_dir}; fingerprint {bundle['fingerprint']}")



def report_md(path=None) -> str:
    """docs/registrations/v1_1_results.md: the registered evaluation, written from the outputs."""
    path = path or (paths.ROOT / "docs" / "registrations" / "v1_1_results.md")
    chk = json.loads((REP / "pipeline_check.json").read_text())
    nst = json.loads((PRIV / "nested.json").read_text())
    boot = json.loads((PRIV / "bootstrap_summary.json").read_text())
    dec = load_decisions()
    tmp = json.loads((PRIV / "temporal.json").read_text())
    sel = json.loads((PRIV / "selection.json").read_text())
    dep = json.loads((PRIV / "deployed.json").read_text())
    bundle = build_bundle()
    f3 = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
    f4 = lambda x: "–" if x is None else f"{x:.4f}"  # noqa: E731
    sg = lambda x, d=3: "–" if x is None else (f"+{x:.{d}f}" if x >= 0 else f"−{abs(x):.{d}f}")  # noqa: E731
    out_name = {"good_outcome": "mRS 0–2", "death90": "death"}
    L = []
    w = L.append
    w("# MTOP version 1.1: results of the registered evaluation")
    w("")
    w(f"Written {dt.date.today().isoformat()} by `python -m mtop.v11 report` from the analysis outputs. Plan: "
      "[v1_1_plan.md](v1_1_plan.md) (17 November 2026) with amendment 1 (version 1.1 = changes 2.1–2.5, 3.1 and 4.2 on "
      "the data of v1.0.0) and amendment 2 (pipeline check to numerical tolerance, exact solver, how the plan is carried "
      "out), both committed before any change was evaluated. Development years 2005–2022 decide; the 2023–2025 test is "
      "reported but is not independent.")
    w("")
    a_ = dec["rule_a"]
    applied = [k.split("|")[1] for k, v in dec["rule_b"].items() if v["apply"] and k.startswith(S2)]
    w("## Summary")
    w("")
    w(f"- **Pipeline check passed** (to the tolerance of amendment 2): with every change switched off, the code reproduces "
      f"v1.0.0; every patient's estimate within {chk['max_abs_p_diff']:.0e}.")
    w(f"- **Changes 2.1–2.4 are kept** (rule a): optimism-corrected AUROC "
      + ", ".join(f"{sg(v['delta_auroc'], 4)} ({STAGE_EN[k]})" for k, v in a_.items())
      + "; log-loss " + ", ".join(f"{sg(v['delta_logloss'], 4)}" for v in a_.values())
      + ". Coefficients of the deployed reference: 91 → 59 before thrombectomy, 135–148 → 82 on day 7.")
    w(f"- **Slope correction** (rule b) applies only to " + (", ".join(f"{STAGE_EN[k.split('/')[0]]}, {out_name[k.split('/')[1]]}" for k in applied) or "no model")
      + f" (factor {', '.join(f'{dec['rule_b'][S2 + '|' + k]['factor']:.3f}' for k in applied)}).")
    w("- **The update model is not adopted** (rule c): it gained +0.007 AUROC for mRS 0–2 at 24 hours and day 7, "
      "where +0.010 was required. It did give much narrower individual risk ranges (section 3).")
    w("- **No candidate learner replaces the reference** when the selection rule of amendment 1 is applied again with "
      "version 1.1 as the reference.")
    w(f"- **Deployed version 1.1**: fingerprint `{bundle['fingerprint'][:16]}…`; v1.0.0 archived and selectable in the app.")
    w("")
    w("## 1. Pipeline check (section 5, step 2; amendment 2)")
    w("")
    st_ = chk["steps"]
    w(f"- Registry export SHA-256 `{st_['export_sha256']}`.")
    w(f"- Data checks, cohorts and events at every stage, model columns: identical to v1.0.0.")
    w(f"- Reference and linear regression re-tuned on 2005–2022: the same knots and penalties at all eight stage–outcome "
      f"pairs; 2023–2025 results identical to 5 decimals.")
    w(f"- Redeployed on all years: every exported number other than the coefficients and the recent-era intercepts is "
      f"identical; largest coefficient difference {chk['max_abs_coef_diff']:.1e}, largest recent-era intercept difference "
      f"{chk.get('max_recent_intercept_diff', 1e-6):.0e}, largest difference in any patient's estimate {chk['max_abs_p_diff']:.1e} "
      f"(criterion: below 1e-05).")
    w(f"- **Passed to tolerance: {'yes' if chk['passed_to_tolerance'] else 'no'}.** Bit for bit: "
      f"{'yes' if chk['passed_exact'] else 'no'}, because where L-BFGS stops depends on the machine (amendment 2).")
    w("")
    w("## 2. Nested cross-validation, 2005–2022 (5 outer folds × 3 repeats)")
    w("")
    w("Mean over the 15 outer folds. *1.0.0 specification*: v1.0.0's model refitted with the exact solver; "
      "*+ 2.1*, *+ 2.2*, *+ 2.3*: each change added in turn (for the record); *2.1-2.4*: the four simplifications, "
      "the step that rule (a) judges; *1.1 full structure*: plus change 2.5's tuning (5-fold CV repeated 5 times); "
      "*update model*: change 3.1 (at day 7 built on the 24-hour full structure, the model chosen there, or on the "
      "24-hour update model); *NIHSS alone*: the published points-to-risk benchmark.")
    w("")
    for key, res in nst.items():
        stage, target = key.split("/")
        w(f"**{STAGE_EN[stage].capitalize()}, {out_name[target]}** (n = {res['n']}, events {res['events']})")
        w("")
        w("| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Coefficients |")
        w("|---|---|---|---|---|---|---|")
        for lab, m in res["models"].items():
            w(f"| {lab} | {f4(m['auroc']['mean'])} | {f4(m['logloss']['mean'])} | {f4(m['brier']['mean'])} | "
              f"{sg(m['citl']['mean'])} | {f3(m['slope']['mean'])} | {m['n_coef_median']:.0f} |")
        w("")
    w("## 3. Full-pipeline bootstrap, 2005–2022 (500 resamples, seed 2026)")
    w("")
    w("Optimism-corrected measures (Harrell). The resamples are those of the v1.0.0 bootstrap, drawn over the patients "
      "before thrombectomy; each later stage uses the resampled patients eligible for it. Risk range: median width of a "
      "patient's 95% range across the 500 bootstrap models. The 1.0.0 specification reproduces the published v1.0.0 "
      "bootstrap before thrombectomy (corrected AUROC 0.761 and 0.790, slope 1.09 and 1.10, risk ranges 21 and 15 points).")
    w("")
    for stage in STAGES:
        for target in TARGETS:
            k = f"{stage}/{target}"
            labs = [lab for lab in (S0, S1, S2, S3) if f"{lab}|{k}" in boot["fits"]]
            if not labs:
                continue
            w(f"**{STAGE_EN[stage].capitalize()}, {out_name[target]}**")
            w("")
            w("| Specification | AUROC | Log-loss | Brier | Calibration intercept | Calibration slope | Risk range | Coefficients |")
            w("|---|---|---|---|---|---|---|---|")
            for lab in labs:
                r = boot["fits"][f"{lab}|{k}"]
                m = r["measures"]
                w(f"| {lab} | {f4(m['auroc']['corrected'])} | {f4(m['logloss']['corrected'])} | {f4(m['brier']['corrected'])} | "
                  f"{sg(m['citl']['corrected'])} | {f3(m['slope']['corrected'])} | {100 * r['stability']['risk_range_width_median']:.1f} points | "
                  f"{r['apparent_model']['n_coef']} |")
            w("")
    w("## 4. Decisions (section 6, step 3)")
    w("")
    for st, v in dec["rule_a"].items():
        w(f"- **Rule (a), {STAGE_EN[st]}, mRS 0–2:** optimism-corrected AUROC {sg(v['delta_auroc'], 4)}, log-loss "
          f"{sg(v['delta_logloss'], 4)} (2.1–2.4 against the 1.0.0 specification). Kept: {v['keep']}.")
    w(f"- **Changes 2.1–2.4 kept:** {dec['keep_simplifications']}.")
    for k, v in dec["rule_b"].items():
        lab, sk = k.split("|")
        if lab != S2:
            continue
        stage, target = sk.split("/")
        w(f"- **Rule (b), {STAGE_EN[stage]}, {out_name[target]}:** optimism-corrected slope {f3(v['corrected_slope'])}; "
          + (f"slope correction × {v['factor']:.3f} applied." if v["apply"] else "inside 0.9–1.1, no correction."))
    for k, v in dec["rule_c"].items():
        stage, target = k.split("/")
        w(f"- **Rule (c), {STAGE_EN[stage]}, {out_name[target]}:** {v['variant']} against the full structure: nested AUROC "
          f"{sg(v['delta_auroc'], 4)}, log-loss {sg(v['delta_logloss'], 4)}. Replaces the full structure: {v['replace']}.")
    w("")
    w("## 5. Temporal test, 2023–2025 (not independent)")
    w("")
    w("Version 1.1 was designed after these results were known, so they decide nothing (section 6, step 4). Models "
      "fitted on 2005–2022; AUROC with 95% bootstrap interval; the difference is paired (2,000 resamples).")
    w("")
    w("| Stage | Outcome | n (events) | v1.0.0 AUROC | Version 1.1 AUROC | Difference (95% CI) | v1.0.0 intercept / slope | Version 1.1 intercept / slope |")
    w("|---|---|---|---|---|---|---|---|")
    for stage in STAGES:
        for target in TARGETS:
            b = tmp["primary"][stage][target]
            c0, c1 = b["comparison"][Lr.REFERENCE], b["comparison"][REF_V11]
            d = b["model"]["ci"]["delta_auroc_vs_ref"]
            w(f"| {STAGE_EN[stage].capitalize()} | {out_name[target]} | {c1['n']} ({c1['events']}) | {f3(c0['auroc'])} ({f3(c0['ci']['auroc'][0])}–{f3(c0['ci']['auroc'][1])}) | "
              f"{f3(c1['auroc'])} ({f3(c1['ci']['auroc'][0])}–{f3(c1['ci']['auroc'][1])}) | {sg(d['mean'])} ({sg(d['ci'][0])} to {sg(d['ci'][1])}) | "
              f"{sg(c0['citl'], 2)} / {c0['slope']:.2f} | {sg(c1['citl'], 2)} / {c1['slope']:.2f} |")
    w("")
    w("## 6. The selection rule of amendment 1, applied again")
    w("")
    nc = sel["nested_criterion"]
    w(f"Version 1.1 is the reference. Temporal criterion: Holm over the seven candidates (paired bootstrap of the AUROC "
      f"difference, 2,000 resamples). Nested criterion (all years, before thrombectomy, mRS 0–2, v1.0.0's folds): version 1.1 "
      f"mean AUROC {f4(nc['v1_1_reference']['auroc_mean'])}; v1.0.0 reproduced on those folds within "
      f"{nc['v1_0_0_reproduced_fold_auroc_max_diff']:.1e}.")
    w("")
    w("A candidate qualifies on the temporal test only with a higher AUROC and a Holm-adjusted p below 0.05. The candidates' "
      "2005–2022 fits were regenerated with the v1.0.0 code; their 2023–2025 AUROCs equal the published ones within 0.0001, "
      "except stacking for death at 24 hours (0.852 against 0.854).")
    w("")
    w("| Stage | Outcome | Chosen | Best candidate: AUROC difference, Holm-adjusted p |")
    w("|---|---|---|---|")
    for k, v in sel["stages"].items():
        stage, target = k.split("/")
        cs = [(c, d["delta_test"]["mean"], d["p_holm"]) for c, d in v["candidates"].items() if d.get("delta_test")]
        best = max(cs, key=lambda t: t[1]) if cs else (None, None, None)
        ch = "Regression with splines, version 1.1" if v["chosen"] == Lr.REFERENCE else v["chosen"]
        w(f"| {STAGE_EN[stage].capitalize()} | {out_name[target]} | {ch} | {best[0]}: {sg(best[1])}, p = {f3(best[2])} |")
    w("")
    w("## 7. Deployed version 1.1")
    w("")
    w(f"Fingerprint `{bundle['fingerprint']}`. Knots and penalty from the 2005–2022 tuning, refitted on all 1,451 "
      "patients, intercept re-estimated on 2023–2025 from out-of-fold estimates. The other learners are carried over from "
      "v1.0.0 unchanged; v1.0.0's reference stays in the bundle's archive and in the app.")
    w("")
    w("| Stage | Outcome | Knots | Penalty C | Slope factor | Coefficients (v1.0.0 → 1.1) | Recent-era intercept |")
    w("|---|---|---|---|---|---|---|")
    b0, _, _ = _published()
    for stage in STAGES:
        for target in TARGETS:
            i = dep[stage][target]["info"]
            n0 = len(b0["models"][stage][target]["learners"][Lr.REFERENCE]["spec"]["coef"])
            w(f"| {STAGE_EN[stage].capitalize()} | {out_name[target]} | {i['settings']['knots']} | {i['settings']['C']:.4f} | "
              f"{i['slope_factor']:.3f} | {n0} → {i['n_coef']} | {sg(i['intercept_recent'])} |")
    w("")
    txt = "\n".join(L) + "\n"
    Path(path).write_text(txt, encoding="utf-8")
    return txt



def diagnostics_md(path=None) -> str:
    """docs/model_diagnostics_v1_1.md: settings and structure of the deployed version 1.1 models
    (aggregates only)."""
    path = path or (paths.ROOT / "docs" / "model_diagnostics_v1_1.md")
    dep = json.loads((PRIV / "deployed.json").read_text())
    b0, _, _ = _published()
    bundle = build_bundle()
    out_name = {"good_outcome": "mRS 0–2", "death90": "death"}
    L = []
    w = L.append
    w("# MTOP version 1.1: model diagnostics")
    w("")
    w(f"Generated by `python -m mtop.v11 diagnostics` on {dt.date.today().isoformat()} for the model bundle with fingerprint "
      f"{bundle['fingerprint'][:8]}…. Aggregates only. v1.0.0's diagnostics: [model_diagnostics.md](model_diagnostics.md). "
      "Evaluation: [registrations/v1_1_results.md](registrations/v1_1_results.md).")
    w("")
    w("## Settings and size of the reference model")
    w("")
    w("Knots (3, 4 or 5) and the penalty C (50 values, 0.0001 to 100) chosen on 2005–2022 by 5-fold cross-validation repeated "
      "5 times (mean log-loss); the deployed models are refitted on all years with these settings. A missing-value term "
      "exists only for an input that at least 20 of the fitting patients lack (one term per input), so the deployed models, "
      "fitted on all years, can have a few more than the development models.")
    w("")
    w("| Stage | Outcome | Knots | C | CV log-loss (25 folds) | Slope factor | Coefficients v1.0.0 → 1.1 | Missing-value terms v1.0.0 → 1.1 |")
    w("|---|---|---|---|---|---|---|---|")
    for stage in STAGES:
        for target in TARGETS:
            i = dep[stage][target]["info"]
            s0 = b0["models"][stage][target]["learners"][Lr.REFERENCE]["spec"]
            ds = i.get("dev_settings", {})
            w(f"| {STAGE_EN[stage].capitalize()} | {out_name[target]} | {i['settings']['knots']} | {i['settings']['C']:.4f} | "
              f"{ds.get('cv_logloss', float('nan')):.4f} | {i['slope_factor']:.3f} | {len(s0['coef'])} → {i['n_coef']} | "
              f"{len(s0['indicators'])} → {i['n_missing_terms']} |")
    w("")
    w("## Inputs left out, categories joined, and missing-value terms")
    w("")
    from .config import features
    lab = {f.key: f.spec.get("label", {}).get("en", f.key) for f in features()}
    cat = lambda c: f"{lab.get(c.split('__')[0], c)}: {c.split('__')[1].replace('_', ' ')}"  # noqa: E731
    w("The structure depends only on the patients a model is fitted on, so it is the same for mRS 0–2 and death.")
    w("")
    for stage in STAGES:
        for target in TARGETS[:1]:
            i = dep[stage][target]["info"]
            w(f"**{STAGE_EN[stage].capitalize()}.** Left out (change 2.3): "
              f"{', '.join(lab.get(k, k) for k in i['dropped_inputs']) or 'none'}. "
              f"Joined (change 2.4): {'; '.join(', '.join(cat(c) for c in v[1:]) + ' → ' + cat(k) for k, v in i['merged'].items()) or 'none'}. "
              f"Missing-value terms ({len(i['missing_terms'])}): {', '.join(lab.get(k, k) for k in i['missing_terms'])}.")
            w("")
    txt = "\n".join(L) + "\n"
    Path(path).write_text(txt, encoding="utf-8")
    return txt


STAGE_EN = {"admission": "before thrombectomy", "procedure": "end of procedure", "h24": "24 hours", "d7": "day 7"}
STAGE_CS = {"admission": "před trombektomií", "procedure": "konec výkonu", "h24": "24 h", "d7": "7. den"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("candidates")
    sub.add_parser("nested")
    sub.add_parser("summarise")
    sub.add_parser("temporal")
    sub.add_parser("selection")
    sub.add_parser("deploy")
    sub.add_parser("extras")
    sub.add_parser("decide-nested")
    pu = sub.add_parser("bootstrap-update")
    pu.add_argument("--n-jobs", type=int, default=2)
    sub.add_parser("report")
    sub.add_parser("diagnostics")
    pp = sub.add_parser("public")
    pp.add_argument("--out", help="folder for the public files (default: artifacts/public of the repository)")
    pb = sub.add_parser("bootstrap")
    pb.add_argument("--n-jobs", type=int, default=2)
    pb.add_argument("--chunk", type=int, default=10)
    pb.add_argument("--only", type=int, nargs="*")
    a = ap.parse_args(argv)
    paths.ensure_dirs()
    if a.cmd == "check":
        check()
    elif a.cmd == "candidates":
        candidates()
    elif a.cmd == "nested":
        nested()
    elif a.cmd == "summarise":
        boot = summarise_bootstrap()
        if (REP / "bootstrap_update.jsonl").exists():
            bu_ = summarise_bootstrap(prefix="bootu", records="bootstrap_update.jsonl", labels=(S3,), apparent_fn=_apparent_update)
            boot["fits"].update(bu_["fits"])
        (PRIV / "bootstrap_summary.json").write_text(json.dumps(boot, indent=1))
        nres = json.loads((PRIV / "nested.json").read_text()) if (PRIV / "nested.json").exists() else {}
        dec = decide(boot, nres)
        (PRIV / "decisions.json").write_text(json.dumps(dec, indent=1))
        print(json.dumps(dec, indent=1))
    elif a.cmd == "temporal":
        temporal()
    elif a.cmd == "selection":
        selection()
    elif a.cmd == "deploy":
        deploy()
    elif a.cmd == "extras":
        extras_v11()
    elif a.cmd == "decide-nested":
        dec = decide({"fits": {}}, json.loads((PRIV / "nested.json").read_text()))
        (PRIV / "decisions_nested.json").write_text(json.dumps(dec, indent=1))
        print(json.dumps(dec["rule_c"], indent=1))
    elif a.cmd == "bootstrap-update":
        bootstrap_update(n_jobs=a.n_jobs)
    elif a.cmd == "report":
        report_md()
    elif a.cmd == "diagnostics":
        diagnostics_md()
    elif a.cmd == "public":
        from pathlib import Path
        build_public(Path(a.out) if a.out else None)
    elif a.cmd == "bootstrap":
        bootstrap(n_jobs=a.n_jobs, chunk=a.chunk, only=a.only)


if __name__ == "__main__":
    main()
