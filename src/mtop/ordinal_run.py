"""Running the ordinal-model plan (docs/registrations/ordinal_plan.md) on the registry.

Order of work (plan, section 5): pipeline check (`check`), then the development-year evaluation
(`nested`, `bootstrap`, `brant`, `sample_size`) and the decisions (`develop`), which are written and
committed before any 2023-2025 or 2021-2025 result of the ordinal model or of the comparison-only
learner is computed. Only then `temporal`, `deploy` and the release.

Row-level outputs go to artifacts/private/ordinal (git-ignored); aggregates to reports/ordinal and,
for publication, artifacts/public.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import ordinal as O
from . import paths
from . import reference as R
from . import v11

STAGES = v11.STAGES
TARGETS = v11.TARGETS
PRIMARY = ("admission", "h24")                 # plan, section 6
LGB_STAGES = ("procedure", "h24", "d7")        # plan, section 9
K = 7
SEED = v11.SEED                                # 20261016, the folds of v1.0.0
N_BOOT, SEED_BOOT = 500, 2026                  # the resamples of v1.0.0 and v1.1
V11_SLOPE = {("h24", "death90"): 1.10193}      # v1.1 rule (b) (validation.json, v1_1.rows)
PRIV = paths.PRIVATE / "ordinal"
REP = paths.REPORTS / "ordinal"
EXPORT_SHA = "1a19f809ed688a3940c38ed4d0b1198f0a978b806f9ed7b50721777d980d50f7"
COHORTS = {"admission": (1225, 226), "procedure": (1210, 221), "h24": (1115, 120), "d7": (1000, 139)}
MIN_GRADE = 20
EPS = np.finfo(float).eps
UW = np.array([1.0, 0.91, 0.76, 0.65, 0.33, 0.0, 0.0])

ORD = "Ordinal regression"
LAYER = "v1.1 layer"
HYB = "v1.1 bands with ordinal shares"
LGB = "Ordinal LightGBM (comparison only)"
BIN = "v1.1 binary models"
DICH = (("mrs01", 1), ("mrs02", 2), ("mrs03", 3))
BIN_MEASURES = ("auroc", "brier", "logloss", "citl", "slope", "oe")


def now() -> str:
    return v11.now()


def _threads1():
    os.environ["MTOP_THREADS"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")


# --------------------------------------------------------------------------- data
_U = {}


def universe(test_from=2023) -> v11.Universe:
    if test_from not in _U:
        _U[test_from] = v11.Universe(v11.load_base(), test_from)
    return _U[test_from]


def grades(U) -> np.ndarray:
    return U.df["mrs90"].to_numpy(int)


def grade_map(y_dev: np.ndarray, min_n: int = MIN_GRADE) -> dict:
    """Plan, section 3: a grade with fewer than `min_n` development patients joins a neighbour in its
    band (0-2 or 3-5); death (6) is never merged."""
    counts = {g: int(np.sum(y_dev == g)) for g in range(7)}
    target = {0: 1, 2: 1, 3: 4, 5: 4}
    mapping = {g: g for g in range(7)}
    for g in (0, 2, 3, 5, 1, 4):
        if counts[g] >= min_n:
            continue
        if g in target:
            mapping[g] = target[g]
        else:
            nb = (0, 2) if g == 1 else (3, 5)
            mapping[g] = min(nb, key=lambda h: counts[h])
    return {"mapping": mapping, "counts": counts, "merged": any(mapping[g] != g for g in range(7))}


# --------------------------------------------------------------------------- measures (section 6)
def cumulative(P) -> np.ndarray:
    return np.cumsum(P, axis=1)[:, : K - 1]


def rps(y, P) -> float:
    I = (np.asarray(y)[:, None] <= np.arange(K - 1)[None, :]).astype(float)
    return float(np.mean(np.sum((cumulative(P) - I) ** 2, axis=1) / (K - 1)))


def ologloss(y, P) -> float:
    y = np.asarray(y, int)
    return float(np.mean(-np.log(np.clip(P[np.arange(len(y)), y], EPS, None))))


def gen_c(score_better, y) -> float:
    """Generalised concordance: of two patients with different mRS, the probability that the one with
    the better score had the better outcome (ties in the score count 1/2). O(n log n) by ranks."""
    from scipy.stats import rankdata
    s = np.asarray(score_better, float)
    y = np.asarray(y, int)
    # pairs (i better outcome than j): y_i < y_j. Concordant if s_i > s_j.
    order = np.argsort(y, kind="stable")
    s, y = s[order], y[order]
    tot = conc = 0.0
    seen = np.empty(0)
    for g in np.unique(y)[::-1]:              # from the worst grade: compare each grade with all worse ones
        cur = s[y == g]
        if len(seen):
            allv = np.concatenate([seen, cur])
            r = rankdata(allv)                # average ranks
            rc = r[len(seen):]
            # for each current patient: number of worse patients with a lower score + 1/2 ties
            # = (rank among all) - (rank among current)  computed with average ranks
            rcur = rankdata(cur)
            conc += float(np.sum(rc - rcur))
            tot += float(len(cur) * len(seen))
        seen = np.concatenate([seen, cur])
    return conc / tot if tot else float("nan")


def dist_measures(y, P, lp=None) -> dict:
    """Every pre-specified measure of a predicted distribution (n x 7) against observed grades."""
    y = np.asarray(y, int)
    P = np.asarray(P, float)
    out = {"n": int(len(y)), "rps": rps(y, P), "ologloss": ologloss(y, P),
           "genc_uw": gen_c(P @ UW, y), "genc_grade": gen_c(-(P @ np.arange(K)), y),
           "uw_pred": float(np.mean(P @ UW)), "uw_obs": float(np.mean(UW[y])),
           "grade_pred": [float(v) for v in P.mean(0)], "grade_obs": [float(np.mean(y == g)) for g in range(K)]}
    out["uw_diff"] = out["uw_pred"] - out["uw_obs"]
    for name, k in DICH:
        out[name] = v11.measures((y <= k).astype(float), P[:, : k + 1].sum(1))
    out["death"] = v11.measures((y == K - 1).astype(float), P[:, K - 1])
    out["min_p_observed"] = float(np.min(P[np.arange(len(y)), y]))
    if lp is not None:
        out["ord_slope"] = O.calibration_slope(lp, y, K)[0]
    return out


def flat(m: dict) -> dict:
    """Scalar measures of dist_measures, for optimism and summaries."""
    out = {k: m[k] for k in ("rps", "ologloss", "genc_uw", "genc_grade", "uw_pred", "uw_diff") if k in m}
    if "ord_slope" in m:
        out["ord_slope"] = m["ord_slope"]
    for d in [n for n, _ in DICH] + ["death"]:
        for k in BIN_MEASURES:
            out[f"{d}_{k}"] = m[d][k]
    for g in range(K):
        out[f"grade{g}_diff"] = m["grade_pred"][g] - m["grade_obs"][g]
    return out


# --------------------------------------------------------------------------- models
def fit_binary(X, y, seed, stage, target, settings=None, groups=None):
    """The v1.1 binary model: the v1.1 full structure, with rule (b)'s slope factor where it applies."""
    m = R.fit(X, y, seed, R.V110, groups=groups, settings=settings)
    f = V11_SLOPE.get((stage, target), 1.0)
    return R.with_slope(m, X, y, f) if f != 1.0 else m


def expit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def layer_fit(U, stage, Tr, seed, models: dict, y_mrs) -> tuple:
    """The v1.1 proportional-odds layer on the training patients Tr: out-of-fold log-odds of both
    binary models (5 folds, seed + 1, settings fixed, as v1.1 deploy), then extras.fit_ordinal."""
    from sklearn.model_selection import StratifiedKFold

    from .extras import fit_ordinal
    z = {}
    for t in TARGETS:
        yt = U.y[t][Tr]
        zt = np.zeros(len(Tr))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed + 1).split(np.zeros(len(Tr)), yt):
            mk = fit_binary(U.X[stage].iloc[Tr[tr]], yt[tr], seed, stage, t, settings=models[t].settings)
            zt[va] = mk.logit(U.X[stage].iloc[Tr[va]])
        z[t] = zt
    return fit_ordinal(z["good_outcome"], z["death90"], y_mrs[Tr])


def layer_dist(pg, pdeath, params) -> np.ndarray:
    from .extras import full_distribution, ordinal_probs
    bg, bd, th = params
    q = ordinal_probs(logit(pg), logit(pdeath), bg, bd, th)
    return full_distribution(pg, pdeath, q)


def hybrid_dist(pg, pdeath, P_ord) -> np.ndarray:
    """Rule O1's alternative: the binary models' bands split by the ordinal model's shares."""
    from .extras import full_distribution
    return full_distribution(pg, pdeath, P_ord)


def lgb_settings(split: str, stage: str) -> dict:
    v = json.loads((paths.PUBLIC / "validation.json").read_text())
    return dict(v["splits"][split]["stages"][stage]["targets"]["good_outcome"]["settings"]["LightGBM"])


def fit_lgb(X, y, settings, stage) -> list:
    """Plan, section 9: one LightGBM per cut-point P(mRS <= k), the v1.0.0 settings for mRS 0-2, the
    monotone constraints of mRS 0-2 (each points towards a better outcome), >= 20 patients a leaf."""
    from . import learners as Lr
    from .features import monotone_vector
    mono = monotone_vector(stage, "good_outcome")
    y = np.asarray(y, int)
    return [Lr.refit("LightGBM", settings, X, (y <= k).astype(float), SEED, mono, min_leaf=20) for k in range(K - 1)]


def lgb_cum(models, X) -> np.ndarray:
    C = np.column_stack([m.proba(X) for m in models])
    return np.minimum(np.maximum.accumulate(C, axis=1), 1.0)


def cum_to_P(C) -> np.ndarray:
    n = C.shape[0]
    return np.diff(np.hstack([np.zeros((n, 1)), C, np.ones((n, 1))]), axis=1)


# --------------------------------------------------------------------------- step 3: pipeline check
def check() -> dict:
    """Section 5, step 3: same export, same cohorts as v1.1; a small unpenalised ordinal model on the
    registry matches statsmodels to 1e-6; the grade counts for the sparse-grade rule."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    out = {"export_sha256": v11.sha256_file(paths.DEFAULT_RAW_FILE)}
    out["export_identical"] = out["export_sha256"] == EXPORT_SHA
    base = v11.load_base()
    out["cohorts"] = {}
    for stage in STAGES:
        d, dev, test, X = v11.stage_data(base, stage)
        out["cohorts"][stage] = {"dev": int(len(dev)), "test": int(len(test)),
                                 "identical": (int(len(dev)), int(len(test))) == COHORTS[stage],
                                 "grades_dev": grade_map(dev["mrs90"].to_numpy(int))}
    d, dev, test, X = v11.stage_data(base, "admission")
    Xd = X.loc[dev.index, ["age", "nihss_adm"]].to_numpy(float)
    ok = ~np.isnan(Xd).any(axis=1)
    Z = (Xd[ok] - Xd[ok].mean(0)) / Xd[ok].std(0)
    y = dev["mrs90"].to_numpy(int)[ok]
    par = O.newton(Z, y, np.ones(len(y)), 1e10, 7)
    mod = OrderedModel(y, Z, distr="logit")
    res = mod.fit(method="newton", disp=False, maxiter=200, tol=1e-12)
    diff = max(float(np.max(np.abs(par[:2] - res.params[:2]))),
               float(np.max(np.abs(par[2:] - mod.transform_threshold_params(res.params)[1:-1]))))
    out["statsmodels_max_abs_diff"] = diff
    out["statsmodels_match"] = diff < 1e-6
    out["passed"] = bool(out["export_identical"] and all(c["identical"] for c in out["cohorts"].values())
                         and out["statsmodels_match"])
    out["generated"] = now()
    PRIV.mkdir(parents=True, exist_ok=True)
    (PRIV / "pipeline_check.json").write_text(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------- step 4a: nested CV
def outer_folds(U, stage):
    """The v1.0.0 folds (5 x 3, seed 20261016, stratified by mRS 0-2), as v1.1's nested CV."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    P = U.rows(stage)
    return P, list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(
        np.zeros(len(P)), U.y["good_outcome"][P]))


def _nested_one(stage: str, k: int):
    import warnings
    warnings.filterwarnings("ignore")
    _threads1()
    t0 = time.time()
    U = universe()
    y = grades(U)
    P, folds = outer_folds(U, stage)
    tr, va = folds[k]
    sk = SEED + 100 + k
    Tr, Te = P[tr], P[va]
    Xtr, Xte = U.X[stage].iloc[Tr], U.X[stage].iloc[Te]
    m = O.fit(Xtr, y[Tr], K, sk)
    P_ord, lp = m.predict(Xte), m.lp(Xte)
    bins = {t: fit_binary(Xtr, U.y[t][Tr], sk, stage, t) for t in TARGETS}
    pg, pdeath = expit(bins["good_outcome"].logit(Xte)), expit(bins["death90"].logit(Xte))
    lay = layer_fit(U, stage, Tr, sk, bins, y)
    preds = {ORD: P_ord, LAYER: layer_dist(pg, pdeath, lay), HYB: hybrid_dist(pg, pdeath, P_ord)}
    if stage in LGB_STAGES:
        preds[LGB] = cum_to_P(lgb_cum(fit_lgb(Xtr, y[Tr], lgb_settings("primary", stage), stage), Xte))
    rec = {"stage": stage, "fold": k, "n_train": int(len(Tr)), "n_test": int(len(Te)),
           "ordinal_settings": {"knots": m.settings["knots"], "C": m.settings["C"], "C_at_grid_end": m.settings["C_at_grid_end"],
                                "n_coef": m.n_coef},
           "binary_settings": {t: {"knots": bins[t].settings["knots"], "C": bins[t].settings["C"]} for t in TARGETS},
           "layer": {"bg": lay[0], "bd": lay[1], "theta": lay[2]},
           "models": {lab: dist_measures(y[Te], Pm, lp if lab == ORD else None) for lab, Pm in preds.items()},
           "binary": {"mrs02": v11.measures(U.y["good_outcome"][Te], pg), "death": v11.measures(U.y["death90"][Te], pdeath)}}
    arrays = {"rows": Te, "lp": lp, "theta": m.theta, "pg": pg, "pdeath": pdeath,
              **{f"P_{i}": preds[lab] for i, lab in enumerate((ORD, LAYER, HYB, LGB)) if lab in preds}}
    rec["seconds"] = round(time.time() - t0, 1)
    return rec, arrays


def nested(stages=STAGES, n_jobs: int = 2) -> None:
    """Section 6: nested cross-validation, every fold's whole pipeline refitted (checkpointed by fold)."""
    from joblib import Parallel, delayed
    out_dir = PRIV / "nested"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = PRIV / "nested_records.jsonl"
    done = set()
    if log.exists():
        done = {(r["stage"], r["fold"]) for r in map(json.loads, log.read_text().splitlines()) if r}
    todo = [(s, k) for s in stages for k in range(15) if (s, k) not in done]
    print(f"nested: {len(done)} done, {len(todo)} to do ({now()})", flush=True)
    for start in range(0, len(todo), n_jobs * 2):
        part = todo[start:start + n_jobs * 2]
        res = Parallel(n_jobs=n_jobs)(delayed(_nested_one)(s, k) for s, k in part)
        for rec, arrays in res:
            np.savez_compressed(out_dir / f"{rec['stage']}_{rec['fold']:02d}.npz", **arrays)
            with log.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"  nested {rec['stage']} fold {rec['fold'] + 1}/15 ({rec['seconds']:.0f}s)", flush=True)


def _records(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- step 4b: bootstrap
def boot_rows() -> np.ndarray:
    U = universe()
    return np.random.default_rng(SEED_BOOT).integers(0, U.n, size=(N_BOOT, U.n))


def _boot_one(b: int, rows: np.ndarray):
    import warnings
    warnings.filterwarnings("ignore")
    _threads1()
    t0 = time.time()
    U = universe()
    y = grades(U)
    rec, arrays = {"b": b, "stages": {}}, {}
    for stage in STAGES:
        T = rows[U.elig[stage][rows]]
        P = U.rows(stage)
        m = O.fit(U.X[stage].iloc[T], y[T], K, SEED, groups=T)
        rec["stages"][stage] = {"knots": int(m.settings["knots"]), "C": float(m.settings["C"]),
                                "C_at_grid_end": bool(m.settings["C_at_grid_end"]), "n_coef": m.n_coef,
                                "n": int(len(T)), "unique": int(len(np.unique(T)))}
        arrays[f"{stage}__lp"] = m.lp(U.X[stage].iloc[P])
        arrays[f"{stage}__theta"] = m.theta
        if stage in LGB_STAGES:
            ms = fit_lgb(U.X[stage].iloc[T], y[T], lgb_settings("primary", stage), stage)
            arrays[f"{stage}__lgb_cum"] = lgb_cum(ms, U.X[stage].iloc[P])
    rec["seconds"] = round(time.time() - t0, 1)
    return rec, arrays


def bootstrap(n_boot: int = N_BOOT, n_jobs: int = 2, max_seconds: float | None = None) -> None:
    """Section 6: the full-pipeline bootstrap (500 resamples, seed 2026, the resamples of v1.0.0 and
    v1.1; inner folds keep a patient's copies together). Checkpointed by resample: each resample's
    model, as its linear predictor and thresholds on the original patients (private)."""
    from joblib import Parallel, delayed
    out_dir = PRIV / "boot"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = PRIV / "boot_records.jsonl"
    done = {r["b"] for r in _records(log)}
    todo = [b for b in range(n_boot) if b not in done]
    IDX = boot_rows()
    t_start = time.time()
    print(f"bootstrap: {len(done)} done, {len(todo)} to do ({now()})", flush=True)
    for start in range(0, len(todo), n_jobs):
        if max_seconds and time.time() - t_start > max_seconds:
            break
        part = todo[start:start + n_jobs]
        res = Parallel(n_jobs=n_jobs)(delayed(_boot_one)(b, IDX[b]) for b in part)
        for rec, arrays in res:
            np.savez_compressed(out_dir / f"b{rec['b']:03d}.npz", **arrays)
            with log.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        print(f"bootstrap: {len(done) + start + len(part)}/{n_boot} ({now()}, {max(r['seconds'] for r, _ in res):.0f}s)", flush=True)


# --------------------------------------------------------------------------- step 4c: summaries
def _summ(v) -> dict:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"mean": None, "sd": None, "lo": None, "hi": None}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}


def _binflat(m: dict) -> dict:
    return {f"{d}_{k}": m[d][k] for d in ("mrs02", "death") for k in BIN_MEASURES}


def summarise_nested() -> dict:
    """Mean and spread over the 15 outer folds; paired differences fold by fold."""
    U = universe()
    y = grades(U)
    recs = _records(PRIV / "nested_records.jsonl")
    out = {}
    for stage in STAGES:
        rr = sorted([r for r in recs if r["stage"] == stage], key=lambda r: r["fold"])
        if len(rr) < 15:
            continue
        labs = list(rr[0]["models"])
        F = {lab: [flat(r["models"][lab]) for r in rr] for lab in labs}
        B = [_binflat(r["binary"]) for r in rr]
        res = {"folds": len(rr), "models": {}, "differences": {}, "binary": {}}
        for lab in labs:
            res["models"][lab] = {k: _summ([f[k] for f in F[lab]]) for k in F[lab][0]}
        res["binary"] = {k: _summ([b[k] for b in B]) for k in B[0]}
        pairs = [(ORD, LAYER), (HYB, LAYER), (ORD, HYB)] + ([(LGB, ORD)] if LGB in labs else [])
        for a, b in pairs:
            keys = [k for k in F[a][0] if k in F[b][0]]
            res["differences"][f"{a} - {b}"] = {k: _summ([fa[k] - fb[k] for fa, fb in zip(F[a], F[b])]) for k in keys}
        for lab in (ORD, HYB, LAYER):
            res["differences"][f"{lab} - {BIN}"] = {k: _summ([f[k] - b[k] for f, b in zip(F[lab], B)]) for k in B[0]}
        # learner's zero-probability grades and the ordinal model's settings across folds
        low = {}
        for i, lab in enumerate((ORD, LAYER, HYB, LGB)):
            if lab not in labs:
                continue
            cnt = 0
            for r in rr:
                a = np.load(PRIV / "nested" / f"{stage}_{r['fold']:02d}.npz")
                Pm, rows = a[f"P_{i}"], a["rows"]
                cnt += int(np.sum(Pm[np.arange(len(rows)), y[rows]] < 1e-6))
            low[lab] = cnt
        res["observed_grade_below_1e-6"] = low
        st = [r["ordinal_settings"] for r in rr]
        res["ordinal_settings"] = {"knots_share": {str(k): float(np.mean([s["knots"] == k for s in st])) for k in (3, 4, 5)},
                                   "log10_C_median": float(np.median(np.log10([s["C"] for s in st]))),
                                   "share_C_at_grid_end": float(np.mean([s["C_at_grid_end"] for s in st])),
                                   "n_coef_median": float(np.median([s["n_coef"] for s in st]))}
        res["seconds_median"] = float(np.median([r["seconds"] for r in rr]))
        out[stage] = res
    return out


def apparent(stage: str) -> dict:
    """The development-year models (2005-2022, the stage's whole cohort) and their apparent measures."""
    U = universe()
    y = grades(U)
    P = U.rows(stage)
    X = U.X[stage].iloc[P]
    m = O.fit(X, y[P], K, SEED)
    out = {"model": m, "P": m.predict(X), "lp": m.lp(X)}
    out["measures"] = dist_measures(y[P], out["P"], out["lp"])
    if stage in LGB_STAGES:
        ms = fit_lgb(X, y[P], lgb_settings("primary", stage), stage)
        out["P_lgb"] = cum_to_P(lgb_cum(ms, X))
        out["measures_lgb"] = dist_measures(y[P], out["P_lgb"])
    return out


def summarise_bootstrap(app: dict) -> dict:
    """Harrell's optimism correction, stability and settings across the resamples (section 6)."""
    U = universe()
    y = grades(U)
    IDX = boot_rows()
    recs = sorted(_records(PRIV / "boot_records.jsonl"), key=lambda r: r["b"])
    out = {"n_boot": len(recs), "seed_bootstrap": SEED_BOOT, "stages": {}}
    for stage in STAGES:
        P = U.rows(stage)
        yP = y[P]
        res = {}
        for lab in [ORD] + ([LGB] if stage in LGB_STAGES else []):
            opt, est = [], {"p02": [], "pdeath": [], "uw": []}
            for r in recs:
                a = np.load(PRIV / "boot" / f"b{r['b']:03d}.npz")
                if lab == ORD:
                    lp, th = a[f"{stage}__lp"], a[f"{stage}__theta"]
                    Pb = O.probs(th, lp)
                else:
                    lp, Pb = None, cum_to_P(a[f"{stage}__lgb_cum"])
                rows = IDX[r["b"]]
                T = rows[U.elig[stage][rows]]
                pos = np.searchsorted(P, T)
                m_app = flat(dist_measures(y[T], Pb[pos], None if lp is None else lp[pos]))
                m_test = flat(dist_measures(yP, Pb, lp))
                opt.append({k: m_app[k] - m_test[k] for k in m_test})
                est["p02"].append(Pb[:, :3].sum(1))
                est["pdeath"].append(Pb[:, 6])
                est["uw"].append(Pb @ UW)
            a0 = flat(app[stage]["measures"] if lab == ORD else app[stage]["measures_lgb"])
            meas = {}
            for k in a0:
                o = np.array([d[k] for d in opt], float)
                meas[k] = {"apparent": a0[k], "optimism": float(o.mean()), "monte_carlo_se": float(o.std(ddof=1) / np.sqrt(len(o))),
                           "corrected": float(a0[k] - o.mean())}
            stab = {}
            for k, v in est.items():
                E = np.stack(v)
                w = np.percentile(E, 97.5, axis=0) - np.percentile(E, 2.5, axis=0)
                stab[k] = {"width_median": float(np.median(w)), "width_p90": float(np.percentile(w, 90))}
            res[lab] = {"measures": meas, "stability": stab}
        st = [r["stages"][stage] for r in recs]
        res["ordinal_settings"] = {"knots_share": {str(k): float(np.mean([s["knots"] == k for s in st])) for k in (3, 4, 5)},
                                   "log10_C_median": float(np.median(np.log10([s["C"] for s in st]))),
                                   "log10_C_iqr": [float(np.percentile(np.log10([s["C"] for s in st]), q)) for q in (25, 75)],
                                   "share_C_at_grid_end": float(np.mean([s["C_at_grid_end"] for s in st])),
                                   "n_coef_median": float(np.median([s["n_coef"] for s in st]))}
        out["stages"][stage] = res
    return out


def brant(app: dict) -> dict:
    """Rule O4 (descriptive): a penalised binary model of mRS > k at each cut-point, with the
    development model's layout and penalty, against the shared linear predictor."""
    U = universe()
    y = grades(U)
    out = {}
    for stage in STAGES:
        m = app[stage]["model"]
        P = U.rows(stage)
        Z = m.layout.Z(U.X[stage].iloc[P])
        lp = Z @ m.coef
        res = []
        for k in range(K - 1):
            yb = (y[P] > k).astype(float)
            beta = R.newton(R._with_ones(Z), yb, np.ones(len(yb)), float(m.settings["C"]))
            lk = Z @ beta[:-1]
            c = np.cov(lk, lp)
            res.append({"cut": f"mRS > {k}", "events": int(yb.sum()), "lp_slope": float(c[0, 1] / c[1, 1]),
                        "lp_correlation": float(np.corrcoef(lk, lp)[0, 1]),
                        "coef_correlation": float(np.corrcoef(beta[:-1], m.coef)[0, 1])})
        out[stage] = res
    return out


def approximate_r2(auc: float, prev: float, n: int = 1_000_000, seed: int = SEED_BOOT) -> float:
    """Cox-Snell R2 implied by a C statistic (Riley et al. 2021, as pmsampsize): two normal linear
    predictors one SD apart per sqrt(2) * probit(C), a logistic model refitted on them."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    ne = int(round(prev * n))
    mu = np.sqrt(2) * norm.ppf(auc)
    x = np.concatenate([rng.normal(mu, 1, ne), rng.normal(0, 1, n - ne)])
    yy = np.concatenate([np.ones(ne), np.zeros(n - ne)])
    w = np.zeros(2)
    A = np.column_stack([np.ones(n), x])
    for _ in range(50):
        q = expit(A @ w)
        step = np.linalg.solve((A * (q * (1 - q))[:, None]).T @ A, A.T @ (yy - q))
        w += step
        if np.max(np.abs(step)) < 1e-10:
            break
    q = np.clip(expit(A @ w), 1e-300, 1 - 1e-16)
    ll1 = float(np.sum(yy * np.log(q) + (1 - yy) * np.log(1 - q)))
    ll0 = float(ne * np.log(prev) + (n - ne) * np.log(1 - prev))
    return float(1 - np.exp(-2 * (ll1 - ll0) / n))


def riley(p: int, prev: float, r2: float, S: float = 0.9, delta: float = 0.05, margin: float = 0.05) -> dict:
    r2max = 1 - (prev ** prev * (1 - prev) ** (1 - prev)) ** 2
    n1 = p / ((S - 1) * np.log(1 - r2 / S))
    s_vh = r2 / (r2 + delta * r2max)
    n2 = p / ((s_vh - 1) * np.log(1 - r2 / s_vh))
    n3 = (1.96 / margin) ** 2 * prev * (1 - prev)
    return {"r2_cs": float(r2), "r2_cs_max": float(r2max), "n_shrinkage": int(np.ceil(n1)), "n_optimism": int(np.ceil(n2)),
            "n_intercept": int(np.ceil(n3)), "n_required": int(np.ceil(max(n1, n2, n3)))}


def sample_size(app: dict) -> dict:
    v = json.loads((paths.PUBLIC / "validation.json").read_text())["v1_1"]["rows"]
    U = universe()
    y = grades(U)
    out = {}
    for stage in STAGES:
        P = U.rows(stage)
        p = app[stage]["model"].n_coef
        res = []
        for k in range(K - 1):
            prev = float(np.mean(y[P] > k))
            r = {"cut": f"mRS > {k}", "n": int(len(P)), "events": int(np.sum(y[P] > k)), "parameters": int(p),
                 "events_per_parameter": float(min(prev, 1 - prev) * len(P) / p)}
            r2max = 1 - (prev ** prev * (1 - prev) ** (1 - prev)) ** 2
            r["conservative"] = riley(p, prev, 0.15 * r2max)
            auc = {2: v[stage]["good_outcome"]["nested_auroc_v11"], 5: v[stage]["death90"]["nested_auroc_v11"]}.get(k)
            if auc:
                r["from_v11_auroc"] = {"auroc": auc, **riley(p, prev, approximate_r2(auc, prev))}
            res.append(r)
        out[stage] = res
    return out


def decisions(nested_s: dict, boot_s: dict, check_res: dict) -> dict:
    """Rules O1-O3 (section 7), from the development years only."""
    d = {"O1": {}, "O2": {}, "O3": {}}
    for st in PRIMARY:
        n = nested_s[st]["models"]
        d["O1"][st] = {"rps_ordinal_shares": n[HYB]["rps"]["mean"], "rps_layer": n[LAYER]["rps"]["mean"],
                       "lower": bool(n[HYB]["rps"]["mean"] < n[LAYER]["rps"]["mean"])}
        a_ord, a_bin = n[ORD]["mrs02_auroc"]["mean"], nested_s[st]["binary"]["mrs02_auroc"]["mean"]
        d["O2"][st] = {"auroc_implied_mrs02": a_ord, "auroc_v11_binary": a_bin, "difference": a_ord - a_bin,
                       "within_0_02": bool(a_ord - a_bin >= -0.02)}
    d["O1"]["replace_layer"] = all(d["O1"][st]["lower"] for st in PRIMARY)
    d["O2"]["pipeline_check_passed"] = bool(check_res["passed"])
    d["O2"]["send_to_external_validation"] = bool(check_res["passed"] and all(d["O2"][st]["within_0_02"] for st in PRIMARY))
    for st in STAGES:
        sl = boot_s["stages"][st][ORD]["measures"]["ord_slope"]["corrected"]
        app_sl = boot_s["stages"][st][ORD]["measures"]["ord_slope"]["apparent"]
        out_of = bool(sl < 0.9 or sl > 1.1)
        d["O3"][st] = {"corrected_slope": sl, "apparent_slope": app_sl, "apply": out_of, "factor": sl if out_of else 1.0}
    return d


def develop() -> dict:
    """Step 4: every development-year result and the decisions, written before any test-period result."""
    from .export import clean_json
    check_res = json.loads((PRIV / "pipeline_check.json").read_text())
    t0 = time.time()
    app = {st: apparent(st) for st in STAGES}
    print(f"  apparent models ({time.time() - t0:.0f}s)", flush=True)
    nested_s = summarise_nested()
    boot_s = summarise_bootstrap(app)
    print(f"  bootstrap summarised ({time.time() - t0:.0f}s)", flush=True)
    dec = decisions(nested_s, boot_s, check_res)
    out = {"generated": now(), "plan": "docs/registrations/ordinal_plan.md (with amendment 1)", "pipeline_check": check_res,
           "development_models": {st: {"settings": app[st]["model"].settings, "n_coef": app[st]["model"].n_coef,
                                       "theta": [float(v) for v in app[st]["model"].theta],
                                       "n": int(app[st]["measures"]["n"])} for st in STAGES},
           "nested": nested_s, "bootstrap": boot_s, "brant": brant(app), "sample_size": sample_size(app),
           "decisions": dec}
    out = clean_json(out)
    (PRIV / "development.json").write_text(json.dumps(out, indent=1))
    (PRIV / "decisions.json").write_text(json.dumps(dec, indent=1))
    return out


# --------------------------------------------------------------------------- step 5: temporal tests
class SimplePO:
    """Proportional-odds refit of a comparator's predictor set (amendment 1, item 10): median
    imputation with missing-value indicators, standardisation, L2 penalty by 5-fold CV on ordinal
    log-loss (folds stratified by grade), as the binary refits' pipeline."""

    def __init__(self, D: pd.DataFrame, y, seed: int = SEED):
        from sklearn.model_selection import StratifiedKFold

        from .lr import C_GRID
        A = D.to_numpy(float)
        self.columns = list(D.columns)
        self.med = np.nanmedian(A, axis=0)
        self.ind = [j for j in range(A.shape[1]) if np.isnan(A[:, j]).any()]
        R_ = self._raw(A)
        self.mean, self.scale = R_.mean(0), R_.std(0)
        self.scale[self.scale == 0] = 1.0
        Z = (R_ - self.mean) / self.scale
        y = np.asarray(y, int)
        folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Z, y))
        S = O.path_scores(Z, y, np.ones(len(y)), K, folds)
        j = int(np.argmin(S.mean(0)))
        self.C = float(C_GRID[j])
        par = O.newton(Z, y, np.ones(len(y)), self.C, K)
        self.coef, self.theta = par[: Z.shape[1]], par[Z.shape[1]:]

    def _raw(self, A):
        F = np.where(np.isnan(A), self.med[None, :], A)
        return np.hstack([F, np.isnan(A[:, self.ind]).astype(float)]) if self.ind else F

    def lp(self, D):
        return ((self._raw(D[self.columns].to_numpy(float)) - self.mean) / self.scale) @ self.coef

    def predict(self, D):
        return O.probs(self.theta, self.lp(D))


def nihss_design(df: pd.DataFrame, col: str, knots=None):
    from .splines import rcs_terms
    x = df[col].to_numpy(float)
    if knots is None:
        knots = [float(v) for v in np.unique(np.nanpercentile(x, [10, 50, 90]))]
    D = pd.DataFrame({col: x}, index=df.index)
    if len(knots) == 3:
        D[col + "__rcs1"] = rcs_terms(x, knots)[:, 0]
    return D, knots


COMPARATOR_SETS = {"admission": ("MR PREDICTS",), "procedure": ("MR PREDICTS",),
                   "h24": ("MR PREDICTS@24H",), "d7": ("MR PREDICTS@24H",)}
NIHSS_COL = {"h24": "nihss_24h", "d7": "nihss_7d"}


def _test_predictions(U, stage, Td, Te, split: str, dec: dict) -> tuple[dict, dict]:
    from .scores import REFIT_TERMS, design
    y = grades(U)
    X = U.X[stage]
    m = O.fit(X.iloc[Td], y[Td], K, SEED)
    f = float(dec["O3"][stage]["factor"])
    if f != 1.0:
        m = O.with_slope(m, X.iloc[Td], y[Td], f)
    P_ord, lp = m.predict(X.iloc[Te]), m.lp(X.iloc[Te])
    bins = {t: fit_binary(X.iloc[Td], U.y[t][Td], SEED, stage, t) for t in TARGETS}
    pg, pdeath = expit(bins["good_outcome"].logit(X.iloc[Te])), expit(bins["death90"].logit(X.iloc[Te]))
    lay = layer_fit(U, stage, Td, SEED, bins, y)
    preds = {ORD: P_ord, LAYER: layer_dist(pg, pdeath, lay), HYB: hybrid_dist(pg, pdeath, P_ord)}
    lps = {ORD: lp}
    info = {"ordinal": {**m.settings, "n_coef": m.n_coef}}
    dfd, dft = U.df.iloc[Td], U.df.iloc[Te]
    for name in COMPARATOR_SETS[stage]:
        terms = REFIT_TERMS[name]
        cm = SimplePO(design(dfd, terms), y[Td])
        lab = f"{name} predictors, proportional odds" + (" (no collaterals)" if name == "MR PREDICTS" else "")
        preds[lab], lps[lab] = cm.predict(design(dft, terms)), cm.lp(design(dft, terms))
        info[lab] = {"C": cm.C, "n_coef": len(cm.coef)}
    if stage in NIHSS_COL:
        Dd, kn = nihss_design(dfd, NIHSS_COL[stage])
        Dt, _ = nihss_design(dft, NIHSS_COL[stage], kn)
        cm = SimplePO(Dd, y[Td])
        lab = "NIHSS alone, proportional odds"
        preds[lab], lps[lab] = cm.predict(Dt), cm.lp(Dt)
        info[lab] = {"C": cm.C, "knots": kn}
    if stage in LGB_STAGES:
        preds[LGB] = cum_to_P(lgb_cum(fit_lgb(X.iloc[Td], y[Td], lgb_settings(split, stage), stage), X.iloc[Te]))
    preds[BIN] = (pg, pdeath)
    return preds, {"lps": lps, "info": info}


def _measures_any(lab, y, pr, lp=None):
    if lab == BIN:
        pg, pdeath = pr
        return {"n": int(len(y)), "mrs02": v11.measures((y <= 2).astype(float), pg),
                "death": v11.measures((y == 6).astype(float), pdeath)}
    return dist_measures(y, pr, lp)


def _flat_any(lab, m):
    return _binflat(m) if lab == BIN else flat(m)


def _subset(pr, idx):
    return (pr[0][idx], pr[1][idx]) if isinstance(pr, tuple) else pr[idx]


def temporal(n_boot: int = 2000) -> dict:
    """Section 8: the development-year models tested on 2023-2025 (primary) and 2021-2025
    (sensitivity), with 2,000 paired bootstrap resamples of the test patients. Reported, not deciding;
    not independent (the specification is v1.1's)."""
    from .export import clean_json
    dec = json.loads((PRIV / "decisions.json").read_text())
    U = universe(None)
    y = grades(U)
    out = {"generated": now(), "n_boot": n_boot, "splits": {}}
    for split, test_from in (("primary", 2023), ("sensitivity", 2021)):
        dev, test = U.year < test_from, U.year >= test_from
        out["splits"][split] = {"test_from": test_from, "stages": {}}
        for stage in STAGES:
            t0 = time.time()
            Td, Te = U.rows(stage, dev), U.rows(stage, test)
            preds, extra = _test_predictions(U, stage, Td, Te, split, dec)
            yt = y[Te]
            lps = extra["lps"]
            res = {"n_dev": int(len(Td)), "n_test": int(len(Te)), "grades_test": [int(np.sum(yt == g)) for g in range(K)],
                   "models": {lab: _measures_any(lab, yt, pr, lps.get(lab)) for lab, pr in preds.items()},
                   "settings": extra["info"]}
            rng = np.random.default_rng(SEED)
            boots = {lab: [] for lab in preds}
            for _ in range(n_boot):
                idx = rng.integers(0, len(yt), len(yt))
                for lab, pr in preds.items():
                    lp = lps.get(lab)
                    boots[lab].append(_flat_any(lab, _measures_any(lab, yt[idx], _subset(pr, idx), None if lp is None else lp[idx])))
            ci, diff = {}, {}
            for lab in preds:
                keys = boots[lab][0].keys()
                ci[lab] = {k: [float(np.nanpercentile([b[k] for b in boots[lab]], q)) for q in (2.5, 97.5)] for k in keys}
                if lab == ORD:
                    continue
                common = [k for k in keys if k in boots[ORD][0]]
                pt_lab, pt_ord = _flat_any(lab, res["models"][lab]), flat(res["models"][ORD])
                diff[f"{lab} - {ORD}"] = {k: {"estimate": float(pt_lab[k] - pt_ord[k]),
                                              "ci": [float(np.nanpercentile([a[k] - b[k] for a, b in zip(boots[lab], boots[ORD])], q))
                                                     for q in (2.5, 97.5)]} for k in common}
            res["ci"], res["differences"] = ci, diff
            # subgroups (section 8): sex and age, where each of mRS 0-2, 3-5 and 6 has >= 10 patients
            dft = U.df.iloc[Te]
            groups = {"men": dft["sex_male"].to_numpy() == 1, "women": dft["sex_male"].to_numpy() == 0,
                      "age under 80": dft["age"].to_numpy(float) < 80, "age 80 or older": dft["age"].to_numpy(float) >= 80}
            res["subgroups"] = {}
            for g, msk in groups.items():
                yy = yt[msk]
                bands = [int(np.sum(yy <= 2)), int(np.sum((yy >= 3) & (yy <= 5))), int(np.sum(yy == 6))]
                if min(bands) < 10:
                    res["subgroups"][g] = {"shown": False}
                    continue
                res["subgroups"][g] = {"shown": True, "n": int(msk.sum()), "models": {
                    lab: _flat_any(lab, _measures_any(lab, yy, _subset(pr, msk), None if lps.get(lab) is None else lps[lab][msk]))
                    for lab, pr in preds.items()}}
            out["splits"][split]["stages"][stage] = res
            if split == "primary":
                pid = U.df["patient_id"].to_numpy()[Te]
                np.savez_compressed(PRIV / f"temporal_{stage}.npz", patient_id=pid.astype(str), y=yt,
                                    **{f"P{i}": (np.column_stack(pr) if isinstance(pr, tuple) else pr) for i, pr in enumerate(preds.values())})
            print(f"  temporal {split} {stage}: {len(Te)} test patients ({time.time() - t0:.0f}s)", flush=True)
    out = clean_json(out)
    (PRIV / "temporal.json").write_text(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------- step 6: deployment
def shift_cum(L, y, tol: float = 1e-12) -> float:
    """Maximum-likelihood common shift d of per-patient cumulative log-odds L (n x 6):
    P(mRS <= k) = expit(L_k + d) (amendment 1, item 11)."""
    y = np.asarray(y, int)
    n = len(y)
    Le = np.hstack([np.full((n, 1), -np.inf), L, np.full((n, 1), np.inf)])
    a0, b0 = Le[np.arange(n), y + 1], Le[np.arange(n), y]
    d = 0.0
    for _ in range(100):
        a, b = a0 + d, b0 + d
        pi = np.maximum(O._pi(a, b), 1e-300)
        _, fa, fpa = O._cdf_pdf(a)
        _, fb, fpb = O._cdf_pdf(b)
        fa, fpa = np.where(np.isfinite(a), fa, 0.0), np.where(np.isfinite(a), fpa, 0.0)
        fb, fpb = np.where(np.isfinite(b), fb, 0.0), np.where(np.isfinite(b), fpb, 0.0)
        g = np.sum((fa - fb) / pi)                                   # d log-likelihood / d d
        h = np.sum((fpa - fpb) / pi - ((fa - fb) / pi) ** 2)          # second derivative (< 0)
        step = g / h
        d -= step
        if abs(step) < tol:
            break
    return float(d)


def slope_with_se(eta, y) -> dict:
    """Ordinal calibration slope with its Wald 95 % interval (observed information)."""
    sl, th = O.calibration_slope(eta, y, K)
    g, H = O.grad_hess(np.r_[sl, th], np.asarray(eta, float)[:, None], np.asarray(y, int), np.ones(len(y)), 1.0, K, 0.0, 0.0)
    se = float(np.sqrt(np.linalg.inv(H)[0, 0]))
    return {"slope": sl, "se": se, "lo": sl - 1.96 * se, "hi": sl + 1.96 * se}


def deploy() -> dict:
    """Section 11: refitted on every patient of 2005-2025 with the development-year settings and rule
    O3's factor; the recent-era threshold shift from out-of-fold estimates in 2023-2025; the slope only
    under the update policy."""
    from sklearn.model_selection import StratifiedKFold

    from .export import clean_json
    dec = json.loads((PRIV / "decisions.json").read_text())
    devj = json.loads((PRIV / "development.json").read_text())
    U = universe(None)
    y = grades(U)
    recent = U.year >= 2023
    out, models = {}, {}
    for stage in STAGES:
        t0 = time.time()
        T = U.rows(stage)
        X, yT = U.X[stage].iloc[T], y[T]
        st = {k: devj["development_models"][stage]["settings"][k] for k in ("knots", "C")}
        f = float(dec["O3"][stage]["factor"])
        fit = lambda Xa, ya: (lambda mm: O.with_slope(mm, Xa, ya, f) if f != 1.0 else mm)(O.fit_fixed(Xa, ya, K, st))  # noqa: E731
        m = fit(X, yT)
        L = np.zeros((len(T), K - 1))
        eta_al = np.zeros(len(T))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED + 1).split(np.zeros(len(T)), yT):
            mk = fit(X.iloc[tr], yT[tr])
            eta = mk.lp(X.iloc[va])
            L[va] = mk.theta[None, :] - eta[:, None]
            eta_al[va] = eta - (mk.theta.mean() - m.theta.mean())
        rec = recent[T]
        shift = shift_cum(L[rec], yT[rec])
        pol = slope_with_se(eta_al[rec], yT[rec])
        trig = bool((pol["slope"] < 0.8 or pol["slope"] > 1.2) and not (pol["lo"] <= 1 <= pol["hi"]))
        policy = {**pol, "triggered": trig}
        if trig:      # update policy: the linear predictor times the recent slope, the shift re-estimated
            m = O.with_slope(m, X, yT, pol["slope"])
            L2 = L.copy()
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED + 1).split(np.zeros(len(T)), yT):
                mk = O.with_slope(fit(X.iloc[tr], yT[tr]), X.iloc[tr], yT[tr], pol["slope"])
                L2[va] = mk.theta[None, :] - mk.lp(X.iloc[va])[:, None]
            L = L2
            shift = shift_cum(L[rec], yT[rec])
        P_oof = cum_to_P(expit(L))
        P_rec = cum_to_P(expit(L[rec] + shift))
        phi = m.contributions(X)
        imp = phi.abs().T.groupby(lambda c: R.input_of(c)).sum().T.mean().sort_values(ascending=False)
        spec = m.export(shift)
        models[stage] = {"kind": "ordinal", "spec": spec, "columns": list(X.columns), "n_train": int(len(T)),
                         "slope_factor": f, "settings": st}
        out[stage] = clean_json({
            "settings": st, "slope_factor": f, "n_coef": m.n_coef, "n": int(len(T)), "n_recent": int(rec.sum()),
            "theta": [float(v) for v in m.theta], "threshold_shift_recent": shift, "update_policy": policy,
            "grades_recent": [int(np.sum(yT[rec] == g)) for g in range(K)],
            "oof_all": flat(dist_measures(yT, P_oof)), "oof_recent_uncalibrated": flat(dist_measures(yT[rec], P_oof[rec])),
            "oof_recent_recalibrated": flat(dist_measures(yT[rec], P_rec)),
            "mean_abs_contribution": {k: float(v) for k, v in imp.items()},
            "missing_terms": [R.input_of(m.layout.columns[j]) for j in m.layout.ind],
            "merged": {m.layout.base[j]: [m.layout.stage_columns[i] for i in src] for j, src in enumerate(m.layout.sources) if len(src) > 1}})
        import pickle
        (PRIV / f"deployed_{stage}.pkl").write_bytes(pickle.dumps(m))
        print(f"  deploy {stage}: {m.n_coef} coefficients, shift {shift:+.3f}, recent slope {pol['slope']:.2f} "
              f"({pol['lo']:.2f}-{pol['hi']:.2f}){' TRIGGERED' if trig else ''} ({time.time() - t0:.0f}s)", flush=True)
    (PRIV / "deployed.json").write_text(json.dumps(out, indent=1))
    (PRIV / "deployed_models.json").write_text(json.dumps(clean_json(models)))
    return out


ORDINAL_VERSION = "1.0.0"


def build_bundle() -> dict:
    """The MTOP ordinal bundle (section 11): coefficients, thresholds, knots, medians, scaling and
    layout only; SHA-256 fingerprint of the canonical models block."""
    import datetime as dt
    import hashlib
    models = json.loads((PRIV / "deployed_models.json").read_text())
    dec = json.loads((PRIV / "decisions.json").read_text())
    b11 = json.loads((paths.PUBLIC / "model_bundle.json").read_text())
    for stage in STAGES:
        cf = b11["models"][stage]["good_outcome"]
        assert cf["columns"] == models[stage]["columns"], stage
        models[stage]["column_feature"] = cf["column_feature"]
    b = {"app": "MTOP ordinal", "version": ORDINAL_VERSION, "generated": dt.date.today().isoformat(),
         "plan": "docs/registrations/ordinal_plan.md", "outcome": "90-day modified Rankin Scale, grades 0-6",
         "stages": list(STAGES), "grades": list(range(K)), "utility_weights": [float(v) for v in UW],
         "recent_era_from_year": 2023, "binary_version": b11["version"], "binary_fingerprint": b11["fingerprint"],
         "calculator_distribution": "ordinal shares" if dec["O1"]["replace_layer"] else "layer",
         "limited_evidence": {"d7": ["death"]}, "models": models}
    canon = json.dumps(b["models"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    b["fingerprint"] = hashlib.sha256(canon).hexdigest()
    return b


def parity(bundle: dict | None = None) -> dict:
    """Section 11: the browser (web/model.js) against the Python model, on every patient of each
    stage (2005-2025) and on the synthetic demonstration cohort; criterion 1e-7."""
    import pickle
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    from .features import design_matrix
    bundle = bundle or build_bundle()
    node = shutil.which("node")
    from .clean import clean_registry, load_raw
    U = universe(None)
    demo, _ = clean_registry(load_raw(paths.SYNTHETIC / "registry_synthetic.xlsx"))   # the demonstration cohort
    demo = demo[demo["age"] >= 18]
    out = {"node": node is not None, "stages": {}}
    for stage in STAGES:
        Lm = bundle["models"][stage]
        m = pickle.loads((PRIV / f"deployed_{stage}.pkl").read_bytes())
        res = {}
        for name, X in (("registry", U.X[stage].iloc[U.rows(stage)]), ("synthetic", design_matrix(demo, stage).reindex(columns=Lm["columns"], fill_value=0.0))):
            lp_e, P_e, phi_e = O.eval_export(Lm["spec"], X)
            lp_m = m.lp(X)
            P_m = O.probs(m.theta + Lm["spec"]["theta_shift"], lp_m)
            r = {"n": int(len(X)), "export_vs_model": float(max(np.max(np.abs(lp_e - lp_m)), np.max(np.abs(P_e - P_m)),
                                                                np.max(np.abs(phi_e - m.contributions(X).to_numpy()))))}
            if node:
                with tempfile.TemporaryDirectory() as td:
                    inp, outp = Path(td) / "in.json", Path(td) / "out.json"
                    rows = [[None if np.isnan(v) else float(v) for v in row] for row in X.to_numpy(float)]
                    inp.write_text(json.dumps({"model": {"kind": "ordinal", "spec": Lm["spec"]}, "rows": rows}))
                    subprocess.run([node, str(paths.ROOT / "tests" / "node_ordinal.js"), str(inp), str(outp)], check=True)
                    js = json.loads(outp.read_text())
                r["browser_vs_model"] = float(max(np.max(np.abs(np.array([q["lp"] for q in js]) - lp_m)),
                                                  np.max(np.abs(np.array([q["p"] for q in js]) - P_m)),
                                                  np.max(np.abs(np.array([q["phi"] for q in js]) - m.contributions(X).to_numpy()))))
            r["passed"] = bool(r["export_vs_model"] < 1e-7 and r.get("browser_vs_model", 0.0) < 1e-7)
            res[name] = r
        out["stages"][stage] = res
    out["passed"] = bool(node is not None and all(r["passed"] for s_ in out["stages"].values() for r in s_.values()))
    (PRIV / "parity.json").write_text(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------- results file
STAGE_EN = v11.STAGE_EN


def _f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}".replace("-", "−")


def _sg(x, d=3):
    return "–" if x is None else (f"+{x:.{d}f}" if x >= 0 else f"−{abs(x):.{d}f}")


def _rng(s, d=3, signed=False):
    if s is None or s.get("mean") is None:
        return "–"
    f = _sg if signed else _f
    return f"{f(s['mean'], d)} ({f(s['lo'], d)} to {f(s['hi'], d)})"


def report_md(path=None) -> str:
    """docs/registrations/ordinal_results.md, written from the analysis outputs (plan, section 13)."""
    path = path or (paths.ROOT / "docs" / "registrations" / "ordinal_results.md")
    dev = json.loads((PRIV / "development.json").read_text())
    dec = dev["decisions"]
    tmp_path, dep_path = PRIV / "temporal.json", PRIV / "deployed.json"
    L = []
    w = L.append
    w("# MTOP ordinal model: results of the registered evaluation")
    w("")
    w(f"Written {dev['generated'][:10]} by `python -m mtop.ordinal_run report` from the analysis outputs. Plan: "
      "[ordinal_plan.md](ordinal_plan.md) (22 November 2026) with amendment 1, both committed before any ordinal-model "
      "result was computed. The development years 2005–2022 decide (sections 1–6); the decisions were committed before "
      "any 2023–2025 or 2021–2025 result of the ordinal model or of the comparison-only learner was computed.")
    w("")
    w("Model: a penalised proportional-odds regression on the seven 90-day mRS grades with the inputs of the v1.1 "
      "reference at each stage. Comparators within the development years: the v1.1 proportional-odds layer (which splits "
      "the binary models' bands into grades), the v1.1 bands split by the ordinal model's shares (rule O1's alternative), "
      "the v1.1 binary models (for the implied mRS 0–2 and death), and an ordinal LightGBM (comparison only).")
    w("")
    # ---------------------------------------------------------------- decisions
    w("## Decisions (section 7)")
    w("")
    o1 = dec["O1"]
    w(f"- **O1, the calculator's distribution: {'the ordinal model replaces the layer' if o1['replace_layer'] else 'the layer stays'}.** "
      "Nested cross-validated ranked probability score (RPS, lower is better) of the v1.1 bands split by the ordinal "
      "model's shares against the layer's: " + "; ".join(
          f"{STAGE_EN[st]} {_f(o1[st]['rps_ordinal_shares'], 4)} against {_f(o1[st]['rps_layer'], 4)}" for st in PRIMARY)
      + ". The rule needs a lower score at both primary stages.")
    o2 = dec["O2"]
    w(f"- **O2, external validation: {'sent as a secondary model' if o2['send_to_external_validation'] else 'not sent'}.** "
      f"Pipeline check {'passed' if o2['pipeline_check_passed'] else 'failed'}; nested cross-validated AUROC of the "
      "implied P(mRS 0–2) against the v1.1 binary model: " + "; ".join(
          f"{STAGE_EN[st]} {_f(o2[st]['auroc_implied_mrs02'])} against {_f(o2[st]['auroc_v11_binary'])} ({_sg(o2[st]['difference'])})"
          for st in PRIMARY) + ". The model is not sent if it is more than 0.02 lower at a primary stage.")
    o3 = dec["O3"]
    ap = [st for st in STAGES if o3[st]["apply"]]
    w("- **O3, slope correction: " + (("applied at " + ", ".join(f"{STAGE_EN[st]} (× {o3[st]['factor']:.3f})" for st in ap)) if ap else "not applied")
      + ".** Optimism-corrected ordinal calibration slope: " + "; ".join(f"{STAGE_EN[st]} {_f(o3[st]['corrected_slope'])}" for st in STAGES)
      + " (applied only outside 0.9–1.1).")
    w("- **O4, proportional odds:** descriptive, section 4.")
    w("")
    # ---------------------------------------------------------------- pipeline check
    chk = dev["pipeline_check"]
    w("## 1. Pipeline check (section 5, step 3)")
    w("")
    w(f"- Registry export SHA-256 `{chk['export_sha256']}`: {'identical to v1.1' if chk['export_identical'] else 'DIFFERENT'}.")
    w("- Development and test patients at each stage: " + "; ".join(
        f"{STAGE_EN[st]} {c['dev']} and {c['test']}" for st, c in chk["cohorts"].items())
      + (" (identical to v1.1)." if all(c["identical"] for c in chk["cohorts"].values()) else " (NOT identical)."))
    w(f"- Unpenalised ordinal model (age and admission NIHSS) against statsmodels: largest difference "
      f"{chk['statsmodels_max_abs_diff']:.1e} (criterion 10⁻⁶).")
    w("- Sparse grades: every grade has at least 20 development patients at every stage; no grade is merged.")
    w(f"- **Passed: {'yes' if chk['passed'] else 'no'}.**")
    w("")
    # ---------------------------------------------------------------- nested
    w("## 2. Nested cross-validation, 2005–2022 (5 outer folds × 3 repeats)")
    w("")
    w("Mean over the 15 outer folds (2.5th to 97.5th percentile of the folds in brackets, a spread, not a confidence "
      "interval). RPS: ranked probability score; log-loss: ordinal log-loss; C (utility): generalised concordance on the "
      "expected utility-weighted mRS; C (grade): on the expected grade; AUROC for the implied dichotomies; slope: "
      "calibration slope of the implied P(mRS 0–2); utility: mean predicted minus observed utility-weighted mRS.")
    w("")
    for st in STAGES:
        n = dev["nested"].get(st)
        if not n:
            continue
        nd = dev["development_models"][st]["n"]
        w(f"### {STAGE_EN[st].capitalize()} ({nd} patients)")
        w("")
        w("| Model | RPS | Log-loss | C (utility) | C (grade) | AUROC 0–1 | AUROC 0–2 | AUROC 0–3 | AUROC death | Slope 0–2 | Utility |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for lab, m in n["models"].items():
            w(f"| {lab} | {_f(m['rps']['mean'], 4)} | {_f(m['ologloss']['mean'], 3)} | {_f(m['genc_uw']['mean'])} | "
              f"{_f(m['genc_grade']['mean'])} | {_f(m['mrs01_auroc']['mean'])} | {_f(m['mrs02_auroc']['mean'])} | "
              f"{_f(m['mrs03_auroc']['mean'])} | {_f(m['death_auroc']['mean'])} | {_f(m['mrs02_slope']['mean'])} | "
              f"{_sg(m['uw_diff']['mean'])} |")
        b = n["binary"]
        w(f"| {BIN} | – | – | – | – | – | {_f(b['mrs02_auroc']['mean'])} | – | {_f(b['death_auroc']['mean'])} | "
          f"{_f(b['mrs02_slope']['mean'])} | – |")
        w("")
        w("Paired differences, fold by fold:")
        w("")
        w("| Comparison | RPS | Log-loss | C (utility) | AUROC 0–2 | AUROC death |")
        w("|---|---|---|---|---|---|")
        for key, dd in n["differences"].items():
            if key.endswith(BIN):
                w(f"| {key.replace(' - ', ' − ')} | – | – | – | {_rng(dd['mrs02_auroc'], 3, True)} | {_rng(dd['death_auroc'], 3, True)} |")
            else:
                w(f"| {key.replace(' - ', ' − ')} | {_rng(dd['rps'], 4, True)} | {_rng(dd['ologloss'], 3, True)} | {_rng(dd['genc_uw'], 3, True)} | "
                  f"{_rng(dd['mrs02_auroc'], 3, True)} | {_rng(dd['death_auroc'], 3, True)} |")
        w("")
        low = n["observed_grade_below_1e-6"]
        os_ = n["ordinal_settings"]
        w(f"Ordinal model across folds: knots 3/4/5 in {', '.join(f'{100 * os_['knots_share'][k]:.0f} %' for k in ('3', '4', '5'))} "
          f"of folds; median log10 C {_f(os_['log10_C_median'], 2)}; penalty at the end of the grid in "
          f"{100 * os_['share_C_at_grid_end']:.0f} %; median {os_['n_coef_median']:.0f} coefficients. Patients whose "
          "observed grade had a probability below 10⁻⁶: " + ", ".join(f"{k} {v}" for k, v in low.items()) + ".")
        w("")
    # binary consistency
    w("### Consistency with the binary models")
    w("")
    w("Implied P(mRS 0–2) and P(death) of the ordinal model against the v1.1 binary models on the same folds (mean "
      "over folds).")
    w("")
    w("| Stage | Outcome | AUROC ordinal | AUROC binary | Log-loss ordinal | Log-loss binary | CITL ordinal | CITL binary | Slope ordinal | Slope binary |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for st in STAGES:
        n = dev["nested"].get(st)
        if not n:
            continue
        mo, b = n["models"][ORD], n["binary"]
        for d, lab in (("mrs02", "mRS 0–2"), ("death", "death")):
            w(f"| {STAGE_EN[st].capitalize()} | {lab} | {_f(mo[d + '_auroc']['mean'])} | {_f(b[d + '_auroc']['mean'])} | "
              f"{_f(mo[d + '_logloss']['mean'], 4)} | {_f(b[d + '_logloss']['mean'], 4)} | {_sg(mo[d + '_citl']['mean'])} | "
              f"{_sg(b[d + '_citl']['mean'])} | {_f(mo[d + '_slope']['mean'])} | {_f(b[d + '_slope']['mean'])} |")
    w("")
    # ---------------------------------------------------------------- bootstrap
    bs = dev["bootstrap"]
    w(f"## 3. Full-pipeline bootstrap, 2005–2022 ({bs['n_boot']} resamples, seed {bs['seed_bootstrap']})")
    w("")
    w("Apparent, optimism and optimism-corrected (Harrell) measures. Slope (ordinal): ordinal calibration slope.")
    w("")
    w("| Stage | Model | RPS | Log-loss | C (utility) | AUROC 0–2 | AUROC death | Slope (ordinal) | Slope 0–2 | Slope death |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for st in STAGES:
        for lab in (ORD, LGB):
            r = bs["stages"][st].get(lab)
            if not r:
                continue
            m = r["measures"]
            c = lambda k, d=3: _f(m[k]["corrected"], d) if k in m else "–"  # noqa: E731
            w(f"| {STAGE_EN[st].capitalize()} | {lab} | {c('rps', 4)} | {c('ologloss')} | {c('genc_uw')} | {c('mrs02_auroc')} | "
              f"{c('death_auroc')} | {c('ord_slope')} | {c('mrs02_slope')} | {c('death_slope')} |")
    w("")
    w("Stability across the resamples: median (90th percentile) width of each patient's 95 % range.")
    w("")
    w("| Stage | Model | P(mRS 0–2) | P(death) | Expected utility | Knots 3/4/5 | Median log10 C |")
    w("|---|---|---|---|---|---|---|")
    for st in STAGES:
        os_ = bs["stages"][st]["ordinal_settings"]
        for lab in (ORD, LGB):
            r = bs["stages"][st].get(lab)
            if not r:
                continue
            sb = r["stability"]
            ks = " / ".join(f"{100 * os_['knots_share'][k]:.0f} %" for k in ("3", "4", "5")) if lab == ORD else "–"
            lc = _f(os_['log10_C_median'], 2) if lab == ORD else "–"
            w(f"| {STAGE_EN[st].capitalize()} | {lab} | {_f(sb['p02']['width_median'])} ({_f(sb['p02']['width_p90'])}) | "
              f"{_f(sb['pdeath']['width_median'])} ({_f(sb['pdeath']['width_p90'])}) | {_f(sb['uw']['width_median'])} "
              f"({_f(sb['uw']['width_p90'])}) | {ks} | {lc} |")
    w("")
    # ---------------------------------------------------------------- O4
    w("## 4. Proportional odds (rule O4, descriptive)")
    w("")
    w("A penalised binary model of mRS > k at each cut-point, with the development model's layout and penalty. Slope and "
      "correlation of its linear predictor on the shared one: both 1 under proportional odds, apart from the penalty.")
    w("")
    w("| Stage | " + " | ".join(f"> {k}" for k in range(6)) + " |")
    w("|---|" + "---|" * 6)
    for st in STAGES:
        w(f"| {STAGE_EN[st].capitalize()} | " + " | ".join(f"{r['lp_slope']:.2f} ({r['lp_correlation']:.2f})" for r in dev["brant"][st]) + " |")
    w("")
    w("Calibration of the implied dichotomies (nested cross-validation, mean slope over folds): " + "; ".join(
        f"{STAGE_EN[st]} " + ", ".join(f"{lab} {_f(dev['nested'][st]['models'][ORD][k + '_slope']['mean'], 2)}"
                                     for k, lab in (("mrs01", "0–1"), ("mrs02", "0–2"), ("mrs03", "0–3"), ("death", "death")))
        for st in STAGES if st in dev["nested"]) + ".")
    w("")
    # ---------------------------------------------------------------- sample size
    w("## 5. Sample size (Riley et al., descriptive)")
    w("")
    w("Required patients at each cut-point for the development model's number of coefficients, with the conservative "
      "anticipated R² (15 % of the maximum); in brackets, with the R² implied by v1.1's nested AUROC (mRS 0–2 and death).")
    w("")
    w("| Stage | Patients | Coefficients | " + " | ".join(f"> {k}" for k in range(6)) + " |")
    w("|---|---|---|" + "---|" * 6)
    for st in STAGES:
        rr = dev["sample_size"][st]
        cells = []
        for r in rr:
            c = f"{r['conservative']['n_required']}"
            if "from_v11_auroc" in r:
                c += f" ({r['from_v11_auroc']['n_required']})"
            cells.append(c)
        w(f"| {STAGE_EN[st].capitalize()} | {rr[0]['n']} | {rr[0]['parameters']} | " + " | ".join(cells) + " |")
    w("")
    w("Events per coefficient at the scarcest side of each cut-point: " + "; ".join(
        f"{STAGE_EN[st]} " + " / ".join(f"{r['events_per_parameter']:.1f}" for r in dev['sample_size'][st]) for st in STAGES) + " (cut-points > 0 to > 5).")
    w("")
    # ---------------------------------------------------------------- development models
    w("## 6. Development-year models (2005–2022)")
    w("")
    w("| Stage | Patients | Knots | C | Coefficients | Thresholds θ₀…θ₅ |")
    w("|---|---|---|---|---|---|")
    for st in STAGES:
        dm = dev["development_models"][st]
        w(f"| {STAGE_EN[st].capitalize()} | {dm['n']} | {dm['settings']['knots']} | {dm['settings']['C']:.4g} | {dm['n_coef']} | "
          + ", ".join(_f(v, 2) for v in dm["theta"]) + " |")
    w("")
    if tmp_path.exists():
        L += temporal_md(json.loads(tmp_path.read_text()))
    if dep_path.exists():
        L += deploy_md(json.loads(dep_path.read_text()))
        w = L.append
        w("## 9. Reporting (TRIPOD+AI)")
        w("")
        w("| TRIPOD+AI topic | Where |")
        w("|---|---|")
        for a, b in (("Source of data, participants, outcome", "plan, section 3; [model card](../model_card_ordinal.md), section 2; section 1 here"),
                     ("Predictors and their handling, missing data", "plan, section 4; [diagnostics](../model_diagnostics_ordinal.md)"),
                     ("Sample size", "section 5"),
                     ("Model type, tuning, internal validation", "plan, sections 4–7, and amendment 1; sections 2–4"),
                     ("Comparators and the comparison-only learner", "sections 2, 3 and 7"),
                     ("Recalibration and updating", "rule O3 (decisions) and section 8"),
                     ("Performance with intervals, subgroups", "sections 2, 3 and 7"),
                     ("Full model specification", "artifacts/public/ordinal_bundle.json and its fingerprint"),
                     ("Limitations and intended use", "[model card](../model_card_ordinal.md), sections 1 and 6")):
            w(f"| {a} | {b} |")
        w("")
    text = "\n".join(L) + "\n"
    path.write_text(text)
    return text


def temporal_md(t: dict) -> list:
    L = []
    w = L.append
    w("## 7. Temporal and sensitivity tests (section 8, reported, not deciding)")
    w("")
    w("The development-year models tested on later patients. **Not independent:** the specification is that of v1.1, which "
      "was designed with the 2023–2025 results known. 95 % intervals from "
      f"{t['n_boot']:,} bootstrap resamples of the test patients; differences against the ordinal regression are paired "
      "and unadjusted. Rule O3's factor is used in both splits.")
    w("")
    for split, sp in t["splits"].items():
        w(f"### {'Primary' if split == 'primary' else 'Sensitivity'} split: development before {sp['test_from']}, test {sp['test_from']}–2025")
        w("")
        for st in STAGES:
            r = sp["stages"][st]
            w(f"**{STAGE_EN[st].capitalize()}** ({r['n_dev']} development, {r['n_test']} test patients; test grades 0–6: "
              + ", ".join(("<10" if 0 < v < 10 else str(v)) for v in r["grades_test"]) + ")")
            w("")
            w("| Model | RPS | Log-loss | C (utility) | AUROC 0–2 | AUROC death | CITL 0–2 | Slope 0–2 | Slope (ordinal) | Utility |")
            w("|---|---|---|---|---|---|---|---|---|---|")
            for lab, m in r["models"].items():
                ci = r["ci"][lab]
                c = lambda k, d=3: (f"{_f(_flat_any(lab, m)[k], d)} ({_f(ci[k][0], d)}–{_f(ci[k][1], d)})" if k in ci else "–")  # noqa: E731
                if lab == BIN:
                    w(f"| {lab} | – | – | – | {c('mrs02_auroc')} | {c('death_auroc')} | {c('mrs02_citl')} | {c('mrs02_slope')} | – | – |")
                    continue
                fm = _flat_any(lab, m)
                w(f"| {lab} | {c('rps', 4)} | {c('ologloss')} | {c('genc_uw')} | {c('mrs02_auroc')} | {c('death_auroc')} | "
                  f"{c('mrs02_citl')} | {c('mrs02_slope')} | {c('ord_slope') if 'ord_slope' in fm else '–'} | {_sg(fm['uw_diff'])} |")
            w("")
            w("Differences against the ordinal regression (other model minus ordinal; RPS and log-loss: negative favours the other model):")
            w("")
            w("| Comparison | RPS | C (utility) | AUROC 0–2 | AUROC death |")
            w("|---|---|---|---|---|")
            for key, dd in r["differences"].items():
                c = lambda k, d=3: (f"{_sg(dd[k]['estimate'], d)} ({_sg(dd[k]['ci'][0], d)} to {_sg(dd[k]['ci'][1], d)})" if k in dd else "–")  # noqa: E731
                w(f"| {key.replace(' - ', ' − ')} | {c('rps', 4)} | {c('genc_uw')} | {c('mrs02_auroc')} | {c('death_auroc')} |")
            w("")
            shown = {g: v for g, v in r["subgroups"].items() if v.get("shown")}
            hidden = [g for g, v in r["subgroups"].items() if not v.get("shown")]
            if shown:
                w("Subgroups (point estimates; RPS / C (utility) of the ordinal regression and the layer): " + "; ".join(
                    f"{g} (n {v['n']}): {_f(v['models'][ORD]['rps'], 4)} / {_f(v['models'][ORD]['genc_uw'])} and "
                    f"{_f(v['models'][LAYER]['rps'], 4)} / {_f(v['models'][LAYER]['genc_uw'])}" for g, v in shown.items()) + ".")
            if hidden:
                w(f"Not shown (fewer than 10 patients in mRS 0–2, 3–5 or 6): {', '.join(hidden)}.")
            w("")
    return L


def deploy_md(d: dict) -> list:
    L = []
    w = L.append
    w("## 8. Deployed model (section 11)")
    w("")
    w("Refitted on every patient of 2005–2025 with the development-year knots and penalty and rule O3's factor; the "
      "recent-era threshold shift estimated on 2023–2025 from out-of-fold estimates; the slope changed only under the "
      "update policy (slope outside 0.8–1.2 with a 95 % interval that excludes 1).")
    w("")
    w("| Stage | Patients | Coefficients | Knots | C | O3 factor | Recent shift | Recent slope (95 % interval) | Slope updated |")
    w("|---|---|---|---|---|---|---|---|---|")
    for st in STAGES:
        r = d[st]
        pol = r["update_policy"]
        w(f"| {STAGE_EN[st].capitalize()} | {r['n']} | {r['n_coef']} | {r['settings']['knots']} | {r['settings']['C']:.4g} | {r['slope_factor']:.3f} | "
          f"{_sg(r['threshold_shift_recent'])} | {_f(pol['slope'], 2)} ({_f(pol['lo'], 2)}–{_f(pol['hi'], 2)}) | {'yes' if pol['triggered'] else 'no'} |")
    w("")
    w("Out-of-fold estimates in 2023–2025 before and after the threshold shift:")
    w("")
    w("| Stage | Recent patients | RPS before | RPS after | Utility before | Utility after | C (utility) |")
    w("|---|---|---|---|---|---|---|")
    for st in STAGES:
        r = d[st]
        a, b = r["oof_recent_uncalibrated"], r["oof_recent_recalibrated"]
        w(f"| {STAGE_EN[st].capitalize()} | {r['n_recent']} | {_f(a['rps'], 4)} | {_f(b['rps'], 4)} | {_sg(a['uw_diff'])} | {_sg(b['uw_diff'])} | {_f(b['genc_uw'])} |")
    w("")
    bpath = paths.PUBLIC / "ordinal_bundle.json"
    if bpath.exists():
        b = json.loads(bpath.read_text())
        w(f"**MTOP ordinal {b['version']}**, fingerprint `{b['fingerprint']}`; calculator distribution: "
          f"{'the v1.1 bands split by the ordinal model' if b['calculator_distribution'] == 'ordinal shares' else 'the v1.1 layer'}. "
          f"Built on the binary models of version {b['binary_version']} (`{b['binary_fingerprint'][:16]}…`), which are unchanged.")
        w("")
    ppath = PRIV / "parity.json"
    if ppath.exists():
        pr = json.loads(ppath.read_text())
        worst = max(r.get("browser_vs_model", 0.0) for s_ in pr["stages"].values() for r in s_.values())
        worst_e = max(r["export_vs_model"] for s_ in pr["stages"].values() for r in s_.values())
        n_reg = sum(s_["registry"]["n"] for s_ in pr["stages"].values())
        w(f"Browser parity (web/model.js against the Python model, every patient of each stage, {n_reg} stage-patient rows, and "
          f"the synthetic demonstration cohort): largest difference in the linear predictor, any grade probability or any "
          f"explanation term {worst:.1e} (exported numbers {worst_e:.1e}); criterion 10⁻⁷: **{'passed' if pr['passed'] else 'FAILED'}**.")
        w("")
    return L


def public_json() -> dict:
    """artifacts/public/ordinal.json: aggregates for the app and the results file (no patient rows)."""
    from .export import clean_json
    dev = json.loads((PRIV / "development.json").read_text())
    out = {"plan": "docs/registrations/ordinal_plan.md", "results": "docs/registrations/ordinal_results.md",
           "pipeline_check": {k: dev["pipeline_check"][k] for k in ("export_sha256", "export_identical", "statsmodels_max_abs_diff", "passed")},
           "decisions": dev["decisions"], "development_models": dev["development_models"],
           "nested": {st: {"models": {lab: {k: v["mean"] for k, v in m.items()} for lab, m in r["models"].items()},
                           "binary": {k: v["mean"] for k, v in r["binary"].items()}} for st, r in dev["nested"].items()},
           "bootstrap": {st: {lab: {"corrected": {k: v["corrected"] for k, v in r[lab]["measures"].items()}, "stability": r[lab]["stability"]}
                              for lab in (ORD, LGB) if lab in r} for st, r in dev["bootstrap"]["stages"].items()},
           "brant": dev["brant"]}
    tp = PRIV / "temporal.json"
    if tp.exists():
        t = json.loads(tp.read_text())
        out["temporal"] = {}
        for split, sp in t["splits"].items():
            out["temporal"][split] = {}
            for st, r in sp["stages"].items():
                g = r["grades_test"]
                out["temporal"][split][st] = {
                    "n_test": r["n_test"], "test_from": sp["test_from"],
                    "models": {lab: {**{k: v for k, v in _flat_any(lab, m).items()}, "ci": r["ci"][lab]} for lab, m in r["models"].items()},
                    "observed": [None if 0 < v < 10 else v / max(1, r["n_test"]) for v in g],
                    "predicted": {lab: m["grade_pred"] for lab, m in r["models"].items() if lab != BIN}}
    dp = PRIV / "deployed.json"
    if dp.exists():
        d = json.loads(dp.read_text())
        out["deployed"] = {st: {k: r[k] for k in ("settings", "slope_factor", "n_coef", "n", "n_recent", "threshold_shift_recent", "update_policy",
                                                    "mean_abs_contribution")} for st, r in d.items()}
    out = clean_json(out)
    (paths.PUBLIC / "ordinal.json").write_text(json.dumps(out, indent=1))
    return out


def _labels() -> dict:
    from .config import features
    return {f.key: f.spec.get("label", {}).get("en", f.key) for f in features()}


def diagnostics_md(path=None) -> str:
    """docs/model_diagnostics_ordinal.md: settings, structure and explanations of the deployed model."""
    path = path or (paths.ROOT / "docs" / "model_diagnostics_ordinal.md")
    dep = json.loads((PRIV / "deployed.json").read_text())
    dev = json.loads((PRIV / "development.json").read_text())
    b = json.loads((paths.PUBLIC / "ordinal_bundle.json").read_text())
    lab = _labels()
    L = []
    w = L.append
    w("# MTOP ordinal: model diagnostics")
    w("")
    w(f"Generated by `python -m mtop.ordinal_run diagnostics` on {b['generated']} for MTOP ordinal {b['version']} (fingerprint "
      f"`{b['fingerprint'][:16]}…`). Aggregates only. Evaluation: [registrations/ordinal_results.md](registrations/ordinal_results.md); "
      "model card: [model_card_ordinal.md](model_card_ordinal.md).")
    w("")
    w("## Settings and size")
    w("")
    w("Knots (3, 4 or 5) and the penalty C (50 values, 0.0001 to 100) chosen on 2005–2022 by 5-fold cross-validation repeated "
      "5 times (mean ordinal log-loss, folds stratified by grade); the deployed model is refitted on all years with these "
      "settings and rule O3's factor.")
    w("")
    w("| Stage | Knots | C | CV ordinal log-loss | O3 factor | Coefficients | Missing-value terms | Thresholds θ₀…θ₅ (deployed) | Recent shift |")
    w("|---|---|---|---|---|---|---|---|---|")
    for st in STAGES:
        d, dm = dep[st], dev["development_models"][st]
        w(f"| {STAGE_EN[st].capitalize()} | {d['settings']['knots']} | {d['settings']['C']:.4g} | {_f(dm['settings'].get('cv_logloss'), 4)} | "
          f"{d['slope_factor']:.3f} | {d['n_coef']} | {len(d['missing_terms'])} | " + ", ".join(_f(v, 2) for v in d["theta"])
          + f" | {_sg(d['threshold_shift_recent'])} |")
    w("")
    w("## Missing-value terms and joined categories")
    w("")
    cat = lambda c: f"{lab.get(c.split('__')[0], c)}: {c.split('__')[1].replace('_', ' ')}" if "__" in c else lab.get(c, c)  # noqa: E731
    for st in STAGES:
        d = dep[st]
        w(f"**{STAGE_EN[st].capitalize()}.** Joined: {'; '.join(', '.join(cat(c) for c in v[1:]) + ' → ' + cat(k) for k, v in d['merged'].items()) or 'none'}. "
          f"Missing-value terms ({len(d['missing_terms'])}): {', '.join(lab.get(k, k) for k in d['missing_terms'])}.")
        w("")
    w("## What drives the estimates")
    w("")
    w("Mean absolute contribution to the linear predictor (log-odds of a worse grade; the same at every cut-point) over all "
      "patients of the stage, the ten largest:")
    w("")
    for st in STAGES:
        imp = list(dep[st]["mean_abs_contribution"].items())[:10]
        w(f"- **{STAGE_EN[st].capitalize()}:** " + ", ".join(f"{lab.get(k, k)} {v:.2f}" for k, v in imp) + ".")
    w("")
    w("## Stability of the settings across 500 bootstrap resamples")
    w("")
    w("| Stage | Knots 3 / 4 / 5 | Median log10 C (IQR) | Penalty at the grid end | Median coefficients |")
    w("|---|---|---|---|---|")
    for st in STAGES:
        o = dev["bootstrap"]["stages"][st]["ordinal_settings"]
        w(f"| {STAGE_EN[st].capitalize()} | " + " / ".join(f"{100 * o['knots_share'][k]:.0f} %" for k in ("3", "4", "5"))
          + f" | {o['log10_C_median']:.2f} ({o['log10_C_iqr'][0]:.2f} to {o['log10_C_iqr'][1]:.2f}) | {100 * o['share_C_at_grid_end']:.0f} % | {o['n_coef_median']:.0f} |")
    w("")
    txt = "\n".join(L) + "\n"
    path.write_text(txt, encoding="utf-8")
    return txt


def model_card_md(path=None) -> str:
    """docs/model_card_ordinal.md, from the outputs (structured after TRIPOD+AI)."""
    path = path or (paths.ROOT / "docs" / "model_card_ordinal.md")
    dep = json.loads((PRIV / "deployed.json").read_text())
    dev = json.loads((PRIV / "development.json").read_text())
    t = json.loads((PRIV / "temporal.json").read_text())
    b = json.loads((paths.PUBLIC / "ordinal_bundle.json").read_text())
    dec = dev["decisions"]
    L = []
    w = L.append
    w("# MTOP ordinal model card")
    w("")
    w(f"**MTOP ordinal {b['version']}** (fingerprint `{b['fingerprint']}`): a proportional-odds model of the whole 90-day "
      "modified Rankin Scale (mRS), developed under a plan committed before any of its code "
      "([plan](registrations/ordinal_plan.md), [results](registrations/ordinal_results.md), "
      "[diagnostics](model_diagnostics_ordinal.md), [changelog](../CHANGELOG.md)). It is a separate model next to the "
      f"binary models of MTOP {b['binary_version']}, which it does not change. Structured after TRIPOD+AI (Collins et al., BMJ 2024).")
    w("")
    w("## 1. Purpose and intended use")
    w("")
    w("- **Predicts:** the probability of each mRS grade 0–6 at 90 days after endovascular treatment for acute ischaemic "
      "stroke, at four moments (before the procedure, at its end, at 24 hours and on day 7), each from what is known then.")
    w("- **Role in the calculator:** " + ("it splits the two binary models' bands (independent, mRS 0–2; dependent, 3–5) into "
      "single grades (rule O1). The headline estimates of independence and death stay those of the binary models."
      if dec["O1"]["replace_layer"] else "none; the calculator keeps the v1.1 layer (rule O1). The model is shown on the Validation tab."))
    w("- **External validation:** " + ("frozen and sent as a secondary model (rule O2): v1.0.0 stays the primary model, v1.1 and "
      "this model are secondary." if dec["O2"]["send_to_external_validation"] else "not sent (rule O2)."))
    w("- **Who for:** research, teaching and trial planning (for example the ordinal \"shift\" outcomes of stroke trials). "
      "**Not for** deciding whether or how to treat an individual patient. Not a medical device under EU MDR 2017/745.")
    w("")
    w("## 2. Data and outcome")
    w("")
    w("The registry export, cleaning, patients and stage cohorts of v1.1 (SHA-256 `1a19f809…`; see the "
      "[main model card](model_card.md), sections 2–4). Outcome: the 90-day mRS as recorded, seven grades; no grade had "
      "fewer than 20 development patients, so none was merged. Development 2005–2022; test 2023–2025 (sensitivity split "
      "2005–2020 against 2021–2025); the deployed model is refitted on 2005–2025.")
    w("")
    w("## 3. Model")
    w("")
    w("Cumulative logit: logit P(mRS ≤ k) = θₖ − x′β, one coefficient vector for all six cut-points, thresholds ordered. "
      "Inputs: the v1.1 reference specification at each stage (restricted cubic splines for age, NIHSS, onset-to-puncture "
      "time, glucose and procedure duration; one missing-value indicator per input lacked by at least 20 patients). L2 "
      "penalty on β, fitted by Newton–Raphson; knots and penalty tuned by repeated cross-validation on ordinal log-loss. "
      "Explanations are exact: each input's coefficient × standardised value, the same at every cut-point.")
    w("")
    w("| Stage | Patients | Coefficients | Knots | O3 slope factor | Recent-era threshold shift |")
    w("|---|---|---|---|---|---|")
    for st in STAGES:
        d = dep[st]
        w(f"| {STAGE_EN[st].capitalize()} | {d['n']} | {d['n_coef']} | {d['settings']['knots']} | {d['slope_factor']:.3f} | {_sg(d['threshold_shift_recent'])} |")
    w("")
    w("## 4. Performance in the development years (2005–2022)")
    w("")
    w("Nested cross-validation (15 outer folds) and the optimism-corrected bootstrap (500 resamples). RPS: ranked probability "
      "score (lower is better); C (utility): generalised concordance on the expected utility-weighted mRS.")
    w("")
    w("| Stage | RPS, nested | RPS, corrected | C (utility), nested | AUROC mRS 0–2 (binary model) | AUROC death (binary model) | Ordinal calibration slope, corrected |")
    w("|---|---|---|---|---|---|---|")
    for st in STAGES:
        n, bs = dev["nested"][st], dev["bootstrap"]["stages"][st][ORD]["measures"]
        w(f"| {STAGE_EN[st].capitalize()} | {_f(n['models'][ORD]['rps']['mean'], 4)} | {_f(bs['rps']['corrected'], 4)} | "
          f"{_f(n['models'][ORD]['genc_uw']['mean'])} | {_f(n['models'][ORD]['mrs02_auroc']['mean'])} ({_f(n['binary']['mrs02_auroc']['mean'])}) | "
          f"{_f(n['models'][ORD]['death_auroc']['mean'])} ({_f(n['binary']['death_auroc']['mean'])}) | {_f(bs['ord_slope']['corrected'])} |")
    w("")
    w("## 5. Temporal test (2023–2025; not independent)")
    w("")
    w("| Stage | Test patients | RPS (95 % CI) | C (utility) | Mean utility predicted − observed |")
    w("|---|---|---|---|---|")
    for st in STAGES:
        r = t["splits"]["primary"]["stages"][st]
        m, ci = flat(r["models"][ORD]), r["ci"][ORD]
        w(f"| {STAGE_EN[st].capitalize()} | {r['n_test']} | {_f(m['rps'], 4)} ({_f(ci['rps'][0], 4)}–{_f(ci['rps'][1], 4)}) | "
          f"{_f(m['genc_uw'])} | {_sg(m['uw_diff'])} |")
    w("")
    w("The inputs are those of v1.1, which was designed with the 2023–2025 results known, so this test is not independent.")
    w("")
    w("## 6. Limitations")
    w("")
    ss = dev["sample_size"]
    w("- One centre; no external validation yet (planned on the MR CLEAN Registry, section 12 of the plan).")
    w("- Proportional odds is an assumption: separate models at each cut-point differ from the shared one (results, section "
      "4); each implied dichotomy's calibration is reported.")
    w("- Sample size: with " + ", ".join(f"{ss[st][0]['parameters']}" for st in STAGES) + " coefficients by stage, Riley et "
      "al.'s conservative criteria call for more patients than the registry has at the extreme cut-points (results, "
      "section 5); the penalty and the optimism-corrected checks are what guard against overfitting.")
    w("- The 90-day mRS is missing in about one in five patients of 2023–2025; complete-case analysis.")
    w("- Death on day 7 carries the \"limited evidence\" label of v1.1 (fewer than 10 deaths in that stage's temporal test).")
    w("")
    w("## 7. Privacy and maintenance")
    w("")
    w("The bundle holds coefficients, thresholds, knots, medians and scaling only, no patient-level data; counts from 1 to 9 are "
      "shown as fewer than 10. The browser reimplementation (web/model.js) matches the Python model within 10⁻⁷ on every patient "
      "and on the synthetic demonstration cohort. A new version gets a new fingerprint, a changelog entry and a Zenodo "
      "version under the MTOP all-versions DOI (10.5281/zenodo.24396392).")
    w("")
    txt = "\n".join(L) + "\n"
    path.write_text(txt, encoding="utf-8")
    return txt


def external_md(path=None) -> str:
    """docs/registrations/external_validation_models.md (plan, section 12): the frozen models and their
    roles, recorded before any MR CLEAN Registry data are received."""
    path = path or (paths.ROOT / "docs" / "registrations" / "external_validation_models.md")
    b = json.loads((paths.PUBLIC / "ordinal_bundle.json").read_text())
    dec = json.loads((PRIV / "decisions.json").read_text())
    L = []
    w = L.append
    w("# MTOP models frozen for external validation")
    w("")
    w(f"Recorded {b['generated']}, before any MR CLEAN Registry data have been received. Every external validation names the "
      "fingerprint of each model it tests; a results file is accepted only for a fingerprint listed here.")
    w("")
    w("| Model | Version | Fingerprint (SHA-256 of the models block) | Role | Archive |")
    w("|---|---|---|---|---|")
    w("| MTOP (binary models: mRS 0–2, death) | 1.0.0 | `77bb37a365930cdf3e5551d000f8ae40fdfa41f50a44c0593bdc7fea80ba4425` | **Primary** | "
      "[10.5281/zenodo.24396393](https://doi.org/10.5281/zenodo.24396393) |")
    w("| MTOP (binary models) | 1.1.0 | `841d810c28097e00d32ea2e0de6ea6f6b432f235a5a8e228ed7021b955556296` | Secondary | "
      "[10.5281/zenodo.24420424](https://doi.org/10.5281/zenodo.24420424) |")
    if dec["O2"]["send_to_external_validation"]:
        w(f"| MTOP ordinal (all seven mRS grades) | {b['version']} | `{b['fingerprint']}` | Secondary | "
          "Zenodo version under 10.5281/zenodo.24396392 (at release) |")
    w("")
    w("Plans: [v1_1_plan.md](v1_1_plan.md), [ordinal_plan.md](ordinal_plan.md). The external analysis plan (the CONTRAST data "
      "request) sets the analyses and margins, including the comparison of MTOP ordinal with MR PREDICTS (its published ordinal "
      "formula with the collateral score): generalised concordance, ranked probability score and the calibration of the "
      "distribution. MRS CLEAN (synthetic data) is a pipeline check only, run with the MR CLEAN team's agreement.")
    w("")
    txt = "\n".join(L) + "\n"
    path.write_text(txt, encoding="utf-8")
    return txt


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m mtop.ordinal_run")
    ap.add_argument("step", choices=["check", "timing", "nested", "bootstrap", "develop", "report", "temporal", "deploy", "bundle", "public", "docs"])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--max-seconds", type=float, default=None)
    a = ap.parse_args(argv)
    if a.step == "check":
        print(json.dumps(check(), indent=1))
    elif a.step == "timing":
        t0 = time.time()
        rec, _ = _boot_one(0, boot_rows()[0])
        print(json.dumps(rec, indent=1), f"{time.time() - t0:.0f} s")
    elif a.step == "nested":
        nested(n_jobs=a.jobs)
    elif a.step == "bootstrap":
        bootstrap(n_jobs=a.jobs, max_seconds=a.max_seconds)
    elif a.step == "develop":
        develop()
    elif a.step == "report":
        report_md()
    elif a.step == "temporal":
        temporal()
    elif a.step == "deploy":
        deploy()
    elif a.step == "bundle":
        b = build_bundle()
        (paths.PUBLIC / "ordinal_bundle.json").write_text(json.dumps(b, indent=1))
        print(json.dumps(parity(b)["passed"]), b["fingerprint"])
    elif a.step == "public":
        public_json()
    elif a.step == "docs":
        report_md()
        diagnostics_md()
        model_card_md()
        external_md()


if __name__ == "__main__":
    main()
