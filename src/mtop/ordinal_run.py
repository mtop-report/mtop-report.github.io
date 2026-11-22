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


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m mtop.ordinal_run")
    ap.add_argument("step", choices=["check", "timing", "nested", "bootstrap"])
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


if __name__ == "__main__":
    main()
