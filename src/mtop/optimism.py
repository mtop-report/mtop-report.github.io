"""Internal validation of the reference model: bootstrap optimism and stability (Harrell's method).

Plan: docs/registrations/bootstrap_optimism_plan.md (committed before the analysis was run).
MTOP v1.0.0 is not changed: this module only refits copies of the reference pipeline in memory.

    python -m mtop.optimism [--stage admission] [--targets good_outcome death90] [--n-boot 500]

In each bootstrap sample the whole pipeline of `splines.fit_spline_logreg` is repeated: knots at
Harrell's percentiles of the sample (3, 4 and 5), median imputation with missing-value indicators and
standardisation fitted on the sample, and the penalty tuned over the same 50 values by 5-fold
cross-validated log-loss. The inner folds keep the copies of a patient together (stratified group
folds); with groups=None the published folds are used, which reproduces the published model.

Output (aggregates; git-ignored): reports/optimism/<stage>_<target>.json, with per-resample records
in <stage>_<target>_resamples.jsonl and each bootstrap model's estimates for the development patients
in artifacts/private/optimism/<stage>_<target>_preds.npy (row-level, never published).
"""
from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import warnings

import numpy as np
import pandas as pd

from . import paths

SEED, N_BOOT = 2026, 500
REP = paths.REPORTS / "optimism"
PRIV = paths.PRIVATE / "optimism"
MEASURES = ("auroc", "brier", "logloss", "citl", "slope", "oe")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- the reference pipeline
def grouped_folds(y, groups, seed: int) -> list:
    """5 stratified folds that keep all rows of one patient in the same fold."""
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.asarray(y)
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, np.asarray(groups)))


def fit_reference(X: pd.DataFrame, y, seed: int, groups=None):
    """`splines.fit_spline_logreg` with the inner folds made explicit. groups=None: the published folds
    (tuning.shared_folds); otherwise stratified group folds. Returns (model, settings, n_warnings)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegressionCV

    from .lr import C_GRID, _pipeline, cv_logloss
    from .splines import KNOT_OPTIONS, SplineLogReg, choose_knots, expand
    from .tuning import shared_folds

    y = np.asarray(y, float)
    folds = shared_folds(y, seed) if groups is None else grouped_folds(y, groups, seed)
    accepted = inspect.signature(LogisticRegressionCV).parameters
    best, n_warn = None, 0
    for k in KNOT_OPTIONS:
        knots = choose_knots(X, k=k)
        kw = dict(Cs=C_GRID, cv=folds, scoring="neg_log_loss", max_iter=5000, random_state=seed)
        if "use_legacy_attributes" in accepted:
            kw.update(use_legacy_attributes=False, l1_ratios=(0.0,))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = _pipeline(LogisticRegressionCV(**kw)).fit(expand(X, knots), y)
        n_warn += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        clf = model.steps[2][1]
        ll = cv_logloss(clf)
        if best is None or ll < best[3]:
            best = (k, knots, model, ll, float(np.ravel(clf.C_)[0]))
    k, knots, model, ll, C = best
    out = SplineLogReg(knots, model, list(X.columns))
    settings = {"knots": int(k), "C": C, "cv_logloss": float(ll),
                "C_at_grid_end": bool(np.isclose(C, C_GRID[0]) or np.isclose(C, C_GRID[-1]))}
    return out, settings, n_warn


def measures(y, p) -> dict:
    from sklearn.metrics import log_loss, roc_auc_score

    from .metrics import calibration_intercept_slope, oe_ratio
    y, p = np.asarray(y, float), np.asarray(p, float)
    citl, slope = calibration_intercept_slope(y, p)
    return {"auroc": float(roc_auc_score(y, p)), "brier": float(np.mean((p - y) ** 2)),
            "logloss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
            "citl": citl, "slope": slope, "oe": oe_ratio(y, p)}


# --------------------------------------------------------------------------- one resample
def one_resample(b: int, idx: np.ndarray, X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[dict, np.ndarray]:
    Xb = X.iloc[idx].reset_index(drop=True)
    yb = y[idx]
    m, s, n_warn = fit_reference(Xb, yb, seed, groups=idx)
    p_app = m.predict_proba(Xb)[:, 1]
    p_test = m.predict_proba(X)[:, 1]
    app, test = measures(yb, p_app), measures(y, p_test)
    rec = {"b": b, **s, "convergence_warnings": n_warn, "events": int(yb.sum()), "unique_patients": int(len(np.unique(idx))),
           "apparent": app, "test": test, "optimism": {k: app[k] - test[k] for k in MEASURES}}
    return rec, p_test.astype(np.float32)


# --------------------------------------------------------------------------- data
def development_data(stage: str):
    from .cohort import flow, split_temporal, stage_cohort
    from .features import design_matrix
    base, _ = flow(pd.read_parquet(paths.PROCESSED / "clean.parquet"))
    train = json.loads((paths.PRIVATE / "train_primary.json").read_text())
    test_from = int(train["test_from_year"])
    d = stage_cohort(base, stage)
    dev, test = split_temporal(d, test_from)
    X_all = design_matrix(d, stage)
    return train, test_from, dev, test, X_all


# --------------------------------------------------------------------------- the analysis
def run(stage: str, target: str, n_boot: int = N_BOOT, seed_boot: int = SEED, n_jobs: int = 2, chunk: int = 20) -> dict:
    from joblib import Parallel, delayed

    from .config import settings as cfg

    REP.mkdir(parents=True, exist_ok=True)
    PRIV.mkdir(parents=True, exist_ok=True)
    seed = int(cfg()["seed"])
    train, test_from, dev, test, X_all = development_data(stage)
    X = X_all.loc[dev.index].reset_index(drop=True)
    y = dev[target].to_numpy(float)
    res = {"plan": "docs/registrations/bootstrap_optimism_plan.md", "stage": stage, "target": target,
           "n": int(len(y)), "events": int(y.sum()), "years": f"to {test_from - 1}", "n_boot": n_boot,
           "seed_bootstrap": seed_boot, "seed_folds": seed, "started": now()}

    # 1. the pipeline reproduces the published model
    saved = train["stages"][stage]["targets"][target]["settings"]["Logistic regression (splines)"]
    m0, s0, w0 = fit_reference(X, y, seed)
    stored = pd.read_parquet(paths.PRIVATE / "predictions_primary.parquet")
    ref = stored[(stored["stage"] == stage) & (stored["target"] == target) & (stored["model"] == "Logistic regression (splines)")]
    ref = ref.set_index("patient_id").loc[test["patient_id"].to_numpy(), "p"].to_numpy()
    repro = float(np.max(np.abs(m0.predict_proba(X_all.loc[test.index])[:, 1] - ref)))
    res["reproduction"] = {"saved": {"knots": int(saved["knots"]), "C": float(saved["C"])}, "refit": s0,
                           "test_predictions_max_difference": repro}
    if s0["knots"] != int(saved["knots"]) or abs(s0["C"] - float(saved["C"])) > 5e-6 or repro > 1e-6:
        raise SystemExit(f"the pipeline does not reproduce the published model: {res['reproduction']}")

    # 2. apparent performance, and what the grouped folds choose on the original data
    res["apparent"] = measures(y, m0.predict_proba(X)[:, 1])
    _, s_g, _ = fit_reference(X, y, seed, groups=np.arange(len(y)))
    res["original_with_grouped_folds"] = s_g

    # 3. resamples (all drawn first), in chunks with a checkpoint
    rng = np.random.default_rng(seed_boot)
    IDX = rng.integers(0, len(y), size=(n_boot, len(y)))
    log = REP / f"{stage}_{target}_resamples.jsonl"
    pred_path = PRIV / f"{stage}_{target}_preds.npy"
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            done[r["b"]] = r
    P = np.load(pred_path) if pred_path.exists() else np.full((n_boot, len(y)), np.nan, np.float32)
    todo = [b for b in range(n_boot) if b not in done or np.isnan(P[b]).any()]
    for start in range(0, len(todo), chunk):
        part = todo[start:start + chunk]
        out = Parallel(n_jobs=n_jobs)(delayed(one_resample)(b, IDX[b], X, y, seed) for b in part)
        with log.open("a") as fh:
            for rec, p in out:
                P[rec["b"]] = p
                done[rec["b"]] = rec
                fh.write(json.dumps(rec) + "\n")
        np.save(pred_path, P)
        print(f"{stage}/{target}: {len(done)}/{n_boot} at {now()}", flush=True)

    recs = [done[b] for b in range(n_boot)]
    res["finished"] = now()
    res.update(summarise(res["apparent"], recs, P))
    path = REP / f"{stage}_{target}.json"
    path.write_text(json.dumps(res, indent=1))
    return res


def summarise(apparent: dict, recs: list[dict], P: np.ndarray) -> dict:
    from .lr import C_GRID
    B = len(recs)
    opt = {k: np.array([r["optimism"][k] for r in recs]) for k in MEASURES}
    out = {"optimism": {}}
    for k in MEASURES:
        o = opt[k]
        out["optimism"][k] = {"apparent": apparent[k], "mean_optimism": float(o.mean()),
                              "monte_carlo_se": float(o.std(ddof=1) / np.sqrt(B)),
                              "optimism_p2_5": float(np.percentile(o, 2.5)), "optimism_p97_5": float(np.percentile(o, 97.5)),
                              "corrected": float(apparent[k] - o.mean())}
    knots = np.array([r["knots"] for r in recs])
    logC = np.log10([r["C"] for r in recs])
    width = np.percentile(P, 97.5, axis=0) - np.percentile(P, 2.5, axis=0)
    out["stability"] = {
        "knots_share": {str(k): float(np.mean(knots == k)) for k in (3, 4, 5)},
        "log10_C": {"median": float(np.median(logC)), "q1": float(np.percentile(logC, 25)), "q3": float(np.percentile(logC, 75)),
                    "p2_5": float(np.percentile(logC, 2.5)), "p97_5": float(np.percentile(logC, 97.5))},
        "C_grid": [float(C_GRID[0]), float(C_GRID[-1])],
        "share_C_at_grid_end": float(np.mean([r["C_at_grid_end"] for r in recs])),
        "risk_range_width": {"median": float(np.median(width)), "p90": float(np.percentile(width, 90)),
                             "share_wider_than_10_points": float(np.mean(width > 0.10))},
        "resamples_with_convergence_warnings": int(sum(r["convergence_warnings"] > 0 for r in recs)),
        "mean_unique_patients": float(np.mean([r["unique_patients"] for r in recs])),
    }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", default="admission")
    ap.add_argument("--targets", nargs="+", default=["good_outcome", "death90"])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args(argv)
    for tg in a.targets:
        r = run(a.stage, tg, a.n_boot, n_jobs=a.n_jobs)
        print(json.dumps({k: r[k] for k in ("stage", "target", "reproduction", "optimism", "stability")}, indent=1))


if __name__ == "__main__":
    main()
