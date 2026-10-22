"""Analyses of the 2026 FNO report that the app adds (aggregates only; artifacts/public/analyses.json).

1. Missing 90-day outcome (temporal test 2023-2025, admission stage, mRS 0-2): the reference and
   XGBoost, developed on 2005-2022 with their tuned settings, evaluated under four assumptions about
   the test-year patients without a 90-day mRS: complete case (the published validation); all of them
   with a poor outcome; all with a good outcome; and multiple imputation of the outcome (20
   imputations, logistic model with posterior draws) from the admission data plus what was known
   after the procedure (mTICI, NIHSS at 24 h and on day 7, sICH, oedema, decompression) and the year.
2. Why fewer patients were independent in 2023-2025 than in 2016-2022: crude and adjusted odds
   ratios for the period (main model: patient and stroke characteristics; secondary: + IV
   thrombolysis, transfer and onset-to-puncture time), observed / expected mRS 0-2 by year from the
   reference specification fitted on 2016-2022 (each 2016-2022 year predicted by a model without it),
   the share of the drop the recorded characteristics account for (200 bootstrap samples), and the
   mRS distribution of the two periods.
3. Time to groin puncture: adjusted odds ratio of mRS 0-2 per hour within 0-6 h and after 6 h
   (piecewise linear), the same with the year of treatment, its modification by IV thrombolysis, and
   the standardised chance of mRS 0-2 had every patient been punctured at 1-6 h.
Associations in one registry, not causal effects. Adjusted models: logistic regression with the
covariates of associations.py, multiply imputed (20 imputations by chained equations with the 90-day
mRS in the imputation model) and pooled with Rubin's rules.
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score

from . import learners as Lr
from . import metrics as M
from .associations import BINARY, OCC_COLS, covariates
from .cohort import split_temporal, splits, stage_cohort
from .config import settings
from .features import design_matrix, monotone_vector

PATIENT = ["age10", "female", "prestroke_disability", "htn", "dm", "arrhythmia", "stroke_prior", "anticoag",
           "nihss5", "aspects", "glucose", "sbp10"] + OCC_COLS
PROCESS = ["ivt", "transfer", "otp_h"]


# ------------------------------------------------------------------------------------- helpers
def _fit(X: pd.DataFrame, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.Logit(np.asarray(y, float), sm.add_constant(X, has_constant="add")).fit(disp=0, maxiter=300)
    return res.params.to_numpy(), res.cov_params().to_numpy()


def _imputations(X: pd.DataFrame, aux: np.ndarray, m: int, seed: int):
    """m completed copies of X (chained equations; aux joins the imputation model only)."""
    data = np.column_stack([X.to_numpy(float), aux])
    lo, hi = np.nanmin(data, axis=0), np.nanmax(data, axis=0)
    binary = [c for c in X.columns if c in BINARY or c.startswith("occ_") or c in ("period", "ivt")]
    occ_cols = [c for c in OCC_COLS if c in X.columns]
    miss_occ = X[occ_cols].isna().any(axis=1).to_numpy() if occ_cols else np.zeros(len(X), bool)
    out = []
    for k in range(m):
        imp = IterativeImputer(sample_posterior=True, max_iter=15, random_state=seed + k, min_value=lo, max_value=hi)
        Z = pd.DataFrame(imp.fit_transform(data.copy())[:, :-1], columns=X.columns, index=X.index)
        for c in binary:
            Z[c] = (Z[c] >= 0.5).astype(float)
        if miss_occ.any():
            occ = Z[occ_cols].to_numpy().copy()
            part = occ[miss_occ]
            top, keep = part.argmax(axis=1), part.max(axis=1) >= 0.5
            filled = np.zeros_like(part)
            filled[np.arange(len(top))[keep], top[keep]] = 1.0
            occ[miss_occ] = filled
            Z[occ_cols] = occ
        if "aspects" in Z:
            Z["aspects"] = Z["aspects"].round()
        out.append(Z)
    return out


def _pooled(Zs: list[pd.DataFrame], y, extra=None) -> tuple[pd.Series, pd.DataFrame, float]:
    """Rubin's rules over the imputations: estimates, total covariance, degrees of freedom (smallest)."""
    est, cov = [], []
    for Z in Zs:
        D = Z if extra is None else extra(Z)
        b, V = _fit(D, y)
        est.append(b)
        cov.append(V)
        cols = ["const"] + list(D.columns)
    E, W = np.array(est), np.mean(cov, axis=0)
    B = np.cov(E.T, ddof=1)
    m = len(Zs)
    T = W + (1 + 1 / m) * B
    r = (1 + 1 / m) * np.diag(B) / np.maximum(np.diag(W), 1e-12)
    df = float(np.min((m - 1) * (1 + 1 / np.maximum(r, 1e-12)) ** 2))
    return pd.Series(E.mean(0), index=cols), pd.DataFrame(T, index=cols, columns=cols), df


def _or(b: pd.Series, T: pd.DataFrame, df: float, term: str, contrast: dict | None = None) -> dict:
    """OR with 95% CI and p for one term, or for a linear combination {term: weight}."""
    w = pd.Series(contrast or {term: 1.0}).reindex(b.index, fill_value=0.0)
    est, se = float(w @ b), float(np.sqrt(w @ T @ w))
    q = stats.t.ppf(0.975, df)
    p = float(2 * stats.t.sf(abs(est / se), df))
    return {"or": round(float(np.exp(est)), 3), "lo": round(float(np.exp(est - q * se)), 3),
            "hi": round(float(np.exp(est + q * se)), 3), "p": p}


def _cohort_all(clean: pd.DataFrame) -> pd.DataFrame:
    """The analysis cohort before the 'known 90-day mRS' step (cohort.flow without it)."""
    s = settings()["cohort"]
    d = clean[~(clean["age"] < s["min_age"])]
    if s.get("exclude_diagnostic_only", False):
        d = d[d["diagnostic_only"] != 1]
    if s.get("exclude_impossible_times", False) and "time_conflict" in d:
        d = d[d["time_conflict"] != 1]
    return d.copy()


# ---------------------------------------------------------------------------- 1. missing outcome
def missing_outcome(clean: pd.DataFrame, base: pd.DataFrame, primary: dict, m: int = 20, seed: int = 0) -> dict:
    test_from = splits()["primary"]
    stage, target = "admission", "good_outcome"
    allc = stage_cohort(_cohort_all(clean), stage)
    d = stage_cohort(base, stage)
    dev, _ = split_temporal(d, test_from)
    X_dev = design_matrix(d, stage).loc[dev.index]
    tst = allc[allc["year"] >= test_from]
    X_tst = design_matrix(allc, stage).loc[tst.index]
    if list(X_tst.columns) != list(X_dev.columns):
        raise ValueError("design columns differ")
    st = primary["stages"][stage]["targets"][target]["settings"]
    mono = monotone_vector(stage, target)
    names = [Lr.REFERENCE, "XGBoost"]
    preds = {}
    for nm in names:
        f = Lr.refit(nm, st[nm], X_dev, dev[target].to_numpy(float), seed, mono)
        preds[nm] = f.proba(X_tst)
    known = tst["mrs90"].notna().to_numpy()
    y_obs = tst["good_outcome"].to_numpy(float)
    rows = []

    def row(label, y, mask):
        r = {"scenario": label, "n": int(mask.sum()), "good": float(np.mean(y[mask]))}
        for nm in names:
            r[f"auroc::{nm}"] = float(roc_auc_score(y[mask], preds[nm][mask]))
            r[f"citl::{nm}"] = float(M.calibration_intercept_slope(y[mask], preds[nm][mask])[0])
        return r

    rows.append(row("complete_case", y_obs, known))
    allm = np.ones(len(tst), bool)
    rows.append(row("missing_poor", np.where(known, y_obs, 0.0), allm))
    rows.append(row("missing_good", np.where(known, y_obs, 1.0), allm))
    # multiple imputation of the outcome: logistic model on every patient with a known outcome
    post = ["tici", "nihss_24h", "nihss_7d", "sich", "edema", "decompression"]
    pool = allc.copy()
    Xi = design_matrix(pool, stage)
    for c in post:
        Xi[c] = pool[c].astype(float)
    Xi["year"] = pool["year"].astype(float)
    med = Xi.median()
    ind = Xi.isna().astype(float).add_suffix("__missing")
    ind = ind.loc[:, ind[pool["mrs90"].notna()].std() > 0]
    Zi = pd.concat([Xi.fillna(med), ind], axis=1)
    Zi = (Zi - Zi.mean()) / Zi.std().replace(0, 1)
    Zi = Zi.loc[:, Zi.std() > 0]
    obs = pool["mrs90"].notna().to_numpy()
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=0.1, max_iter=5000).fit(Zi[obs], pool.loc[obs, "good_outcome"].to_numpy(float))
    # approximate posterior of the coefficients (penalised fit + observed information), for proper imputations
    Zo = np.column_stack([np.ones(obs.sum()), Zi[obs].to_numpy()])
    p_o = lr.predict_proba(Zi[obs])[:, 1]
    info = Zo.T @ (Zo * (p_o * (1 - p_o))[:, None]) + np.diag([0.0] + [1 / 0.1] * Zi.shape[1])
    cov = np.linalg.inv(info)
    beta = np.concatenate([lr.intercept_, lr.coef_[0]])
    rng = np.random.default_rng(seed)
    in_test = pool.index.isin(tst.index)
    Zt = np.column_stack([np.ones(in_test.sum()), Zi[in_test].to_numpy()])
    miss_t = ~tst["mrs90"].notna().to_numpy()
    imp_rows = []
    for _ in range(m):
        b = rng.multivariate_normal(beta, cov)
        p = 1 / (1 + np.exp(-(Zt @ b)))
        y = np.where(miss_t, (rng.random(len(p)) < p).astype(float), y_obs)
        imp_rows.append(row("imputed", y, allm))
    agg = {"scenario": "imputed", "n": imp_rows[0]["n"], "imputations": m}
    for k in imp_rows[0]:
        if k in ("scenario", "n"):
            continue
        v = np.array([r[k] for r in imp_rows])
        agg[k], agg[k + "::min"], agg[k + "::max"] = float(v.mean()), float(v.min()), float(v.max())
    rows.append(agg)
    return {"years": [test_from, int(tst["year"].max())], "n_all": int(len(tst)), "n_missing": int(miss_t.sum()),
            "models": names, "rows": rows,
            "imputation_model": "logistic regression (L2, C = 0.1) of mRS 0-2 on the admission model inputs, mTICI, "
                                "NIHSS at 24 h and day 7, sICH, oedema, decompression and the year (missing values: "
                                "median + indicator), fitted on every patient with a known outcome; coefficients drawn "
                                "from their approximate posterior for each imputation"}


# ------------------------------------------------------------------------------------- 2. decline
def decline(base: pd.DataFrame, primary: dict, m: int = 20, seed: int = 0, n_boot: int = 200) -> dict:
    from .splines import fit_spline_logreg, fit_spline_logreg_fixed

    stage, target = "admission", "good_outcome"
    d = stage_cohort(base, stage)
    d = d[d["year"] >= 2016]
    late = (d["year"] >= 2023).to_numpy()
    y = d[target].to_numpy(float)
    out = {"periods": [[2016, 2022], [2023, int(d["year"].max())]], "n": [int((~late).sum()), int(late.sum())],
           "good": [float(y[~late].mean()), float(y[late].mean())]}
    # mRS distribution and process of care
    mrs = d["mrs90"].to_numpy(int)
    tab = np.array([[np.sum((mrs == k) & (late == g)) for k in range(7)] for g in (False, True)])
    out["mrs"] = [[float(v) for v in r / r.sum()] for r in tab]
    out["mrs_p"] = float(stats.chi2_contingency(tab)[1])
    out["ivt"] = [float(d.loc[~late, "ivt"].mean()), float(d.loc[late, "ivt"].mean())]
    out["otp_median"] = [float(d.loc[~late, "onset_to_puncture"].median()), float(d.loc[late, "onset_to_puncture"].median())]
    # odds ratios for the period
    X = covariates(d)
    X["period"] = late.astype(float)
    crude_b, crude_V = _fit(X[["period"]], y)
    q = 1.959964
    out["or_crude"] = {"or": round(float(np.exp(crude_b[1])), 3), "lo": round(float(np.exp(crude_b[1] - q * np.sqrt(crude_V[1, 1]))), 3),
                       "hi": round(float(np.exp(crude_b[1] + q * np.sqrt(crude_V[1, 1]))), 3)}
    keep = [c for c in PATIENT if c in X.columns and X[c].notna().sum() > 0 and X[c].nunique(dropna=True) > 1]
    if keep:
        occ = [c for c in keep if c.startswith("occ_")]
        small = [c for c in occ if min((X[c] == 1).sum(), (X[c] == 0).sum()) < 10]
        keep = [c for c in keep if c not in small]
    Zs = _imputations(X[keep + PROCESS + ["period"]], d["mrs90"].to_numpy(float), m, seed)
    b, T, df = _pooled([Z[keep + ["period"]] for Z in Zs], y)
    out["or_adjusted"] = _or(b, T, df, "period")
    b, T, df = _pooled([Z[keep + PROCESS + ["period"]] for Z in Zs], y)
    out["or_adjusted_process"] = _or(b, T, df, "period")
    out["adjusted_for"] = keep
    # observed / expected by year: the reference specification fitted on 2016-2022
    Xd = design_matrix(d, stage)
    years = sorted(int(v) for v in d["year"].unique())
    early = d.index[~late]
    ref = fit_spline_logreg(Xd.loc[early], y[~late], seed)
    k, C = int(ref.settings["knots"]), float(ref.settings["C"])
    oe = []
    for yr in years:
        msk = (d["year"] == yr).to_numpy()
        if yr < 2023:
            f = fit_spline_logreg_fixed(Xd.loc[(~late) & ~msk], y[(~late) & ~msk], k, C)
        else:
            f = ref
        e = float(f.predict_proba(Xd.loc[msk])[:, 1].sum())
        o = float(y[msk].sum())
        n = int(msk.sum())
        lo_p, hi_p = _wilson(o, n)
        oe.append({"year": yr, "n": n, "observed": o / n, "expected": e / n, "oe": o / e if e else None,
                   "lo": lo_p * n / e if e else None, "hi": hi_p * n / e if e else None})
    out["oe_by_year"] = oe
    # share of the drop accounted for by the recorded characteristics
    def explained(idx_early, idx_late):
        f = fit_spline_logreg_fixed(Xd.loc[idx_early], d.loc[idx_early, target].to_numpy(float), k, C)
        p_e = f.predict_proba(Xd.loc[idx_early])[:, 1].mean()
        p_l = f.predict_proba(Xd.loc[idx_late])[:, 1].mean()
        drop = d.loc[idx_early, target].mean() - d.loc[idx_late, target].mean()
        return float((p_e - p_l) / drop) if drop else np.nan, float(p_l)

    point, exp_late = explained(early, d.index[late])
    rng = np.random.default_rng(seed)
    boots = []
    e_idx, l_idx = np.asarray(early), np.asarray(d.index[late])
    for _ in range(n_boot):
        be = rng.choice(e_idx, len(e_idx))
        bl = rng.choice(l_idx, len(l_idx))
        boots.append(explained(pd.Index(be), pd.Index(bl))[0])
    boots = np.asarray(boots, float)
    out["explained"] = {"share": point, "lo": float(np.nanpercentile(boots, 2.5)), "hi": float(np.nanpercentile(boots, 97.5)),
                        "expected_late": exp_late, "n_boot": n_boot}
    return out


def _wilson(k: float, n: int, z: float = 1.959964):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


# ------------------------------------------------------------------------------ 3. time to puncture
def time_to_puncture(base: pd.DataFrame, m: int = 20, seed: int = 0) -> dict:
    target = "good_outcome"
    d = base[base["onset_to_puncture"].notna()].copy()
    y = d[target].to_numpy(float)
    X = covariates(d)
    h = X["otp_h"]
    X["otp_0_6"] = h.clip(upper=6)
    X["otp_after6"] = (h - 6).clip(lower=0)
    cov = [c for c in PATIENT + ["ivt", "transfer"] if c in X.columns and X[c].nunique(dropna=True) > 1]
    cov = [c for c in cov if not (c.startswith("occ_") and min((X[c] == 1).sum(), (X[c] == 0).sum()) < 10)]
    Zs = _imputations(X[cov + ["otp_0_6", "otp_after6", "year5"]], d["mrs90"].to_numpy(float), m, seed)
    terms = cov + ["otp_0_6", "otp_after6"]
    b, T, df = _pooled([Z[terms] for Z in Zs], y)
    out = {"n": int(len(d)), "events": int(y.sum()),
           "bands": [_band(lab, ((h >= lo) & (h < hi)).to_numpy(), y) for lab, lo, hi in
                     (("0-3 h", 0, 3), ("3-6 h", 3, 6), ("6-12 h", 6, 12), ("12-24 h", 12, 24.01))],
           "per_hour_0_6": _or(b, T, df, "otp_0_6"), "per_hour_after_6": _or(b, T, df, "otp_after6"),
           "slope_change_p": _or(b, T, df, "", {"otp_after6": 1.0, "otp_0_6": -1.0})["p"]}
    # with the year of treatment
    b2, T2, df2 = _pooled([Z[terms + ["year5"]] for Z in Zs], y)
    out["per_hour_0_6_year"] = _or(b2, T2, df2, "otp_0_6")
    # modification by IV thrombolysis (0-6 h slope)
    def with_int(Z):
        D = Z[terms].copy()
        D["otp_0_6_x_ivt"] = D["otp_0_6"] * D["ivt"]
        return D

    b3, T3, df3 = _pooled(Zs, y, extra=with_int)
    out["per_hour_0_6_ivt"] = _or(b3, T3, df3, "", {"otp_0_6": 1.0, "otp_0_6_x_ivt": 1.0})
    out["per_hour_0_6_no_ivt"] = _or(b3, T3, df3, "otp_0_6")
    out["interaction_p"] = _or(b3, T3, df3, "otp_0_6_x_ivt")["p"]
    # standardised chance of mRS 0-2 at 1..6 h (first model, averaged over the imputations)
    curve = []
    for hh in (1, 2, 3, 4, 5, 6):
        ps = []
        for Z in Zs:
            D = Z[terms].copy()
            D["otp_0_6"], D["otp_after6"] = float(hh), 0.0
            lin = b["const"] + D.to_numpy(float) @ b[terms].to_numpy()
            ps.append(float(np.mean(1 / (1 + np.exp(-lin)))))
        curve.append({"hours": hh, "p": float(np.mean(ps))})
    out["standardised"] = curve
    out["adjusted_for"] = cov
    return out


def _band(label: str, mask: np.ndarray, y: np.ndarray) -> dict:
    """Patients in a time band and their observed share with mRS 0-2 (Wilson 95% CI); the share is
    published only when both outcome groups have at least privacy.min_cell patients."""
    n, k = int(mask.sum()), int(y[mask].sum())
    row = {"band": label, "n": _cell(n)}
    if _cell(k) is not None and _cell(n - k) is not None:
        lo, hi = _wilson(k, n)
        row.update({"good": k / n, "lo": lo, "hi": hi})
    return row


def _cell(n: int):
    """A count published only from privacy.min_cell patients."""
    return n if n >= int(settings()["privacy"]["min_cell"]) else None


def build_analyses(clean: pd.DataFrame) -> dict:
    from .cohort import flow
    from .paths import PRIVATE, PUBLIC

    cfg = settings()
    seed = int(cfg["seed"])
    base, _ = flow(clean)
    primary = json.loads((PRIVATE / "train_primary.json").read_text())
    t0 = time.time()
    out = {"missing_outcome": missing_outcome(clean, base, primary, seed=seed)}
    print(f"analyses: missing outcome ({time.time() - t0:.0f}s)", flush=True)
    out["decline"] = decline(base, primary, seed=seed)
    print(f"analyses: decline ({time.time() - t0:.0f}s)", flush=True)
    out["time_to_puncture"] = time_to_puncture(base, seed=seed)
    print(f"analyses: time to puncture ({time.time() - t0:.0f}s)", flush=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    from .export import clean_json

    with open(PUBLIC / "analyses.json", "w", encoding="utf-8") as fh:
        json.dump(clean_json(out), fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return out
