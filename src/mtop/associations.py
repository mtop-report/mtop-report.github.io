"""Factors associated with 90-day outcome: unadjusted and adjusted odds ratios.

For mRS 0-2 and for death within 90 days, every admission and procedure factor gets
  - an unadjusted odds ratio (one-variable logistic regression, patients with the value known), and
  - an adjusted odds ratio from one multivariable logistic regression with all factors and the
    treatment year. Missing values are multiply imputed (20 imputations by chained equations,
    with the 90-day mRS and the year in the imputation model) and estimates are pooled with
    Rubin's rules, so patients with a missing ASPECTS or glucose still count.
These are associations in one registry, not causal effects: e.g. successful reperfusion is
partly a marker of favourable anatomy. Output is aggregate only (coefficients and confidence
intervals); a binary factor is reported only when both of its groups have >= min_cell patients.
Writes artifacts/public/associations.json (python -m mtop associations).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

OCCLUSION_REF = "M1"
OCCLUSIONS = [("ICA", "Intracranial ICA", "Intrakraniální ACI"),
              ("tandem", "Tandem (cervical ICA/CCA + intracranial)", "Tandem (krční ACI/ACC + intrakraniální)"),
              ("cervical_ICA", "Cervical ICA / CCA", "Krční ACI / ACC"),
              ("M2", "MCA M2 / M3", "ACM M2 / M3"),
              ("basilar", "Basilar artery", "Arteria basilaris"),
              ("basilar_vertebral", "Basilar + vertebral artery", "A. basilaris + AV"),
              ("other_posterior", "Vertebral / PCA", "AV / ACP")]

# key, English, Czech, unit (EN, CZ), group, kind
TERMS = [
    ("age10", "Age", "Věk", ("per 10 years", "na 10 let"), "patient", "num"),
    ("female", "Female sex", "Žena", None, "patient", "bin"),
    ("prestroke_disability", "Pre-stroke mRS ≥ 1", "mRS před CMP ≥ 1", None, "patient", "bin"),
    ("htn", "Hypertension", "Hypertenze", None, "patient", "bin"),
    ("dm", "Diabetes mellitus", "Diabetes mellitus", None, "patient", "bin"),
    ("arrhythmia", "Atrial fibrillation / arrhythmia", "Fibrilace síní / arytmie", None, "patient", "bin"),
    ("stroke_prior", "Previous stroke", "Předchozí CMP", None, "patient", "bin"),
    ("anticoag", "Anticoagulated before stroke", "Antikoagulace před CMP", None, "patient", "bin"),
    ("nihss5", "NIHSS on admission", "NIHSS při přijetí", ("per 5 points", "na 5 bodů"), "stroke", "num"),
    ("aspects", "ASPECTS", "ASPECTS", ("per point", "na bod"), "stroke", "num"),
    ("glucose", "Glucose on admission", "Glykémie při přijetí", ("per mmol/L", "na mmol/l"), "stroke", "num"),
    ("sbp10", "Systolic blood pressure", "Systolický tlak", ("per 10 mmHg", "na 10 mmHg"), "stroke", "num"),
] + [(f"occ_{k}", en, cs, ("vs MCA M1", "vs ACM M1"), "stroke", "occ") for k, en, cs in OCCLUSIONS] + [
    ("otp_h", "Onset to groin puncture", "Od vzniku do punkce", ("per hour", "na hodinu"), "treatment", "num"),
    ("ivt", "IV thrombolysis", "IV trombolýza", None, "treatment", "bin"),
    ("transfer", "Transferred from another hospital", "Překlad z jiné nemocnice", None, "treatment", "bin"),
    ("general_anaesthesia", "General anaesthesia", "Celková anestezie", ("vs sedation or local", "vs analgosedace/lokální"), "treatment", "bin"),
    ("procedure30", "Procedure duration", "Délka výkonu", ("per 30 min", "na 30 min"), "treatment", "num"),
    ("reperfusion", "Successful reperfusion (mTICI 2b–3)", "Úspěšná reperfuze (mTICI 2b–3)", None, "treatment", "bin"),
    ("year5", "Treatment year", "Rok léčby", ("per 5 years", "na 5 let"), "treatment", "num"),
]
BINARY = [t[0] for t in TERMS if t[5] == "bin"]
OCC_COLS = [f"occ_{k}" for k, _, _ in OCCLUSIONS]


def covariates(base: pd.DataFrame) -> pd.DataFrame:
    d = base
    X = pd.DataFrame(index=d.index)
    X["age10"] = d["age"] / 10
    X["female"] = 1.0 - d["sex_male"]
    X["prestroke_disability"] = (d["mrs_pre"] >= 1).astype(float).where(d["mrs_pre"].notna())
    for c in ("htn", "dm", "arrhythmia", "stroke_prior", "anticoag", "ivt", "transfer"):
        X[c] = d[c].astype(float)
    X["nihss5"] = d["nihss_adm"] / 5
    X["aspects"] = d["aspects"]
    X["glucose"] = d["glucose"]
    X["sbp10"] = d["sbp"] / 10
    for k, _, _ in OCCLUSIONS:
        X[f"occ_{k}"] = (d["occlusion"] == k).astype(float).where(d["occlusion"].notna())
    X["otp_h"] = d["onset_to_puncture"].clip(upper=1440) / 60
    X["general_anaesthesia"] = (d["anesthesia"] == "general").astype(float).where(d["anesthesia"].notna())
    X["procedure30"] = d["procedure_min"] / 30
    X["reperfusion"] = d["reperfusion_2b3"].astype(float)
    X["year5"] = (d["year"] - 2015) / 5
    return X[[t[0] for t in TERMS]].astype(float)


def _fit(X: pd.DataFrame, y: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0, maxiter=200)
    return res.params.drop("const"), res.cov_params().drop(index="const", columns="const")


def unadjusted(X: pd.DataFrame, y: np.ndarray) -> dict:
    out = {}
    occ_cols = [c for c in OCC_COLS if c in X.columns]
    occ = X[occ_cols]
    ok = occ.notna().all(axis=1).to_numpy()
    b, V = _fit(occ[ok], y[ok])
    for c in occ_cols:
        out[c] = (b[c], np.sqrt(V.loc[c, c]), int(ok.sum()))
    for c in X.columns:
        if c in occ_cols:
            continue
        ok = X[c].notna().to_numpy()
        b, V = _fit(X.loc[ok, [c]], y[ok])
        out[c] = (b[c], np.sqrt(V.loc[c, c]), int(ok.sum()))
    return out


def adjusted(X: pd.DataFrame, ys: dict[str, np.ndarray], aux: np.ndarray, m: int, seed: int) -> dict:
    """Multiple imputation by chained equations, then Rubin's rules.

    The imputation model holds every factor plus the 90-day mRS (aux), so one set of m imputed
    data sets serves both outcomes, which are both functions of the mRS."""
    data = np.column_stack([X.to_numpy(), aux])
    lo = np.nanmin(data, axis=0)
    hi = np.nanmax(data, axis=0)
    occ_cols = [c for c in OCC_COLS if c in X.columns]
    binary = [c for c in BINARY if c in X.columns]
    miss_occ = X[occ_cols].isna().any(axis=1).to_numpy()
    est = {t: [] for t in ys}
    var = {t: [] for t in ys}
    for k in range(m):
        imp = IterativeImputer(sample_posterior=True, max_iter=15, random_state=seed + k, min_value=lo, max_value=hi)
        Z = pd.DataFrame(imp.fit_transform(data)[:, :-1], columns=X.columns, index=X.index)
        for c in binary:
            Z[c] = (Z[c] >= 0.5).astype(float)
        if miss_occ.any():                    # an imputed occlusion site: at most one category
            occ = Z[occ_cols].to_numpy().copy()
            part = occ[miss_occ]
            top = part.argmax(axis=1)
            keep = part.max(axis=1) >= 0.5
            filled = np.zeros_like(part)
            filled[np.arange(len(top))[keep], top[keep]] = 1.0
            occ[miss_occ] = filled
            Z[occ_cols] = occ
        if "aspects" in Z:
            Z["aspects"] = Z["aspects"].round()
        for t, y in ys.items():
            b, V = _fit(Z, y)
            est[t].append(b.to_numpy())
            var[t].append(np.diag(V.to_numpy()))
    out = {}
    for t in ys:
        e, v = np.array(est[t]), np.array(var[t])
        qbar = e.mean(0)
        W = v.mean(0)
        B = e.var(0, ddof=1)
        T = W + (1 + 1 / m) * B
        r = (1 + 1 / m) * B / np.maximum(W, 1e-12)
        df = (m - 1) * (1 + 1 / np.maximum(r, 1e-12)) ** 2
        tq = stats.t.ppf(0.975, df)
        out[t] = {c: (qbar[i], np.sqrt(T[i]), float(tq[i])) for i, c in enumerate(X.columns)}
    return out


def _or(b, se, q=1.959964) -> dict:
    return {"or": round(float(np.exp(b)), 3), "lo": round(float(np.exp(b - q * se)), 3), "hi": round(float(np.exp(b + q * se)), 3)}


def build(base: pd.DataFrame, min_cell: int, m: int = 20, seed: int = 20261016) -> dict:
    d = base[base["mrs90"].notna()]
    X = covariates(d)
    out = {"n": int(len(d)), "imputations": m, "min_cell": min_cell, "terms": [], "outcomes": {}}
    groups = {"patient": ("Patient", "Pacient"), "stroke": ("Stroke", "Iktus"), "treatment": ("Treatment", "Léčba")}
    # A factor or occlusion category with fewer than min_cell patients on either side is neither
    # estimable nor publishable, so it leaves the model. Patients of a dropped occlusion category
    # count as an unknown site (imputed), not as the reference M1.
    keep = []
    for key, en, cs, unit, grp, kind in TERMS:
        x = X[key]
        term = {"key": key, "label": {"en": en, "cs": cs}, "unit": {"en": unit[0], "cs": unit[1]} if unit else None,
                "group": grp, "kind": kind, "missing": round(100 * float(x.isna().mean()), 1)}
        if kind in ("bin", "occ"):
            n1, n0 = int((x == 1).sum()), int((x == 0).sum())
            if min(n1, n0) < min_cell:
                if kind == "occ" and n1:
                    X.loc[x == 1, OCC_COLS] = np.nan
                continue
            term["n1"], term["n0"] = n1, n0
        elif x.nunique(dropna=True) < 2:
            continue
        keep.append(key)
        out["terms"].append(term)
    X = X[keep]
    out["groups"] = {k: {"en": v[0], "cs": v[1]} for k, v in groups.items()}
    ys = {t: d[t].to_numpy(float) for t in ("good_outcome", "death90")}
    ad = adjusted(X, ys, d["mrs90"].to_numpy(float), m, seed)
    for target, y in ys.items():
        un = unadjusted(X, y)
        res = {"n": int(len(y)), "events": int(y.sum()), "terms": {}}
        for t in out["terms"]:
            k = t["key"]
            b, se, n = un[k]
            ba, sa, q = ad[target][k]
            res["terms"][k] = {"unadjusted": {**_or(b, se), "n": n}, "adjusted": _or(ba, sa, q)}
        out["outcomes"][target] = res
    return out


def build_associations(clean: pd.DataFrame, m: int = 20) -> dict:
    from .cohort import flow
    from .config import settings
    from .paths import PUBLIC

    base, _ = flow(clean)
    out = build(base, int(settings()["privacy"]["min_cell"]), m=m)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    with open(PUBLIC / "associations.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"associations: {len(out['terms'])} factors, {out['n']} patients, {out['imputations']} imputations")
    return out
