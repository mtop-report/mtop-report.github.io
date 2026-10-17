"""Reconcile MTOP with the earlier FNO analysis ("Predikce výsledku trombektomie", HTML report).

The earlier report (2026) used the same registry export with different choices:
  cohort   1,577 -> 1,543 (age < 18 and 29 patients with impossible times removed) -> 1,422 with a
           known 90-day mRS; diagnostic-only angiographies were kept
  split    train 2005-2022 (n = 1,201), test 2023-2025 (n = 221, 26.2 % mRS 0-2)
  models   33 pre-procedural predictors; L2 logistic regression with splines was best (AUC 0.815),
           no machine-learning model beat it; 7-item bedside score AUC 0.774.
MTOP keeps patients with impossible times (the times become missing), drops the diagnostic-only
angiographies, tests on 2021-2025 and uses 17 admission predictors.

This module rebuilds the report's cohort and split from the raw registry, then compares the five
learners (LightGBM, XGBoost, MLP, logistic regression linear and with splines; each tuned anew on
every set-up's training patients with the same budget, tuning.py) and the scores on both set-ups,
with and without 12 extra history and lifestyle predictors. Output (aggregates only):
reports/reconciliation.md and artifacts/public/reconciliation.json.
"""
from __future__ import annotations

import json

import pandas as pd

from . import metrics as M
from .config import settings
from .features import design_matrix, monotone_vector
from .paths import PUBLIC, REPORTS
from .scores import POINT_SCORES, apply_points_map, fit_points_map
from .splines import fit_spline_logreg
from .train import _params, fit_lgbm, tune_lgbm
from .tuning import comparator_models, jsonable, lgbm_settings

REPORT_NUMBERS = {"registry": 1577, "after_exclusions": 1543, "known_mrs": 1422, "train": 1201, "test": 221,
                  "test_good_rate": 0.262, "auc_spline_lr": 0.815, "auc_bedside": 0.774, "auc_bedside_lr": 0.793,
                  "auc_span100": 0.732, "auc_thrive": 0.713}
# timing errors the report treated as impossible (it removed the patients; MTOP sets the times missing)
IMPOSSIBLE_TIMES = [("onset_to_recan", "recanalisation before puncture"), ("onset_to_recan", "outside"),
                    ("procedure_min", "outside")]
BEDSIDE_VARS = ["age", "nihss_adm", "glucose", "mrs_pre", "aspects", "onset_to_puncture", "stroke_prior"]
EXTRA_PREDICTORS = ["hld", "smoking", "alcohol", "tia_prior", "statin", "antiplatelet", "heart_structural", "coronary",
                    "bmi", "dbp", "onset_hour", "posterior"]


def report_cohort(clean: pd.DataFrame, issues: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    idcol = "patient_id"
    bad = set()
    for col, action in IMPOSSIBLE_TIMES:
        m = (issues["column"] == col) & issues["action"].str.contains(action, regex=False)
        bad |= set(issues.loc[m, idcol])
    steps = [{"step": "registry", "n": len(clean)}]
    d = clean[~(clean["age"] < 18)]
    d = d[~d["patient_id"].isin(bad)]
    steps.append({"step": "age < 18 and impossible times removed", "n": len(d)})
    d = d[d["mrs90"].notna()]
    steps.append({"step": "90-day mRS known", "n": len(d)})
    steps.append({"step": "of which diagnostic angiography only", "n": int((d["diagnostic_only"] == 1).sum())})
    return d.copy(), steps


def _lgbm(X_tr, y_tr, X_te, seed, trials):
    """LightGBM tuned on this set-up's training patients exactly as in train.py."""
    cfg = settings()
    mono = monotone_vector("admission", "good_outcome")
    if len(mono) != X_tr.shape[1]:
        mono = mono + [0] * (X_tr.shape[1] - len(mono))
    best, rounds, cv_ll = tune_lgbm(X_tr, y_tr, mono, seed, trials, cfg["tuning"]["max_rounds"],
                                    cfg["tuning"]["early_stopping"], int(cfg["privacy"]["min_leaf"]))
    model = fit_lgbm(X_tr, y_tr, _params(best, mono, seed), rounds)
    return model.predict(X_te), lgbm_settings(best, rounds, cv_ll, trials)


NAMES = {"Logistic regression": "Logistic regression (linear)"}


def compare(d: pd.DataFrame, test_from: int, seed: int, n_boot: int, extended: bool = False, trials: int = 50) -> dict:
    d = d[d["year"].notna()]
    tr, te = d[d["year"] < test_from], d[d["year"] >= test_from]
    X = design_matrix(d, "admission")
    if extended:
        X = pd.concat([X, d[EXTRA_PREDICTORS].astype(float)], axis=1)
    X_tr, X_te = X.loc[tr.index], X.loc[te.index]
    y_tr, y_te = tr["good_outcome"].to_numpy(float), te["good_outcome"].to_numpy(float)
    preds, used = {}, {}
    preds["LightGBM"], used["LightGBM"] = _lgbm(X_tr, y_tr, X_te, seed, trials)
    for name, mdl in comparator_models(X_tr, y_tr, seed, trials).items():
        preds[NAMES.get(name, name)] = mdl.predict_proba(X_te)[:, 1]
        used[NAMES.get(name, name)] = jsonable(mdl.settings)
    order = ["Logistic regression (splines)", "LightGBM", "XGBoost", "MLP", "Logistic regression (linear)"]
    preds = {k: preds[k] for k in order}
    if not extended:
        b7 = d[BEDSIDE_VARS].astype(float)
        preds["7-variable regression (splines)"] = fit_spline_logreg(b7.loc[tr.index], y_tr, seed).predict_proba(b7.loc[te.index])[:, 1]
        for name in ("Bedside score", "SPAN-100", "THRIVE"):
            mp = fit_points_map(POINT_SCORES[name](tr), y_tr)
            preds[name] = apply_points_map(mp, POINT_SCORES[name](te))
        span = lambda df: df["age"] + df["nihss_adm"]  # noqa: E731  (the report's SPAN, as a continuous index)
        preds["SPAN index (age + NIHSS)"] = apply_points_map(fit_points_map(span(tr), y_tr), span(te))
    ci = M.bootstrap_ci(y_te, preds, n_boot=n_boot, seed=seed, ref="Logistic regression (splines)")
    out = {"n_train": int(len(tr)), "n_test": int(len(te)), "test_good_rate": float(y_te.mean()),
           "train_good_rate": float(y_tr.mean()), "predictors": int(X.shape[1]), "models": {}, "settings": used}
    for k, p in preds.items():
        s = M.summary(y_te, p)
        out["models"][k] = {"auroc": s["auroc"], "auroc_ci": ci[k].get("auroc"), "citl": s["citl"], "slope": s["slope"],
                            "delta_vs_spline_lr": ci[k].get("delta_auroc_vs_ref")}
    return out


def run(clean: pd.DataFrame, n_boot: int = 1000, quick: bool = False) -> dict:
    seed = int(settings()["seed"])
    trials = 8 if quick else int(settings()["tuning"]["trials"])
    issues = pd.read_excel(REPORTS / "data_quality.xlsx", sheet_name="issues")
    rep, rep_steps = report_cohort(clean, issues)
    from .cohort import flow
    mtop_base, mtop_steps = flow(clean)
    mtop_test_from = int(settings()["split"]["temporal_test_from_year"])
    out = {
        "report_numbers": REPORT_NUMBERS,
        "report_cohort_rebuilt": rep_steps,
        "mtop_cohort": mtop_steps,
        "setups": {
            "report cohort, report split (test 2023-2025)": compare(rep, 2023, seed, n_boot, trials=trials),
            "report cohort, report split, + 12 extra predictors": compare(rep, 2023, seed, n_boot, extended=True, trials=trials),
            "MTOP cohort, report split (test 2023-2025)": compare(mtop_base, 2023, seed, n_boot, trials=trials),
            "MTOP cohort, MTOP split (test 2021-2025)": compare(mtop_base, mtop_test_from, seed, n_boot, trials=trials),
            "MTOP cohort, MTOP split, + 12 extra predictors": compare(mtop_base, mtop_test_from, seed, n_boot, extended=True,
                                                                      trials=trials),
        },
    }
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "reconciliation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (REPORTS / "reconciliation.md").write_text(to_markdown(out), encoding="utf-8")
    return out


def to_markdown(r: dict) -> str:
    L = ["# MTOP vs the earlier FNO analysis (HTML report)", "",
         "## Cohort", "", "| Step | Report | Rebuilt here |", "|---|---|---|"]
    rn = r["report_numbers"]
    rb = {s["step"]: s["n"] for s in r["report_cohort_rebuilt"]}
    L.append(f"| Registry | {rn['registry']} | {rb['registry']} |")
    L.append(f"| Age < 18 and impossible times removed | {rn['after_exclusions']} | {rb['age < 18 and impossible times removed']} |")
    L.append(f"| 90-day mRS known | {rn['known_mrs']} | {rb['90-day mRS known']} |")
    L.append(f"| of which diagnostic angiography only | not reported | {rb['of which diagnostic angiography only']} |")
    L += ["", "MTOP: " + " -> ".join(f"{s['step']} {s['n']}" for s in r["mtop_cohort"]), ""]
    for name, s in r["setups"].items():
        L += [f"## {name}", "",
              f"Train n = {s['n_train']} ({s['train_good_rate']:.1%} mRS 0-2), test n = {s['n_test']} "
              f"({s['test_good_rate']:.1%}), {s['predictors']} predictor columns.", "",
              "| Model | AUROC (95% CI) | Δ vs spline regression (95% CI) | Cal. intercept | Cal. slope |", "|---|---|---|---|---|"]
        for k, m in s["models"].items():
            ci = m["auroc_ci"] or [float("nan")] * 2
            dl = m["delta_vs_spline_lr"]
            dtxt = "–" if not dl else f"{-dl['mean']:+.3f} ({-dl['ci'][1]:+.3f} to {-dl['ci'][0]:+.3f})"
            L.append(f"| {k} | {m['auroc']:.3f} ({ci[0]:.3f}–{ci[1]:.3f}) | {dtxt} | {m['citl']:+.2f} | {m['slope']:.2f} |")
        L.append("")
    return "\n".join(L)
