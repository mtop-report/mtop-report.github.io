"""Model diagnostics of the reference model (MTOP v1.0.0), as aggregates for the model card and the
paper's supplement: tuned settings, temporal-test calibration of the development models before any
recalibration, missing values and rare categories behind the published missing-value terms, and the
bootstrap internal validation (`mtop.optimism`) when its results exist.

    python -m mtop.diagnostics        writes docs/model_diagnostics.md

Reads the private training results and the cleaned registry; writes counts only. A count from 1 to 9
is shown as "<10", and an outcome count is withheld (·) when it or its complement within the group is
under 10, as everywhere in the app. Nothing here changes a model.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

DOC = paths.ROOT / "docs" / "model_diagnostics.md"
REF = "Logistic regression (splines)"
STAGE_LABEL = {"admission": "Admission", "procedure": "End of procedure", "h24": "24 hours", "d7": "Day 7"}
TARGET_LABEL = {"good_outcome": "mRS 0–2", "death90": "death"}
MIN_CELL = 10
SPARSE = 20          # the minimum for a missing-value term proposed for version 1.1 (model card)


def cnt(n: int) -> str:
    n = int(n)
    return "0" if n == 0 else ("<10" if n < MIN_CELL else f"{n:,}")


def events_cell(events: int, group: int) -> str:
    """An outcome count within a group, withheld when it or its complement is under 10."""
    events, group = int(events), int(group)
    if group == 0:
        return "—"
    if events < MIN_CELL or group - events < MIN_CELL:
        return "·"
    return f"{events:,}"


def ci(v) -> str:
    return f"{v[0]:.2f} to {v[1]:.2f}"


def _table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- sections
def tuning_section(train: dict, bundle: dict) -> str:
    from .lr import C_GRID
    rows = []
    for st, v in train["stages"].items():
        for tg, w in v["targets"].items():
            s = w["settings"][REF]
            spec = bundle["models"][st][tg]["learners"][REF]["spec"]
            rows.append([STAGE_LABEL[st], TARGET_LABEL[tg], f"{w['n_dev']:,} ({w['events_dev']:,})", s["knots"],
                         f"{float(s['C']):.4f}", f"{float(s['cv_logloss']):.4f}", len(spec["coef"]), len(spec["indicators"])])
    return ("## Tuned settings of the reference model\n\n"
            "Chosen on the development years (2005–2022) by 5-fold cross-validated log-loss: 3, 4 or 5 spline knots and the "
            f"penalty C from 50 values ({C_GRID[0]:g} to {C_GRID[-1]:g}, log scale; smaller C means stronger shrinkage). The "
            "deployed models were refitted on all years with these settings; their coefficient counts include the "
            "missing-value terms.\n\n"
            + _table(["Stage", "Outcome", "Development n (events)", "Knots", "C", "CV log-loss", "Coefficients, deployed model",
                      "of which missing-value terms"], rows)
            + "\n\nNo C lies at either end of the grid. Every model except death at day 7 chose 3 knots, the smallest option.\n")


def calibration_section(train: dict) -> str:
    rows = []
    for st, v in train["stages"].items():
        for tg, w in v["targets"].items():
            m, lr = w["models"][REF], w["models"].get("Logistic regression")
            rows.append([STAGE_LABEL[st], TARGET_LABEL[tg], f"{w['n_test']:,} ({cnt(w['events_test'])})",
                         f"{m['auroc']:.3f} ({ci(m['ci']['auroc'])})", f"{m['citl']:.2f} ({ci(m['ci']['citl'])})",
                         f"{m['slope']:.2f} ({ci(m['ci']['slope'])})", f"{lr['slope']:.2f}" if lr else "—"])
    return ("## Temporal test: the development models before recalibration\n\n"
            f"Development models (2005–2022) on the test years ({train['test_from_year']}–2025), before any recalibration; the "
            "deployed models' intercepts were re-estimated on these years only afterwards, so these are the untouched "
            "validation results. 95 % bootstrap intervals. Calibration-in-the-large below 0: the model over-predicts the "
            "outcome in these years. Slope above 1: the estimates are too close together (too timid); below 1: too spread "
            "out (too extreme).\n\n"
            + _table(["Stage", "Outcome", "Test n (events)", "AUROC", "Calibration-in-the-large", "Calibration slope",
                      "Slope, linear regression without splines"], rows) + "\n")


def missing_section(base: pd.DataFrame, bundle: dict) -> str:
    from .cohort import stage_cohort
    from .config import stage_features
    from .features import design_matrix
    rows, complete = [], set()
    for st in bundle["models"]:
        d = stage_cohort(base, st)
        X = design_matrix(d, st)
        n_train = bundle["models"][st]["good_outcome"]["n_train"]
        assert len(d) == n_train, f"{st}: cohort {len(d)} differs from the deployed models' {n_train}"
        terms = set()
        for tg in ("good_outcome", "death90"):
            spec = bundle["models"][st][tg]["learners"][REF]["spec"]
            terms |= {spec["columns"][i] for i in spec["indicators"]}
        for f in stage_features(st):
            if f.stage != st:
                continue
            miss = X[f.columns[0]].isna().to_numpy()
            n = int(miss.sum())
            label = f.spec["label"]["en"]
            if n == 0:
                complete.add(label)
                continue
            has = any(c in terms for c in f.columns)
            g = d["good_outcome"].to_numpy()[miss].sum()
            de = d["death90"].to_numpy()[miss].sum()
            rows.append([STAGE_LABEL[st], label, cnt(n), "yes" if has else "no", events_cell(g, n), events_cell(de, n),
                         "**yes**" if has and n < SPARSE else ""])
    return ("## Missing values behind the missing-value terms\n\n"
            "Patients of each stage's cohort (all years, the data the deployed models were fitted on) without the input, by the "
            "stage at which the input enters. Every later stage's model carries the same inputs and terms again. Missing "
            "includes a category without a model term (for example ACA). Counts under 10 are shown as <10; outcome counts are "
            "withheld (·) when they or their complement are under 10. The last column marks terms that the rule proposed for "
            "version 1.1 (at least 20 patients without the input) would remove.\n\n"
            + _table(["Stage", "Input", "Patients without it", "Missing-value term", "mRS 0–2 among them", "Deaths among them",
                      "Under 20 patients"], rows)
            + "\n\nRecorded for every patient: " + ", ".join(sorted(complete)) + ".\n")


def rare_section(base: pd.DataFrame, bundle: dict) -> str:
    from .cohort import stage_cohort
    from .config import features
    rows = []
    for f in features():
        if f.type not in ("categorical", "binary") or f.stage not in bundle["models"]:
            continue
        d = stage_cohort(base, f.stage)
        src = d[f.key]
        if f.type == "categorical":
            opts = [(o["value"], o["label"]["en"], o["value"] == f.spec.get("reference"), o.get("model", True)) for o in f.spec["options"]]
        else:
            opts = [(1, "yes", False, True), (0, "no", False, True)]
        for val, lab, is_ref, model in opts:
            n = int((src == val).sum())
            if n >= SPARSE:
                continue
            kind = "reference" if is_ref else ("model term" if model else "no model term (counted as missing)")
            rows.append([STAGE_LABEL[f.stage], f.spec["label"]["en"], lab, cnt(n), kind])
    return ("## Rare categories\n\n"
            "Categories and yes/no values held by fewer than 20 patients of the stage's cohort (all years). Every other "
            "category has at least 20. Ordinal inputs (pre-stroke mRS, mTICI) enter as one numeric term each, so their "
            "rare grades (for example pre-stroke mRS 3–5) add no term of their own but sit in a sparse part of that term.\n\n"
            + _table(["Stage", "Input", "Category", "Patients", "In the model"], rows) + "\n")


def optimism_section() -> str:
    files = sorted((paths.REPORTS / "optimism").glob("*_*.json")) if (paths.REPORTS / "optimism").exists() else []
    files = [f for f in files if not f.name.endswith("_resamples.jsonl")]
    if not files:
        return "## Bootstrap internal validation\n\nNot yet run (plan: docs/registrations/bootstrap_optimism_plan.md).\n"
    names = {"auroc": "AUROC", "brier": "Brier score", "logloss": "Log-loss", "citl": "Calibration-in-the-large",
             "slope": "Calibration slope", "oe": "Observed/expected"}
    parts = ["## Bootstrap internal validation\n\n"
             "Plan: docs/registrations/bootstrap_optimism_plan.md (committed before the run). In each of 500 bootstrap samples "
             "of the development patients the whole pipeline was repeated: knot placement and number, imputation and "
             "scaling, and the penalty tuned by cross-validated log-loss. Optimism = performance on the bootstrap sample "
             "minus performance of the same model on the original patients; corrected = apparent − mean optimism. "
             "Percentiles: 2.5th–97.5th of the 500 optimism values.\n"]
    for f in files:
        r = json.loads(f.read_text())
        rows = [[names[k], f"{o['apparent']:.3f}", f"{o['mean_optimism']:+.3f} (±{o['monte_carlo_se']:.3f})",
                 f"{o['optimism_p2_5']:+.3f} to {o['optimism_p97_5']:+.3f}", f"{o['corrected']:.3f}"]
                for k, o in r["optimism"].items()]
        s = r["stability"]
        rep = r["reproduction"]
        parts.append(f"### {STAGE_LABEL[r['stage']]}, {TARGET_LABEL[r['target']]} (n = {r['n']:,}; {r['events']:,} events)\n\n"
                     + _table(["Measure", "Apparent", "Mean optimism (Monte Carlo SE)", "Optimism, 2.5th–97.5th percentile",
                               "Optimism-corrected"], rows)
                     + f"\n\n- Pipeline check: the refit reproduced the published settings ({rep['refit']['knots']} knots, "
                       f"C = {rep['refit']['C']:.4f}) and test-year estimates (largest difference {rep['test_predictions_max_difference']:.1e}). "
                       f"With grouped inner folds on the original data: {r['original_with_grouped_folds']['knots']} knots, "
                       f"C = {r['original_with_grouped_folds']['C']:.4f}.\n"
                     + f"- Knots chosen: 3 in {s['knots_share']['3']:.0%}, 4 in {s['knots_share']['4']:.0%}, 5 in "
                       f"{s['knots_share']['5']:.0%} of samples. C: median {10 ** s['log10_C']['median']:.4f} (interquartile range "
                       f"{10 ** s['log10_C']['q1']:.4f}–{10 ** s['log10_C']['q3']:.4f}); at an end of the grid in "
                       f"{s['share_C_at_grid_end']:.0%}.\n"
                     + f"- Each development patient's risk across the 500 models: median width of its 95 % range "
                       f"{100 * s['risk_range_width']['median']:.1f} percentage points (90th percentile "
                       f"{100 * s['risk_range_width']['p90']:.1f}); wider than 10 points for "
                       f"{s['risk_range_width']['share_wider_than_10_points']:.0%} of patients.\n"
                     + f"- Resamples with convergence warnings: {s['resamples_with_convergence_warnings']}; on average "
                       f"{s['mean_unique_patients']:.0f} distinct patients per sample.\n")
    return "\n".join(parts)


def build() -> Path:
    from .cohort import flow
    train = json.loads((paths.PRIVATE / "train_primary.json").read_text())
    bundle = json.loads((paths.PUBLIC / "model_bundle.json").read_text())
    base, _ = flow(pd.read_parquet(paths.PROCESSED / "clean.parquet"))
    head = ("# MTOP v1.0.0: model diagnostics\n\n"
            f"Generated by `python -m mtop.diagnostics` on {dt.date.today().isoformat()} for the model bundle with fingerprint "
            f"{bundle['fingerprint'][:8]}… (unchanged). Aggregates only: counts from 1 to 9 are shown as <10. These tables "
            "describe the published models; they do not change them.\n")
    md = "\n".join([head, tuning_section(train, bundle), calibration_section(train), missing_section(base, bundle),
                    rare_section(base, bundle), optimism_section()])
    DOC.write_text(md, encoding="utf-8")
    return DOC


if __name__ == "__main__":
    print(build())
