"""Add or refresh comparators in the temporal validation without retraining LightGBM.

`python -m mtop compare` re-fits every comparator on the development cohort (onset <= 2020),
keeps the stored LightGBM, logistic-regression, XGBoost and MLP predictions for the 2021-2025
patients, and recomputes for all models: temporal metrics, paired bootstrap CIs against LightGBM,
ROC points and decision curves. The deployed models and their fingerprint do not change.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import metrics as M
from .cohort import split_temporal, stage_cohort
from .config import settings, stage_ids, targets
from .features import design_matrix
from .paths import PRIVATE
from .scores import benchmark_predictions
from .splines import fit_spline_logreg

ORDER = ["LightGBM", "Logistic regression", "Logistic regression (splines)", "XGBoost", "MLP"]
STORED = {"LightGBM", "Logistic regression", "XGBoost", "MLP"}


def comparator_predictions(dev: pd.DataFrame, test: pd.DataFrame, X_dev, X_test, stage, target, seed) -> dict:
    y_dev = dev[target].to_numpy(float)
    out = {"Logistic regression (splines)": fit_spline_logreg(X_dev, y_dev, seed).predict_proba(X_test)[:, 1]}
    out.update(benchmark_predictions(dev, test, stage, target, seed))
    return out


def decision_curves(y, preds: dict) -> dict:
    return {k: [{"t": r["t"], "nb": r["model"]} for r in M.decision_curve(y, p)] for k, p in preds.items()}


def run(base: pd.DataFrame) -> None:
    cfg = settings()
    seed = int(cfg["seed"])
    n_boot = int(cfg["bootstrap"]["n"])
    res_path = PRIVATE / "train_results.json"
    res = json.loads(res_path.read_text())
    stored = pd.read_parquet(PRIVATE / "temporal_predictions.parquet")
    rows = []
    for stage in stage_ids():
        d = stage_cohort(base, stage)
        dev, test = split_temporal(d)
        X = design_matrix(d, stage)
        X_dev, X_test = X.loc[dev.index], X.loc[test.index]
        for target in targets():
            y_test = test[target].to_numpy(float)
            s = stored[(stored["stage"] == stage) & (stored["target"] == target)]
            preds = {}
            for name in ORDER:
                if name in STORED:
                    m = s[s["model"] == name].set_index("patient_id")
                    p = m.loc[test["patient_id"].to_numpy(), "p"].to_numpy(float)
                    assert np.allclose(m.loc[test["patient_id"].to_numpy(), "y"].to_numpy(float), y_test)
                    preds[name] = p
            new = comparator_predictions(dev, test, X_dev, X_test, stage, target, seed)
            preds = {k: (preds[k] if k in preds else new[k]) for k in ORDER if k in preds or k in new}
            preds.update({k: v for k, v in new.items() if k not in preds})
            temporal = {k: M.summary(y_test, v) for k, v in preds.items()}
            ci = M.bootstrap_ci(y_test, preds, n_boot=n_boot, seed=seed, ref="LightGBM")
            for k in temporal:
                temporal[k]["ci"] = ci[k]
            tres = res["stages"][stage]["targets"][target]
            tres["temporal"] = temporal
            tres["curves"]["roc"] = {k: M.roc_points(y_test, v) for k, v in preds.items()}
            tres["curves"]["dca_models"] = decision_curves(y_test, preds)
            for k, v in preds.items():
                rows.append(pd.DataFrame({"patient_id": test["patient_id"].to_numpy(), "stage": stage,
                                          "target": target, "model": k, "y": y_test, "p": v}))
            print(f"[compare {stage}/{target}] " + ", ".join(f"{k} {temporal[k]['auroc']:.3f}" for k in preds), flush=True)
    res_path.write_text(json.dumps(res, indent=1))
    pd.concat(rows).to_parquet(PRIVATE / "temporal_predictions.parquet", index=False)
