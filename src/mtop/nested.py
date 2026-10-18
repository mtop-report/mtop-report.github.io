"""Nested cross-validation on all years (internal validation of the whole modelling procedure).

As in the 2026 FNO analysis: the patients of all years are split into 5 outer folds, 3 times over
(15 outer folds); inside every outer training part each learner is tuned again with the same inner
budget (settings.nested_cv.trials, 5 inner folds), then predicts the held-out outer fold. The outer
folds give the mean and spread of AUROC and calibration, and the per-fold differences against the
reference model (regression with splines) feed the model-choice rule (train.select_models):
a candidate qualifies if its mean AUROC gain is >= 0.01 and it is better in >= 80% of the folds,
without worse calibration-in-the-large (mean |CITL| at most 0.05 above the reference's).

Output: artifacts/private/nested_cv.json (aggregates; also published in validation.json).
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from . import learners as Lr
from . import metrics as M
from .cohort import stage_cohort
from .config import settings
from .features import design_matrix, monotone_vector


def _summ(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}


def run(base, out_private, quick: bool = False, stages=("admission",), targets_=("good_outcome",)) -> dict:
    cfg = settings()
    nc = cfg["nested_cv"]
    seed = int(cfg["seed"])
    trials = 3 if quick else int(nc["trials"])
    folds, repeats = (3, 1) if quick else (int(nc["folds"]), int(nc["repeats"]))
    min_leaf = int(cfg["privacy"]["min_leaf"])
    out = {"folds": folds, "repeats": repeats, "trials": trials, "reference": Lr.REFERENCE, "stages": {}}
    for stage in stages:
        d = stage_cohort(base, stage)
        X = design_matrix(d, stage)
        out["stages"][stage] = {}
        for target in targets_:
            t0 = time.time()
            y = d[target].to_numpy(float)
            mono = monotone_vector(stage, target)
            per = {name: {"auroc": [], "citl": [], "slope": [], "brier": []} for name in Lr.ORDER}
            rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
            for k, (tr, va) in enumerate(rskf.split(X, y)):
                for name in Lr.ORDER:
                    f = Lr.tune(name, X.iloc[tr], y[tr], seed + 100 + k, trials, mono, min_leaf=min_leaf)
                    p = f.proba(X.iloc[va])
                    s = M.summary(y[va], p)
                    for m in per[name]:
                        per[name][m].append(s[m])
                print(f"  [nested {stage}/{target}] outer fold {k + 1}/{folds * repeats} "
                      f"ref {per[Lr.REFERENCE]['auroc'][-1]:.3f} ({time.time() - t0:.0f}s)", flush=True)
            ref = np.asarray(per[Lr.REFERENCE]["auroc"])
            models = {}
            for name, v in per.items():
                a = np.asarray(v["auroc"])
                delta = a - ref
                models[name] = {"auroc": _summ(a), "brier": _summ(v["brier"]), "citl": _summ(v["citl"]),
                                "slope": _summ(v["slope"]), "abs_citl_mean": float(np.mean(np.abs(v["citl"]))),
                                "delta_mean": float(delta.mean()), "delta_lo": float(np.percentile(delta, 2.5)),
                                "delta_hi": float(np.percentile(delta, 97.5)),
                                "share_better": float((delta > 0).mean()) if name != Lr.REFERENCE else None,
                                "folds_better": int((delta > 0).sum()) if name != Lr.REFERENCE else None,
                                "fold_auroc": [round(float(x), 4) for x in a]}
            out["stages"][stage][target] = {"n": int(len(y)), "events": int(y.sum()), "models": models,
                                            "seconds": round(time.time() - t0)}
    (out_private / "nested_cv.json").write_text(json.dumps({**out, **{k: v for k, v in out["stages"].items()}}, indent=1))
    return out
