"""Export the deployed models and the validation results for the browser.

model_bundle.json  — everything the calculator needs: UI metadata from config/, the registry values
                     used to average over unknown inputs, and for each stage x target every
                     publishable learner (regression with splines, linear regression, LightGBM,
                     XGBoost) with its recent-era intercept, plus which one the pre-specified rule chose.
validation.json    — aggregate validation results for both temporal splits, nested cross-validation,
                     the model choice and the deployed models' summaries (no row-level data).

Privacy: tree leaves summarise >= privacy.min_leaf patients (checked here); every published count
is >= privacy.min_cell or suppressed; training-data feature ranges are not exported.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math

import numpy as np
import pandas as pd

from . import __version__
from . import learners as Lr
from .cohort import flow, stage_cohort
from .config import features, future_columns, settings, stage_ids, targets
from .features import column_to_feature, design_matrix, feature_columns
from .paths import PRIVATE, PROCESSED, PUBLIC
from .train import load_deployed, load_results, subgroup_auroc_ci


def clean_json(o):
    """NaN and infinities become null (JSON has no NaN); numpy scalars become Python numbers."""
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def rounded(o, digits: int = 5):
    """Floats rounded for the aggregate results files (a p-value below 10^-digits becomes 0)."""
    if isinstance(o, dict):
        return {k: rounded(v, digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rounded(v, digits) for v in o]
    if isinstance(o, float):
        return round(o, digits)
    return o


def _ui_features() -> list[dict]:
    out = []
    for f in features():
        spec = copy.deepcopy(f.spec)
        spec["columns"] = f.columns
        out.append(spec)
    return out


IMPUTE_K = 20


def imputation_values(base, k: int = IMPUTE_K) -> dict:
    """Registry values used to average over an input the user marks as unknown.

    Numeric: k quantiles at (i + 0.5) / k among patients treated from 2016 (when ASPECTS
    recording began). Categorical: k category values in proportion to their frequency.
    Aggregates only; no minima or maxima.
    """
    recent = base[base["year"] >= 2016]
    out = {}
    for f in features():
        if not f.spec.get("optional"):
            continue
        s = recent[f.key].dropna()
        if f.type == "numeric":
            q = np.quantile(s.to_numpy(float), (np.arange(k) + 0.5) / k)
            step = float(f.spec.get("step", 1))
            vals = [float(round(round(v / step) * step, 4)) for v in q]
            out[f.key] = {"type": "numeric", "values": vals}
        elif f.type in ("categorical", "binary"):
            freq = s.astype(str if f.type == "categorical" else float).value_counts(normalize=True)
            counts = np.floor(freq.to_numpy() * k).astype(int)
            while counts.sum() < k:
                counts[np.argmax(freq.to_numpy() * k - counts)] += 1
            vals = [(str(c) if f.type == "categorical" else float(c)) for c, n in zip(freq.index, counts) for _ in range(n)]
            out[f.key] = {"type": f.type, "values": vals}
    return out


FUTURE_GROUPS = {"perfusion": ["ctp_core_ml", "ctp_tmax6_ml"], "collaterals": ["collaterals_tan"],
                 "gfap": ["gfap_pg_ml", "gfap_hours"]}


def future_summary(base: pd.DataFrame, min_cell: int) -> dict:
    """Predictors collected for future versions (config/future.yaml): the column descriptions for the
    partner template and the calculator preview, and how many registry patients with a known 90-day
    outcome have each group recorded, by outcome (the 100 + 100 rule of amendment 1). Counts of
    1 to min_cell - 1 are published as null (suppressed); a group the export lacks is null."""
    spec = future_columns()
    cols = [{k: v for k, v in c.items() if k != "raw"} for c in spec["columns"]]

    def count(mask) -> int | None:
        n = int(mask.sum())
        return n if n == 0 or n >= min_cell else None

    fno = {}
    for g, keys in FUTURE_GROUPS.items():
        if not all(k in base.columns for k in keys):
            fno[g] = None
            continue
        has = base[keys].notna().all(axis=1)
        good, died = base["good_outcome"] == 1, base["death90"] == 1
        fno[g] = {"n": count(has), "good": count(has & good), "poor": count(has & ~good),
                  "died": count(has & died), "survived": count(has & ~died)}
    return {"min_per_outcome": int(spec.get("min_patients_per_outcome", 100)), "columns": cols,
            "groups": FUTURE_GROUPS, "fno": fno}


def share_stack_bases(learners: dict) -> dict:
    """A stacking base identical to a published learner (the same settings refitted on the same
    patients, as the reference and XGBoost are) is published once, as {"same_as": name}; the browser
    links it back (web/app.js). The base never uses the learner's recent-era intercept."""
    strip = lambda d: json.dumps({k: v for k, v in d.items() if k != "intercept_recent"}, sort_keys=True)  # noqa: E731
    for L in learners.values():
        if L.get("kind") != "stack":
            continue
        for i, (b, name) in enumerate(zip(L["bases"], L.get("base_names", []))):
            other = learners.get(name)
            if other is not None and other is not L and strip(other) == strip(b):
                L["bases"][i] = {"same_as": name}
    return learners


def export_all() -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    cfg = settings()
    min_leaf = int(cfg["privacy"]["min_leaf"])
    results = load_results(PRIVATE)
    dep = json.loads((PRIVATE / "deployment.json").read_text())
    nested_path = PRIVATE / "nested_cv.json"
    nested = json.loads(nested_path.read_text()) if nested_path.exists() else None
    base, _ = flow(pd.read_parquet(PROCESSED / "clean.parquet"))

    bundle = {
        "app": "MTOP",
        "version": __version__,
        "generated": dt.date.today().isoformat(),
        "stages": sorted(cfg["stages"], key=lambda s: s["order"]),
        "targets": targets(),
        "features": _ui_features(),
        "recent_era_from_year": cfg["split"]["recent_era_from_year"],
        "reference": Lr.REFERENCE,
        "impute": imputation_values(base),
        "future": future_summary(base, int(cfg["privacy"]["min_cell"])),
        "models": {},
    }
    for stage in stage_ids():
        d = stage_cohort(base, stage)
        X_all = design_matrix(d, stage)
        cols = feature_columns(stage)
        if list(X_all.columns) != cols:
            raise ValueError(f"column mismatch for {stage}")
        # no published coefficient or split may rest on a category of fewer than min_cell patients
        floor = int(cfg["privacy"]["min_cell"])
        sparse = [c for c in cols if "__" in c and 0 < int((X_all[c] == 1).sum()) < floor]
        if sparse:
            raise ValueError(f"{stage}: model categories with fewer than {floor} patients: {sparse}; give the code a "
                             "parent (clean.OCCLUSION_PARENT) or mark the option `model: false` in config/features.yaml")
        c2f = column_to_feature(stage)
        bundle["models"][stage] = {}
        for target in targets():
            pk = load_deployed(PRIVATE, stage, target)
            if pk["columns"] != cols:
                raise ValueError(f"deployed columns differ for {stage}/{target}")
            if pk["chosen"] not in Lr.PUBLISHED:
                raise ValueError(f"{stage}/{target}: the chosen model {pk['chosen']} cannot be reproduced in the browser")
            learners = share_stack_bases({name: Lr.export(m["fitted"], m["shift"], X_all, min_leaf)
                                          for name, m in pk["models"].items() if name in Lr.PUBLISHED})
            bundle["models"][stage][target] = {
                "chosen": pk["chosen"], "columns": cols, "column_feature": [c2f[c] for c in cols],
                "n_train": int(len(d)), "learners": learners,
            }

    # bootstrap CIs for the subgroup AUROCs, from the saved test predictions
    min_cell = int(cfg["privacy"]["min_cell"])
    for label, r in results.items():
        f = PRIVATE / f"predictions_{label}.parquet"
        if not f.exists():
            continue
        cis = subgroup_auroc_ci(base, pd.read_parquet(f), min_cell, n_boot=1000, seed=int(cfg["seed"]))
        for stage, sres in r["stages"].items():
            for target, tres in sres["targets"].items():
                for model, rows in tres.get("subgroups", {}).items():
                    c = cis.get((stage, target, model), {})
                    for row in rows:
                        key = (row.get("dim"), row.get("group"))
                        if "auroc" in row and key in c:
                            row["auroc_ci"] = c[key]

    sel_cfg = cfg["selection"]
    validation = {
        "generated": bundle["generated"], "version": __version__,
        "meta": results["primary"]["meta"] if "primary" in results else {},
        # candidates for the model choice (published) and learners validated for comparison only
        "groups": {"candidates": list(sel_cfg["deployable"]), "comparison": list(sel_cfg.get("comparison", ["MLP"])),
                   "multiplicity": sel_cfg.get("multiplicity"), "alpha": sel_cfg.get("alpha", 0.05),
                   "amendment": "docs/amendment_1.md"},
        "selection": dep["selection"], "nested": nested,
        "splits": {label: {"test_from_year": r["test_from_year"], "stages": r["stages"]} for label, r in results.items()},
        "deployed": dep["deployed"],
    }
    canon = json.dumps(bundle["models"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    bundle["fingerprint"] = hashlib.sha256(canon).hexdigest()
    for name, obj in (("model_bundle.json", clean_json(bundle)), ("validation.json", rounded(clean_json(validation)))):
        with open(PUBLIC / name, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    print(f"export: model_bundle.json {(PUBLIC / 'model_bundle.json').stat().st_size / 1e6:.2f} MB, "
          f"validation.json {(PUBLIC / 'validation.json').stat().st_size / 1e6:.2f} MB")
