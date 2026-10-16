"""Export trained models and validation results for the browser.

model_bundle.json  — everything the calculator needs: UI metadata from config/, and for each
                     stage x target the LightGBM trees in a compact array form.
validation.json    — aggregate validation results (no row-level data).

Privacy: trees are grown with min_child_samples >= privacy.min_leaf (default 20), so every
split side and leaf summarises at least that many patients. Training-data feature ranges
(LightGBM's feature_infos) are deliberately not exported.
"""
from __future__ import annotations

import copy
import datetime as dt
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import __version__
from .config import features, settings, stage_ids, targets
from .features import column_to_feature, feature_columns
from .cohort import flow
from .paths import CONFIG, PRIVATE, PROCESSED, PUBLIC

MISSING = {"None": 0, "Zero": 1, "NaN": 2}


def _sig(x: float, digits: int = 10) -> float:
    return float(f"{x:.{digits}g}")


def flatten_tree(node: dict) -> dict:
    """Pre-order arrays: f feature (-1 leaf), t threshold, l/r children, d default-left,
    m missing type (0 None, 1 Zero, 2 NaN), v leaf value, c sample count."""
    arr = {k: [] for k in "ftlrdmvc"}

    def visit(n):
        i = len(arr["f"])
        for k in arr:
            arr[k].append(0)
        if "split_feature" in n:
            if n.get("decision_type", "<=") != "<=":
                raise ValueError("categorical splits are not supported by the browser model")
            arr["f"][i] = int(n["split_feature"])
            arr["t"][i] = float(n["threshold"])
            arr["d"][i] = 1 if n["default_left"] else 0
            arr["m"][i] = MISSING[n["missing_type"]]
            arr["c"][i] = int(n["internal_count"])
            arr["l"][i] = visit(n["left_child"])
            arr["r"][i] = visit(n["right_child"])
        else:
            arr["f"][i] = -1
            arr["l"][i] = arr["r"][i] = -1
            arr["v"][i] = _sig(float(n["leaf_value"]))
            arr["c"][i] = int(n.get("leaf_count", 0))
        return i

    visit(node)
    return arr


def booster_to_json(booster: lgb.Booster) -> dict:
    dump = booster.dump_model()
    obj = dump["objective"]
    sigmoid = 1.0
    if "sigmoid:" in obj:
        sigmoid = float(obj.split("sigmoid:")[1].split()[0])
    trees = [flatten_tree(t["tree_structure"]) for t in dump["tree_info"]]
    return {"feature_names": dump["feature_names"], "sigmoid": sigmoid, "trees": trees}


def min_leaf_count(model_json: dict) -> int:
    return min(c for t in model_json["trees"] for f, c in zip(t["f"], t["c"]) if f == -1 and len(t["f"]) > 1)


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
        elif f.type == "categorical":
            freq = s.value_counts(normalize=True)
            counts = np.floor(freq.to_numpy() * k).astype(int)
            while counts.sum() < k:
                counts[np.argmax(freq.to_numpy() * k - counts)] += 1
            vals = [str(c) for c, n in zip(freq.index, counts) for _ in range(n)]
            out[f.key] = {"type": "categorical", "values": vals}
    return out


def export_all() -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE / "train_results.json") as fh:
        res = json.load(fh)
    cfg = settings()
    min_leaf = int(cfg["privacy"]["min_leaf"])

    bundle = {
        "app": "MTOP",
        "version": __version__,
        "generated": dt.date.today().isoformat(),
        "stages": sorted(cfg["stages"], key=lambda s: s["order"]),
        "targets": targets(),
        "features": _ui_features(),
        "recent_era_from_year": cfg["split"]["recent_era_from_year"],
        "impute": {},
        "models": {},
    }
    clean_file = PROCESSED / "clean.parquet"
    if clean_file.exists():
        base, _ = flow(pd.read_parquet(clean_file))
        bundle["impute"] = imputation_values(base)
    validation = {"generated": bundle["generated"], "version": __version__, "meta": res["meta"], "stages": {}}

    for stage in stage_ids():
        sres = res["stages"][stage]
        bundle["models"][stage] = {}
        validation["stages"][stage] = {k: sres[k] for k in ("n_all", "n_dev", "n_test")}
        validation["stages"][stage]["targets"] = {}
        for target in targets():
            tres = sres["targets"][target]
            booster = lgb.Booster(model_file=str(PRIVATE / f"lgbm_{stage}_{target}.txt"))
            mj = booster_to_json(booster)
            cols = feature_columns(stage)
            if mj["feature_names"] != cols:
                raise ValueError(f"column mismatch for {stage}/{target}")
            ml = min_leaf_count(mj)
            if ml < min_leaf:
                raise ValueError(f"{stage}/{target}: a leaf summarises only {ml} patients (< {min_leaf})")
            c2f = column_to_feature(stage)
            bundle["models"][stage][target] = {
                "columns": cols,
                "column_feature": [c2f[c] for c in cols],
                "sigmoid": mj["sigmoid"],
                "intercept_recent": round(float(tres["deployed"]["intercept_recent"]), 6),
                "n_train": int(sres["n_all"]),
                "min_leaf_count": int(ml),
                "trees": mj["trees"],
            }
            v = {k: tres[k] for k in ("internal", "temporal", "curves", "recalibration", "explain")}
            v["deployed"] = tres["deployed"]
            v["params"] = tres["params"]
            v["rounds_dev"] = tres["rounds_dev"]
            validation["stages"][stage]["targets"][target] = v

    with open(PUBLIC / "model_bundle.json", "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, ensure_ascii=False, separators=(",", ":"))
    with open(PUBLIC / "validation.json", "w", encoding="utf-8") as fh:
        json.dump(validation, fh, ensure_ascii=False, separators=(",", ":"))
    size = (PUBLIC / "model_bundle.json").stat().st_size / 1e6
    print(f"export: model_bundle.json {size:.2f} MB, validation.json "
          f"{(PUBLIC / 'validation.json').stat().st_size / 1e3:.0f} kB")
    _ = CONFIG  # config files are embedded via _ui_features()
