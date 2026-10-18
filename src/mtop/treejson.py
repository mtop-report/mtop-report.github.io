"""Tree ensembles as compact arrays for the browser (web/model.js), and a reference evaluator.

Pre-order arrays per tree: f feature (-1 leaf), t threshold, l/r children, d default-left (where a
missing value goes), m missing type (0 None, 1 Zero, 2 NaN), v leaf value, c node weight (LightGBM:
patient count; XGBoost: hessian cover, which its TreeSHAP uses). LightGBM goes left when
x <= t; XGBoost when float32(x) < t (`op: "lt"`).
"""
from __future__ import annotations

import json

import numpy as np

MISSING = {"None": 0, "Zero": 1, "NaN": 2}


def _sig(x: float, digits: int = 10) -> float:
    return float(f"{x:.{digits}g}")


# ------------------------------------------------------------------------------------- LightGBM
def flatten_lgbm(node: dict) -> dict:
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


def lgbm_to_json(booster) -> dict:
    dump = booster.dump_model()
    obj = dump["objective"]
    sigmoid = 1.0
    if "sigmoid:" in obj:
        sigmoid = float(obj.split("sigmoid:")[1].split()[0])
    trees = [flatten_lgbm(t["tree_structure"]) for t in dump["tree_info"]]
    return {"feature_names": dump["feature_names"], "sigmoid": sigmoid, "trees": trees}


def min_leaf_count(model_json: dict) -> int:
    """Smallest leaf among trees with at least one split (single-leaf trees summarise everyone)."""
    counts = [c for t in model_json["trees"] if len(t["f"]) > 1 for f, c in zip(t["f"], t["c"]) if f == -1]
    return min(counts) if counts else 10**9


# -------------------------------------------------------------------------------------- XGBoost
def flatten_xgb(node: dict, fidx: dict) -> dict:
    arr = {k: [] for k in "ftlrdmvc"}

    def visit(n):
        i = len(arr["f"])
        for k in arr:
            arr[k].append(0)
        arr["c"][i] = _sig(float(n.get("cover", 0.0)), 12)
        if "leaf" in n:
            arr["f"][i] = -1
            arr["l"][i] = arr["r"][i] = -1
            arr["v"][i] = float(np.float32(n["leaf"]))
            return i
        kids = {k["nodeid"]: k for k in n["children"]}
        arr["f"][i] = int(fidx[n["split"]])
        arr["t"][i] = float(np.float32(n["split_condition"]))
        arr["d"][i] = 1 if n["missing"] == n["yes"] else 0
        arr["m"][i] = MISSING["NaN"]
        arr["l"][i] = visit(kids[n["yes"]])
        arr["r"][i] = visit(kids[n["no"]])
        return i

    visit(node)
    return arr


def xgb_to_json(booster, columns: list[str], X_train) -> dict:
    """XGBoost trees plus the constant margin (base score), checked against booster predictions."""
    import xgboost as xgb

    fidx = {c: i for i, c in enumerate(columns)}
    fidx.update({f"f{i}": i for i in range(len(columns))})
    trees = [flatten_xgb(json.loads(s), fidx) for s in booster.get_dump(dump_format="json", with_stats=True)]
    spec = {"feature_names": list(columns), "trees": trees, "op": "lt", "base_margin": 0.0}
    X = np.asarray(X_train, float)[:300]
    margin = booster.predict(xgb.DMatrix(X, missing=np.nan, feature_names=list(columns)), output_margin=True)
    leaves = predict_arrays(spec, X)
    diff = margin - leaves
    if float(np.std(diff)) > 1e-4:
        raise ValueError("XGBoost export: browser trees do not reproduce the booster's margin")
    spec["base_margin"] = float(np.mean(diff))
    return spec


def leaf_counts(booster, X_train) -> int:
    """Smallest number of training patients in any XGBoost leaf (trees with a split)."""
    import xgboost as xgb

    idx = booster.predict(xgb.DMatrix(X_train, missing=np.nan), pred_leaf=True)
    idx = np.asarray(idx).reshape(len(X_train), -1)
    smallest = 10**9
    for j in range(idx.shape[1]):
        _, cnt = np.unique(idx[:, j], return_counts=True)
        if len(cnt) > 1:
            smallest = min(smallest, int(cnt.min()))
    return smallest


# ---------------------------------------------------------------------------- reference evaluator
def _decide(t, n, v, lt: bool):
    if lt:
        if np.isnan(v):
            return t["l"][n] if t["d"][n] else t["r"][n]
        return t["l"][n] if np.float32(v) < np.float32(t["t"][n]) else t["r"][n]
    mt = t["m"][n]
    if np.isnan(v) and mt != 2:
        v = 0.0
    if (mt == 1 and abs(v) <= 1e-35) or (mt == 2 and np.isnan(v)):
        return t["l"][n] if t["d"][n] else t["r"][n]
    return t["l"][n] if v <= t["t"][n] else t["r"][n]


def predict_arrays(spec: dict, X: np.ndarray) -> np.ndarray:
    """Sum of leaf values per row (no base margin), as the browser computes it."""
    lt = spec.get("op") == "lt"
    X = np.asarray(X, float)
    out = np.zeros(len(X))
    for i, x in enumerate(X):
        s = 0.0
        for t in spec["trees"]:
            n = 0
            while t["f"][n] >= 0:
                n = _decide(t, n, x[t["f"][n]], lt)
            s += t["v"][n]
        out[i] = s
    return out
