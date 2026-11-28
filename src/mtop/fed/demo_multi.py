"""Federated network demonstration with four centres from three different sources.

Plan: docs/registrations/v2_imaging_plan.md, amendment 1. A software test, not a validation.

  Ostrava      FNO registry (real patients, all MTOP predictors)
  MRCLEAN_A    synthetic MR CLEAN patients (MRS CLEAN), a random sample of 1,200
  MRCLEAN_B    another, disjoint random sample of 1,200
  ISLES24      the ISLES'24 training set (Munich and Zurich), few clinical predictors,
               perfusion volumes from the extractor (src/mtop/isles.py)

A fifth group, a third disjoint MR CLEAN sample of 1,200 (amendment 2), never joins the network:
it plays a new hospital that downloads the finished network model and tests it at home. It comes
from the same synthetic population as two of the training centres, so it checks the pipeline, not
how the model transfers.

The centres differ in country, size, case mix and which predictors they record, as real partners
will. Each centre's rows are written in the MTOP upload template and read by its own site node.
The analyses are those of the original demonstration (fed/demo.py): inside every centre, 5-fold
cross-validation; each centre scores its own held-out patients with a model trained on its own
patients (alone), the network model (network) and the network model adapted locally (adapted); the
regressions alone and in the network; leave-one-centre-out validation; centre size; equivalence
with pooled training. A stage or outcome is run only with the centres that have at least
min_site_patients eligible patients and at least min_cell patients in each outcome group; it is
skipped when fewer than three centres qualify. Groups below min_cell are suppressed in the output.

Licences: MRS CLEAN may be used privately only, and its results go to the MR CLEAN team before any
other use; ISLES'24 is CC BY-NC-SA 4.0. Row-level files stay in data/processed/federated/multi/
(git-ignored). Output: artifacts/public/federated_multi.json (aggregates only), which the app does
not show until the MR CLEAN team has seen it.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..cohort import flow
from ..config import settings
from . import PROTOCOL
from .bins import fed_settings
from .coordinator import Coordinator, LocalTransport
from .demo import (ADAPT_TREES, MODELS, OUTCOMES, STAGE, _r, compare, compare_regression, equivalence, http_run,
                   leave_one_out, size_effect)
from .gbdt import GBDTParams
from .schema import read_site_csv, template_columns, write_site_csv
from .site import SiteNode

log = logging.getLogger("mtop.fed")

MRS_PER_CENTRE = 1200
SEED_SAMPLE = 2026
SIZES = (50, 100)                 # ISLES'24 has about 120 training patients per fold
STAGES = (("admission", "good_outcome"), ("admission", "death90"), ("procedure", "good_outcome"),
          ("h24", "good_outcome"), ("d7", "good_outcome"))
LABELS = {"Ostrava": "Ostrava (FNO registry)", "MRCLEAN_A": "MR CLEAN synthetic A",
          "MRCLEAN_B": "MR CLEAN synthetic B", "ISLES24": "Munich–Zurich (ISLES'24)"}
OUT_DIR = paths.PROCESSED / "federated" / "multi"
VALIDATION = "MRCLEAN_VAL"
DEFAULT_MRS = paths.RAW / "mrs_clean" / "template_primary.csv"
DEFAULT_ISLES = paths.RAW / "isles24" / "isles24_site.csv"


# --------------------------------------------------------------------------- centres
def source_frames(mrs_path: Path = DEFAULT_MRS, isles_path: Path = DEFAULT_ISLES) -> dict[str, pd.DataFrame]:
    """Template rows (raw strings) of the four centres. Ostrava is written from the cleaned registry
    the way a partner would export it; MR CLEAN is split into two disjoint random samples."""
    clean = pd.read_parquet(paths.PROCESSED / "clean.parquet")
    base, _ = flow(clean)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fno = OUT_DIR / "source_ostrava.csv"
    write_site_csv(base.drop(columns=[c for c in ("year",) if c in base.columns]), fno)
    frames = {"Ostrava": pd.read_csv(fno, dtype=str, keep_default_na=False)}
    mrs = pd.read_csv(mrs_path, dtype=str, keep_default_na=False)
    frames.update(split_mrs(mrs))
    frames["ISLES24"] = pd.read_csv(isles_path, dtype=str, keep_default_na=False)
    return frames


def split_mrs(mrs: pd.DataFrame, n: int = MRS_PER_CENTRE, seed: int = SEED_SAMPLE) -> dict[str, pd.DataFrame]:
    """Three disjoint random samples of n rows (fixed seed), drawn without looking at any value:
    two network centres and the held-out validation sample (amendment 2)."""
    if len(mrs) < 3 * n:
        raise ValueError(f"MRS CLEAN has {len(mrs)} rows; three samples of {n} need {3 * n}")
    idx = np.random.default_rng(seed).permutation(len(mrs))
    return {"MRCLEAN_A": mrs.iloc[idx[:n]].reset_index(drop=True),
            "MRCLEAN_B": mrs.iloc[idx[n:2 * n]].reset_index(drop=True),
            VALIDATION: mrs.iloc[idx[2 * n:3 * n]].reset_index(drop=True)}


def make_sites(frames: dict[str, pd.DataFrame], out_dir: Path = OUT_DIR) -> tuple[list[SiteNode], dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("audit_*.jsonl"):
        f.unlink()
    nodes, reports = [], {}
    for cid, raw in frames.items():
        if cid == VALIDATION:
            continue
        path = out_dir / f"centre_{cid}.csv"
        raw.to_csv(path, index=False)
        typed, rep = read_site_csv(path)
        reports[cid] = rep
        nodes.append(SiteNode(cid, typed, label=LABELS.get(cid, cid), audit_path=out_dir / f"audit_{cid}.jsonl"))
    return nodes, reports


def qualifying(sites, stage: str, target: str, k: int) -> list:
    net = fed_settings()["network"]
    keep = []
    for s in sites:
        y = s.cohort(stage, target)[2]
        if len(y) >= int(net["min_site_patients"]) and y.sum() >= k and len(y) - y.sum() >= k:
            keep.append(s)
    return keep


def recorded(df: pd.DataFrame) -> dict:
    """How many template predictors a centre records for at least half of its patients."""
    cols = [c for c in template_columns() if c != "mrs90" and c in df.columns]
    share = {c: float(df[c].notna().mean()) for c in cols}
    return {"predictors": len(cols), "recorded_half_or_more": int(sum(v >= 0.5 for v in share.values())),
            "recorded_any": int(sum(v > 0 for v in share.values()))}


# --------------------------------------------------------------------------- held-out validation
def validation_node(frames: dict, out_dir: Path) -> SiteNode | None:
    if VALIDATION not in frames:
        return None
    path = out_dir / f"centre_{VALIDATION}.csv"
    frames[VALIDATION].to_csv(path, index=False)
    typed, _ = read_site_csv(path)
    return SiteNode(VALIDATION, typed, label="MR CLEAN synthetic, held-out validation sample")


def external_check(sites, val: SiteNode, co, target: str, p: GBDTParams, n_boot: int, seed: int) -> dict:
    """The finished network models (boosted trees and the regression with splines, trained on all
    patients of all network centres) and each centre's own model, scored at the validation sample
    the way a new hospital would score them at home."""
    from .. import metrics as M
    from .coordinator import PooledNetwork, federated_logreg, fit_gbdt
    from .gbdt import raw_score, sigmoid
    from .logreg import predict_spline

    ids = [s.site_id for s in sites]
    d, B, y = val.cohort(STAGE, target)
    preds = {}
    net, _ = federated_gbdt_all(co, ids, target, p)
    preds["network_trees"] = sigmoid(raw_score(net, B))
    lr, _ = federated_logreg(co, ids, STAGE, target, None)
    preds["network_regression"] = predict_spline(lr, d)
    for s in sites:
        own = fit_gbdt(PooledNetwork.from_sites([s], STAGE, target, None), STAGE, target, p)
        preds[f"alone_{s.site_id}"] = sigmoid(raw_score(own, B))
    ci = M.bootstrap_ci(y, preds, n_boot=n_boot, seed=seed + 50, ref="network_trees")
    out = {"n": int(len(y)), "events": int(y.sum()), "observed": _r(y.mean()), "models": {}}
    for name, pr in preds.items():
        sm = M.summary(y, pr)
        out["models"][name] = {**{k: _r(sm[k]) for k in ("auroc", "brier", "citl", "slope", "mean_pred")},
                               "auroc_ci": [_r(v) for v in ci[name]["auroc"]]}
    return out


def federated_gbdt_all(co, ids, target, p):
    from .coordinator import federated_gbdt

    return federated_gbdt(co, ids, STAGE, target, None, p)


# --------------------------------------------------------------------------- suppression
def _small(g: dict, k: int) -> bool:
    return g["n"] < k or g["events"] < k or g["n"] - g["events"] < k


def suppress(out: dict, k: int) -> dict:
    """Blank every result of a group below k patients or k in either outcome group."""
    small_ids: set = set()

    def blank(g: dict) -> None:
        if _small(g, k):
            small_ids.add((g.get("id"), id(g)))
            for key in list(g):
                if key not in ("id", "trained_on", "n_train"):
                    g[key] = None
            g["suppressed"] = True

    for c in out["centres"]:
        for t in OUTCOMES:
            if c.get(t):
                blank(c[t])
    for res in out["outcomes"].values():
        if "skipped" in res:
            continue
        for g in res["per_centre"] + res["loso"]:
            sid, small = g["id"], _small(g, k)
            blank(g)
            if small:
                for part in ("curves", "growth"):
                    if part in res and sid in res[part]:
                        res[part][sid] = None
                if res.get("size") and sid in res["size"].get("per_centre", {}):
                    res["size"]["per_centre"][sid] = None
        for r in res.get("regression", {}).values():
            for g in r["per_centre"]:
                blank(g)
        if res.get("validation"):
            blank(res["validation"])
    for st in out["stages"]:
        for g in st.get("per_centre", []):
            blank(g)
    return out


def check_public(out: dict, k: int) -> None:
    groups = [c[t] for c in out["centres"] for t in OUTCOMES if c.get(t)]
    for res in out["outcomes"].values():
        if "skipped" in res:
            continue
        groups += res["per_centre"] + res["loso"]
        for r in res.get("regression", {}).values():
            groups += r["per_centre"]
        if res.get("validation"):
            groups.append(res["validation"])
    for st in out["stages"]:
        groups += st.get("per_centre", [])
    for g in groups:
        if not g.get("suppressed"):
            assert not _small(g, k), "small group in federated_multi.json"


# --------------------------------------------------------------------------- main
def run(frames: dict[str, pd.DataFrame] | None = None, http: bool = True, quick: bool = False,
        out_dir: Path = OUT_DIR, public_path: Path | None = None, trees: int | None = None) -> dict:
    cfg = settings()
    seed = int(cfg["seed"])
    k = int(cfg["privacy"]["min_cell"])
    p = GBDTParams.default(**({"trees": trees or 20} if quick or trees else {}))
    folds = 3 if quick else 5
    n_boot = 100 if quick else int(cfg["bootstrap"]["n"])
    reps = 1 if quick else 2
    checks = list(range(0, p.trees + 1, 5))
    frames = frames if frames is not None else source_frames()
    sites, reports = make_sites(frames, out_dir)
    val = validation_node(frames, out_dir)
    t0 = time.time()

    outcomes, fold0, ref_sites = {}, None, None
    for target in OUTCOMES:
        q = qualifying(sites, STAGE, target, k)
        if len(q) < 3:
            outcomes[target] = {"skipped": f"{len(q)} centres qualify"}
            continue
        co = Coordinator(LocalTransport(q), log=log.info)
        res, f0 = compare(q, co, STAGE, target, p, folds, seed, n_boot=n_boot, checks=checks, growth=True)
        if fold0 is None and target == OUTCOMES[0]:
            fold0, ref_sites = f0, q
        res["centres_used"] = [s.site_id for s in q]
        n_min = min(int(len(s.cohort(STAGE, target)[2]) * (folds - 1) / folds) for s in q)
        sizes = tuple(m for m in SIZES if m < n_min)          # a centre cannot give more than it has
        res["size"] = size_effect(q, STAGE, target, p, folds, seed, reps, res, sizes=sizes) if sizes else None
        res["loso"] = leave_one_out(q, co, target, p, n_boot, seed)
        res["regression"] = compare_regression(q, co, STAGE, target, folds, seed)
        if val is not None:
            res["validation"] = external_check(q, val, co, target, p, n_boot, seed)
        outcomes[target] = res
    stages = []
    for st, t in STAGES[2:]:
        q = qualifying(sites, st, t, k)
        if len(q) < 3:
            stages.append({"stage": st, "target": t, "skipped": f"{len(q)} centres qualify",
                           "centres_used": [s.site_id for s in q]})
            continue
        co = Coordinator(LocalTransport(q), log=log.info)
        res, _ = compare(q, co, st, t, p, folds, seed)
        stages.append({"stage": st, "target": t, "mean": res["mean"], "centres_used": [s.site_id for s in q],
                       "per_centre": [{"id": e["id"], "n": e["n"], "events": e["events"],
                                       **{m: e[m]["auroc"] for m in MODELS}} for e in res["per_centre"]]})
    eq = run_stats = None
    if ref_sites is not None:
        co = Coordinator(LocalTransport(ref_sites), log=log.info)
        eq, ref = equivalence(ref_sites, co, p, fold0)
        run_stats = http_run(ref_sites, p, ref, fold0) if http else None

    centres = []
    for s in sites:
        rep = reports[s.site_id]
        c = {"id": s.site_id, "label": s.label, "rows": rep["rows"], "with_outcome": rep["with_outcome"],
             "unrecognised_values": int(sum(rep["unrecognised_values"].values())), **recorded(s.data),
             "perfusion_recorded": int(rep["future_recorded"].get("ctp_core_ml", 0))}
        for t in OUTCOMES:
            y = s.cohort(STAGE, t)[2]
            c[t] = {"n": int(len(y)), "events": int(y.sum()), "rate": _r(y.mean(), 3) if len(y) else None}
        c["eligible"] = {st: int(len(s.cohort(st, t)[2])) for st, t in STAGES if t == "good_outcome"}
        centres.append(c)
    net = fed_settings()["network"]
    out = {
        "generated": time.strftime("%Y-%m-%d"), "protocol": PROTOCOL, "kind": "simulation, software test",
        "plan": "docs/registrations/v2_imaging_plan.md, amendment 1", "stage": STAGE, "quick": quick, "folds": folds,
        "adapt_trees": ADAPT_TREES, "mrs_per_centre": MRS_PER_CENTRE, "sample_seed": SEED_SAMPLE,
        "recipe": {k2: v for k2, v in asdict(p).items() if k2 in ("trees", "learning_rate", "max_depth", "min_child", "reg_lambda")},
        "settings": {"min_sites": int(net["min_sites"]), "min_site_patients": int(net["min_site_patients"]),
                     "min_cell": k},
        "centres": centres, "outcomes": outcomes, "stages": stages, "equivalence": eq, "run": run_stats,
        "seconds_total": _r(time.time() - t0, 1),
    }
    out = suppress(out, k)
    check_public(out, k)
    target_path = public_path or (paths.PUBLIC / "federated_multi.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out
