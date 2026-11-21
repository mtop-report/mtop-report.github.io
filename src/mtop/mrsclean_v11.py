"""MRS CLEAN for MTOP version 1.1: the re-run of plan section 6.6 (docs/registrations/v1_1_plan.md).

"MRS CLEAN is re-run for version 1.1 under the rules of plan version 2, as a check of the pipeline,
not as evidence. It would be the third look at those synthetic data." This module reuses the second
dry run's code (mrsclean2.py) unchanged on predictions from the version 1.1 app, which carries both
versions: the calculator's model (version 1.1) and v1.0.0's archived reference. Running v1.0.0 from
the same predictions must reproduce the second dry run's results, which checks the pipeline itself.

Two additions for version 1.1:
  * tool check: the app's estimates against an independent Python computation of the deployed
    version 1.1 models (reference.py), with the same averaging of unknown inputs;
  * the missing-value check: what the version 1.1 models give when they use their own missing-value
    handling instead of the averaging (the setting in which v1.0.0's term for where IV thrombolysis
    was given, learned from 4 patients, shifted every thrombolysed patient in the first dry run).

Row-level files stay in data/raw/mrs_clean/ (git-ignored, MRS CLEAN licence: private use only);
results are aggregates in reports/mrs_clean/v11/ (git-ignored until the MR CLEAN team has seen them).

    node scripts/mtop_predict_records.js <site> data/raw/mrs_clean/run/records.json <preds_all.json> all
    node scripts/mtop_predict_records.js <site> data/raw/mrs_clean/run/records.json <preds_ind.json> indicators
    python -m mtop.mrsclean_v11 <preds_all.json> <preds_ind.json>
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import mrsclean, mrsclean2, paths

REP = paths.REPORTS / "mrs_clean" / "v11"
REF = "Logistic regression (splines)"
REF_V100 = "Logistic regression (splines) · v1.0.0"


def with_chosen(preds: dict, name: str) -> dict:
    """The same predictions with `name` as the model under evaluation."""
    out = copy.deepcopy(preds)
    for st in out["meta"].values():
        for tg in st.values():
            tg["chosen"] = name
    return out


def _v11_models() -> dict:
    """The deployed version 1.1 reference models, refitted in Python from their published settings
    (the independent computation of the tool check)."""
    from . import v11
    dec = v11.load_decisions()
    dep = json.loads((v11.PRIV / "deployed.json").read_text())
    U = v11.Universe(v11.load_base(), None)
    W = np.arange(U.n)
    out = {}
    for st in mrsclean2.STAGES:
        for tg in mrsclean2.TARGETS:
            ch = v11.chain_for(dec, tg)
            m = v11.fit_stage(U, st, tg, W, v11.SEED, ch, v11.R.V110, settings=dep[st][tg]["info"]["settings"])
            out[(st, tg)] = v11._with_slope(m, U, st, tg, W, v11.slope_factor(dec, st, tg, ch)).m
    return out


def python_estimates_v11(records: list[dict], stage: str, target: str, model, shift: float):
    """mrsclean2.python_estimates with the version 1.1 model: the same eligibility, the same averaging
    of unknown inputs over the bundle's registry values, paired as the app pairs them."""
    from .config import features
    from .features import column_to_feature, design_matrix
    from .metrics import expit
    bundle = json.loads((paths.ROOT / "artifacts" / "public" / "model_bundle.json").read_text())
    imp = bundle["impute"]
    req_ivt = {f.key for f in features() if f.spec.get("requires_ivt")}
    used = set(column_to_feature(stage).values())
    keys = [f.key for f in features() if f.type != "derived"]
    perms = {k: mrsclean2._perm(len(v["values"]), k) for k, v in imp.items()}
    pos, rows, owner = [], [], []
    for i, x0 in enumerate(records):
        if x0.get("mrs90") is None:
            continue
        if stage != "admission" and x0.get("tici") is None:
            continue
        if stage == "h24" and x0.get("nihss_24h") is None:
            continue
        unk = [k for k in imp if k in used and x0.get(k) is None and not (k in req_ivt and x0.get("ivt") == 0)]
        n = max((len(imp[k]["values"]) for k in unk), default=1)
        for s in range(n):
            x = {k: x0.get(k) for k in keys}
            for k in unk:
                vals = imp[k]["values"]
                x[k] = vals[perms[k][s % len(vals)]]
            if x.get("ivt") == 0:
                x["onset_to_ivt"] = None
                x["ivt_external"] = 0.0
            rows.append(x)
            owner.append(len(pos))
        pos.append(i)
    df = pd.DataFrame(rows)
    for f in features():
        if f.type == "derived" or f.key not in df:
            continue
        if f.type == "categorical":
            df[f.key] = df[f.key].astype(object).where(df[f.key].notna(), None)
        else:
            df[f.key] = pd.to_numeric(df[f.key], errors="coerce").astype(float)
    X = design_matrix(df, stage)
    p = expit(model.logit(X) + shift)
    owner = np.asarray(owner)
    return np.asarray(pos), np.bincount(owner, weights=p) / np.bincount(owner)


def tool_check(preds: dict) -> dict:
    recs = json.loads((mrsclean2.OUT / "run" / "records.json").read_text())
    n = len(recs)
    bundle = json.loads((paths.ROOT / "artifacts" / "public" / "model_bundle.json").read_text())
    models = _v11_models()
    res, worst = {}, 0.0
    for st in mrsclean2.STAGES:
        for tg in mrsclean2.TARGETS:
            idx, p = mrsclean2._chosen(preds, st, tg, n)
            shift = bundle["models"][st][tg]["learners"][REF]["intercept_recent"]
            pos, q = python_estimates_v11(recs, st, tg, models[(st, tg)], shift)
            assert np.array_equal(pos, idx), f"{st}/{tg}: eligible patients differ"
            d = float(np.max(np.abs(p - q)))
            worst = max(worst, d)
            res[f"{st}/{tg}"] = {"n": len(idx), "max_abs_difference": d}
    return {"stages": res, "max_abs_difference": worst, "passes": bool(worst <= mrsclean2.TOOL_TOLERANCE)}


def missing_value_check(preds_all: dict, preds_ind: dict) -> dict:
    """Mean estimate in thrombolysed and other patients: the models' own missing-value handling
    ('indicators') against the app's averaging of unknown inputs, for version 1.1."""
    recs = json.loads((mrsclean2.OUT / "run" / "records.json").read_text())
    n = len(recs)
    ivt = np.array([r.get("ivt") for r in recs], dtype=object)
    bundle = json.loads((paths.ROOT / "artifacts" / "public" / "model_bundle.json").read_text())
    out = {}
    for st in mrsclean2.STAGES:
        for tg in mrsclean2.TARGETS:
            idx, pa = mrsclean2._chosen(preds_all, st, tg, n)
            idx2, pi = mrsclean2._chosen(preds_ind, st, tg, n)
            assert np.array_equal(idx, idx2)
            sub = ivt[idx]
            spec = bundle["models"][st][tg]["learners"][REF]["spec"]
            terms = sorted({spec["columns"][j].split("__")[0] for j in spec["indicators"]})
            out[f"{st}/{tg}"] = {
                "averaging": {"ivt": float(pa[sub == 1.0].mean()), "no_ivt": float(pa[sub == 0.0].mean())},
                "own_missing_handling": {"ivt": float(pi[sub == 1.0].mean()), "no_ivt": float(pi[sub == 0.0].mean())},
                "max_abs_difference": float(np.max(np.abs(pa - pi))),
                "missing_value_terms": terms,
                "term_for_where_ivt_was_given": "ivt_external" in terms}
    return out


def summary_line(r: dict) -> dict:
    e = {}
    for part in ("primary", "secondary"):
        for st, per in r[part].items():
            for tg, v in per.items():
                e[f"{st}/{tg}"] = {k: v.get(k) for k in ("model", "n", "auroc", "citl", "slope", "oe", "mbc", "loss_beyond_case_mix",
                                                         "update_needed", "update_enough")}
                e[f"{st}/{tg}"]["auroc_ci"] = v["ci"]["auroc"]
                e[f"{st}/{tg}"]["slope_after_update"] = ((v.get("recalibration") or {}).get("check_after") or {}).get("slope")
                if "vs_comparator" in v:
                    c = v["vs_comparator"]
                    e[f"{st}/{tg}"]["vs_comparator"] = {k: c[k] for k in ("comparator", "mtop_auroc", "comparator_auroc",
                                                                             "difference", "difference_ci", "verdict")}
    return e


def run(preds_all_path: Path, preds_ind_path: Path) -> dict:
    REP.mkdir(parents=True, exist_ok=True)
    preds = mrsclean.load_predictions(preds_all_path)
    preds_ind = mrsclean.load_predictions(preds_ind_path)
    assert all(preds["meta"][st][tg]["chosen"] == REF and REF_V100 in preds["meta"][st][tg]["learners"]
               for st in preds["meta"] for tg in preds["meta"][st]), "predictions must come from the version 1.1 app"
    res = {"at": mrsclean2.now(), "predictions_sha256": mrsclean2.sha256(preds_all_path),
           "indicators_sha256": mrsclean2.sha256(preds_ind_path)}
    res["tool_check"] = tool_check(preds)
    tmp = REP / "_preds_v100.json"
    tmp.write_text(json.dumps(with_chosen(preds, REF_V100)))
    for label, path in (("v1.1", preds_all_path), ("v1.0.0", tmp)):
        full = mrsclean2.main_results(path)
        (REP / f"main_{label}.json").write_text(json.dumps(full, indent=1))
        res[label] = summary_line(full)
        sg = mrsclean2.subgroups(path)
        (REP / f"subgroups_{label}.json").write_text(json.dumps(sg, indent=1))
    tmp.unlink()
    res["missing_value_check"] = missing_value_check(preds, preds_ind)
    (REP / "summary.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("preds_all")
    ap.add_argument("preds_indicators")
    a = ap.parse_args(argv)
    r = run(Path(a.preds_all), Path(a.preds_indicators))
    print(json.dumps({k: r[k] for k in ("tool_check",)}, indent=1))
    for lab in ("v1.0.0", "v1.1"):
        for k, v in r[lab].items():
            c = v.get("vs_comparator") or {}
            print(f"{lab:7s} {k:22s} AUROC {v['auroc']:.3f}  O/E {v['oe']:.2f}  slope {v['slope']:.2f}  "
                  f"vs {c.get('comparator', '-')}: {c.get('difference', float('nan')):+.3f} {c.get('verdict', '')}")


if __name__ == "__main__":
    main()
