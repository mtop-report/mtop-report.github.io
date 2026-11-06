"""Second MRS CLEAN dry run, under analysis plan version 2 (6 November 2026).

Plan version 2 was written after the first run (mrsclean.py, plan version 1) and frozen before this
one; the SHA-256 of its files is in docs/registrations/mrs_clean_dry_run_plan_v2.sha256. This module
adds what version 2 asks for: the tool check against an independent computation (step 1), the coding
check against the authors' table of the real patients (step 2), the observed/expected ratio in the
fidelity check (step 3), the expected-AUROC references (step 5), the rules of section 6, the
recalibration of step 7, subgroups with the paired difference (step 8), the new sensitivity analyses
(step 9) and the secondary results (step 10).

The MRS CLEAN licence allows private use only: every row-level file stays in data/raw/mrs_clean/
(git-ignored); results are aggregates in reports/mrs_clean/v2/ (git-ignored too, until the MR CLEAN
team has seen them).

    python -m mtop.mrsclean2 convert SOURCE.csv     templates incl. the codebook occlusion variant
    python -m mtop.mrsclean2 records                records (with and without the outcome) and variants
    python -m mtop.mrsclean2 toolcheck PREDS        step 1; reads no outcome
    python -m mtop.mrsclean2 ui-sample              1,000 patients with random outcomes (browser check)
    python -m mtop.mrsclean2 ui-compare UI PREDS    the browser's AUROCs on that sample against PREDS
    python -m mtop.mrsclean2 codingcheck SOURCE.csv step 2; reads no outcome
    python -m mtop.mrsclean2 fidelity               step 3
    python -m mtop.mrsclean2 casemix                step 4
    python -m mtop.mrsclean2 reference              step 5a (FNO data, local only)
    python -m mtop.mrsclean2 main PREDS             steps 5b, 6, 7 and 10
    python -m mtop.mrsclean2 subgroups PREDS        step 8
    python -m mtop.mrsclean2 sensitivity PREDS --variant NAME FILE ...   step 9
Add --random-outcomes SEED to main, subgroups or sensitivity to test the code on random outcomes
before the real outcomes are analysed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import mrpredicts, mrsclean, paths

OUT = paths.RAW / "mrs_clean"
REP = paths.REPORTS / "mrs_clean" / "v2"
SEED, N_BOOT = 2026, 2000
STAGES = ("admission", "procedure", "h24")
PRIMARY = ("admission", "h24")
TARGETS = ("good_outcome", "death90")
COMPARATOR = mrsclean.COMPARATOR

# ---- section 6 of plan version 2
TOOL_TOLERANCE = 0.01          # 1 percentage point, any patient
UI_TOLERANCE = 0.001           # AUROC, browser against this run
SHARE_TOL, SD_TOL = 5.0, 0.1   # coding check: percentage points; standard deviations
C_TOLERANCE, SLOPE_RANGE, OE_TOLERANCE = 0.03, (0.8, 1.2), 0.1
PUBLISHED_OE = {("mr_predicts", "good_outcome"): 40.8 / 42.7}   # HERMES intercept, Venema et al. 2021
MARGIN = 0.01
MBC_GAP = 0.05
OE_RANGE = (0.9, 1.1)
ICI_REFIT = 0.03


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name: str, obj: dict) -> Path:
    REP.mkdir(parents=True, exist_ok=True)
    f = REP / name
    f.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    return f


# --------------------------------------------------------------------------- conversion and records
VARIANTS = dict(mrsclean.VARIANTS)


def convert_all(source: Path) -> dict:
    """The first run's templates plus the occlusion codes read as the codebook labels them."""
    mrsclean._check_ignored(OUT)
    src = mrsclean.load(source)
    out = {}
    for name, (t2c, smk, anti) in VARIANTS.items():
        t, side = mrsclean.convert(src, tici_2c=t2c, smoker=smk, antithrombotic_none=anti)
        t.to_csv(OUT / f"template_{name}.csv", index=False)
        out[name] = sha256(OUT / f"template_{name}.csv")
        if name == "primary":
            side.to_csv(OUT / "comparators.csv", index=False)
    t, side = mrsclean.convert(src, occlusion="codebook")
    t.to_csv(OUT / "template_occlusion_codebook.csv", index=False)
    side.to_csv(OUT / "comparators_occlusion_codebook.csv", index=False)
    out["occlusion_codebook"] = sha256(OUT / "template_occlusion_codebook.csv")
    return {"source_sha256": sha256(source), "rows": len(src), "templates_sha256": out, "at": now()}


def write_records_all() -> dict:
    """Records for the JavaScript predictor: the primary file without its outcome (step 1), every
    template variant, and the primary records with where IVT was given set to the treating centre."""
    files = {}
    for name in list(VARIANTS) + ["occlusion_codebook"]:
        files[name] = str(mrsclean.write_records(OUT, name))
    recs = json.loads((OUT / "run" / "records.json").read_text())
    blind = [{**x, "mrs90": None} for x in recs]
    (OUT / "run" / "records_blind.json").write_text(json.dumps(blind))
    files["blind"] = str(OUT / "run" / "records_blind.json")
    files["ivt_external_centre"] = str(mrsclean.posthoc_ivt_external_records(OUT))
    return files


# --------------------------------------------------------------------------- step 1: tool check
def _perm(n: int, key: str) -> list[int]:
    """The app's pairing of averaged values (web/app.js perm), reproduced exactly."""
    s = 7
    for ch in key:
        s = (s * 31 + ord(ch)) % 2147483647
    a = list(range(n))
    for i in range(n - 1, 0, -1):
        s = (s * 16807) % 2147483647
        j = s % (i + 1)
        a[i], a[j] = a[j], a[i]
    return a


def model_based_concordance(p) -> float:
    """van Klaveren et al. (Stat Med 2016): the concordance expected if the estimates p were the true
    risks, over all pairs of patients (ties count one half). Needs no outcomes."""
    p = np.sort(np.asarray(p, float))
    vals, counts = np.unique(p, return_counts=True)
    qsum_group = counts * (1 - vals)
    q_less = np.concatenate([[0.0], np.cumsum(qsum_group)[:-1]])
    num = np.sum(counts * vals * (q_less + 0.5 * (counts - 1) * (1 - vals)))
    den = np.sum(p * (np.sum(1 - p) - (1 - p)))
    return float(num / den)


def python_estimates(records: list[dict], stage: str, target: str) -> tuple[np.ndarray, np.ndarray]:
    """Independent computation for step 1: the fitted Python model from which the app's calculator
    model was exported (artifacts/private), its own feature construction (features.design_matrix),
    and the bundle's averaging values with the app's pairing. Returns (positions, estimates) for the
    patients eligible at the stage (as the app: final mTICI from the End of procedure, 24-hour NIHSS
    at 24 hours)."""
    from .config import features
    from .features import column_to_feature, design_matrix
    from .train import load_deployed
    bundle = json.loads((paths.PUBLIC / "model_bundle.json").read_text())
    imp = bundle["impute"]
    req_ivt = {f.key for f in features() if f.spec.get("requires_ivt")}
    dep = load_deployed(paths.PRIVATE, stage, target)
    m = dep["models"][dep["chosen"]]
    fitted, shift = m["fitted"], m["shift"]
    used = set(column_to_feature(stage).values())
    keys = [f.key for f in features() if f.type != "derived"]
    perms = {k: _perm(len(v["values"]), k) for k, v in imp.items()}
    pos, rows, owner = [], [], []
    for i, x0 in enumerate(records):
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
            if x.get("ivt") == 0:                 # web/model.js buildRow; config/features.yaml
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
    from .metrics import expit
    p = expit(fitted.logit(X) + shift)
    owner = np.asarray(owner)
    est = np.bincount(owner, weights=p) / np.bincount(owner)
    return np.asarray(pos), est


def fast_boot_auc(y, scores: dict, n_boot: int = N_BOOT, seed: int = SEED, ref: str | None = None) -> dict:
    """Percentile bootstrap intervals for AUROCs (and their paired differences against `ref`) on the same
    resamples. Each resample is drawn as patient counts (rng.integers then bincount, the same
    distribution as resampling rows); each AUROC is then a weighted Mann-Whitney statistic over the
    scores sorted once, so no resample needs sorting."""
    y = np.asarray(y, float)
    n = len(y)
    prep = {}
    for k, sc in scores.items():
        sc = np.asarray(sc, float)
        order = np.argsort(sc, kind="mergesort")
        _, g = np.unique(sc[order], return_inverse=True)
        prep[k] = (order, g, int(g.max()) + 1)
    rng = np.random.default_rng(seed)
    out = {k: [] for k in scores}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        w1, w0 = w * y, w * (1 - y)
        if w1.sum() == 0 or w0.sum() == 0:
            continue
        for k, (order, g, ng) in prep.items():
            G1 = np.bincount(g, w1[order], ng)
            G0 = np.bincount(g, w0[order], ng)
            below = np.cumsum(G0) - G0
            out[k].append(float(np.sum(G1 * (below + 0.5 * G0)) / (G1.sum() * G0.sum())))
    res = {k: {"auroc": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in out.items()}
    if ref is not None:
        for k in scores:
            if k != ref:
                d = np.asarray(out[k]) - np.asarray(out[ref])
                res[k]["delta_auroc_vs_ref"] = {"ref": ref, "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return res


def _chosen(preds: dict, st: str, tg: str, n: int):
    return mrsclean._chosen_p(preds, st, tg, n)


def tool_check(preds_blind: Path) -> dict:
    """Step 1, outcome-blind: absent and often-unknown inputs, the app's estimates against the
    independent computation, the mean estimate overall and by IV thrombolysis, and the model-based
    concordance."""
    from .config import features
    preds = mrsclean.load_predictions(preds_blind)
    recs = json.loads((OUT / "run" / "records_blind.json").read_text())
    assert all(r["mrs90"] is None for r in recs), "the blind records must not hold the outcome"
    n = len(recs)
    from .fed.schema import read_site_csv
    _, rep = read_site_csv(OUT / "template_primary.csv")
    keys = [f.key for f in features() if f.type != "derived"]
    absent = [k for k in keys if k in rep["columns_missing"]]
    unknown = {k: float(np.mean([r.get(k) is None for r in recs])) for k in keys if k not in absent}
    often = {k: round(100 * v, 2) for k, v in unknown.items() if v >= 0.05}
    some = {k: int(round(v * n)) for k, v in unknown.items() if 0 < v < 0.05}
    ivt = np.array([r.get("ivt") for r in recs], dtype=object)
    res = {"at": now(), "n": n, "absent_inputs": absent, "unknown_5pct_or_more": often,
           "unknown_under_5pct_counts": some, "stages": {}}
    worst = 0.0
    for st in STAGES:
        for tg in TARGETS:
            idx, p = _chosen(preds, st, tg, n)
            pos, q = python_estimates(recs, st, tg)
            assert np.array_equal(pos, idx), f"{st}/{tg}: eligible patients differ"
            d = np.abs(p - q)
            worst = max(worst, float(d.max()))
            sub = ivt[idx]
            res["stages"][f"{st}/{tg}"] = {
                "n": len(idx), "model": preds["meta"][st][tg]["chosen"],
                "max_abs_difference": float(d.max()), "n_over_tolerance": int((d > TOOL_TOLERANCE).sum()),
                "mean_estimate": float(p.mean()),
                "mean_estimate_ivt": float(p[sub == 1.0].mean()), "mean_estimate_no_ivt": float(p[sub == 0.0].mean()),
                "n_ivt": int((sub == 1.0).sum()), "n_no_ivt": int((sub == 0.0).sum()),
                "model_based_concordance": model_based_concordance(p)}
    res["max_abs_difference"] = worst
    res["passes_independent"] = bool(worst <= TOOL_TOLERANCE)
    res["predictions_sha256"] = sha256(preds_blind)
    res["records_sha256"] = sha256(OUT / "run" / "records_blind.json")
    return res


def ui_sample(seed: int = SEED) -> Path:
    """The first 1,000 patients of the primary template with random 90-day mRS values, for the
    browser check (step 1): no real outcome is shown."""
    t = pd.read_csv(OUT / "template_primary.csv", dtype=str, keep_default_na=False)
    s = t.iloc[:1000].copy()
    s["mrs90"] = np.random.default_rng(seed).integers(0, 7, len(s)).astype(str)
    f = OUT / "run" / "sample1000_random_outcomes.csv"
    s.to_csv(f, index=False)
    return f


def ui_compare(ui_json: Path, preds_blind: Path) -> dict:
    """The AUROCs the app shows for the random-outcome sample against those of this run's estimates."""
    from sklearn.metrics import roc_auc_score
    ui = json.loads(Path(ui_json).read_text())
    preds = mrsclean.load_predictions(preds_blind)
    s = pd.read_csv(OUT / "run" / "sample1000_random_outcomes.csv", dtype=str, keep_default_na=False)
    mrs = pd.to_numeric(s["mrs90"]).to_numpy()
    y = {"good_outcome": (mrs <= 2).astype(float), "death90": (mrs == 6).astype(float)}
    found = []

    def walk(o, path=()):
        if isinstance(o, dict):
            if "auroc" in o and isinstance(o["auroc"], (int, float)):
                found.append((path, o))
            for k, v in o.items():
                walk(v, path + (k,))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, path + (i,))
    walk(ui)
    comps = []
    for path, o in found:
        st = next((p for p in path if p in STAGES), None)
        tg = next((p for p in path if p in TARGETS), None)
        nm = o.get("model") or o.get("learner") or next((p for p in reversed(path) if isinstance(p, str)
                                                         and p not in STAGES + TARGETS), None)
        if st is None or tg is None or nm is None:
            continue
        P = preds["preds"]
        idx = [i for i in range(1000) if P[str(i)][st] is not None and nm in P[str(i)][st][tg]]
        if len(idx) < 20:
            continue
        p = np.array([P[str(i)][st][tg][nm] for i in idx])
        a = float(roc_auc_score(y[tg][idx], p))
        comps.append({"stage": st, "target": tg, "model": nm, "ui": float(o["auroc"]), "run": a,
                      "difference": abs(a - float(o["auroc"]))})
    worst = max((c["difference"] for c in comps), default=None)
    return {"at": now(), "n_results": len(comps), "max_abs_difference": worst,
            "passes": bool(comps) and worst <= UI_TOLERANCE, "results": comps,
            "ui_model_fingerprint": ui.get("model_fingerprint"), "ui_model_version": ui.get("model_version")}


# --------------------------------------------------------------------------- step 2: coding check
# The authors' table of the 6,856 real patients (te Nijenhuis et al., preprint, Tables A.5-A.7) and
# their 1,000-patient TabDDPM sample in the same tables. kind: share (%), median (IQR), mean (SD).
REF = [
    # label, kind, original, tabddpm, how MRS CLEAN is read
    ("Age, years", "median", (71, 63, 81), (73, 63, 81), "r_age"),
    ("Male sex", "share", 3565 / 6856 * 100, 53.8, "male"),
    ("ASPECTS", "median", (9, 8, 10), (9, 8, 10), "bl_ASPECTS"),
    ("Diabetes mellitus", "share", 1134 / 6856 * 100, 15.7, "bl_hist_dm"),
    ("Glucose, mmol/L", "mean", (7.46, 2.87), (8.88, 11.81), "bl_lab_glucose"),
    ("NIHSS on admission", "median", (15, 10, 19), (15, 9, 19), "bl_NIHSS_sumscore"),
    ("Pre-stroke mRS", "median", (0, 0, 1), (0, 0, 1), "bl_hist_premrs"),
    ("IV thrombolysis", "share", 4524 / 6856 * 100, 65.1, "ivt"),
    ("Occlusion: ICA (incl. ICA-T)", "share", 23.09, 22.1, "occ:ICA"),
    ("Occlusion: M1", "share", 55.42, 55.9, "occ:M1"),
    ("Occlusion: M2", "share", 20.01, 20.3, "occ:M2"),
    ("Occlusion: other", "share", 0.71, 1.7, "occ:other"),
    ("Collaterals: absent", "share", 5.41, 4.2, "coll:0"),
    ("Collaterals: under 50 %", "share", 33.97, 35.7, "coll:1"),
    ("Collaterals: 50 % to under 100 %", "share", 39.95, 41.7, "coll:2"),
    ("Collaterals: 100 %", "share", 20.66, 18.4, "coll:3"),
    ("Systolic blood pressure, mmHg", "median", (150, 132, 167), (148, 132, 168), "bl_phy_rrsyst"),
    ("Onset to groin puncture, min", "median", (190, 143, 270), (195, 150, 285), "TI.onsetlsw_to_groin"),
    ("Previous stroke", "share", 1165 / 6856 * 100, 15.3, "bl_hist_is"),
    ("Hypertension", "share", 3565 / 6856 * 100, 49.3, "bl_hist_ht"),
    ("Myocardial infarction", "share", 891 / 6856 * 100, 12.3, "bl_hist_mi"),
    ("Hypercholesterolaemia", "share", 1988 / 6856 * 100, 24.8, "bl_hist_hc"),
    ("Smoking", "share", 4044 / 6856 * 100, 56.5, "smoking"),
    ("Antiplatelet use", "share", 2125 / 6856 * 100, 29.8, "bl_hist_apt"),
    ("Heparin use", "share", 137 / 6856 * 100, 2.0, "bl_hist_hep"),
    ("Statin use", "share", 2399 / 6856 * 100, 32.7, "bl_hist_stat"),
    ("Occlusion side left", "share", 3565 / 6856 * 100, 51.7, "left"),
    ("NIHSS at 24 hours", "median", (9, 3, 16), (9, 3, 16), "fu24h_NIHSS_sumscore"),
    ("Symptomatic intracranial haemorrhage", "share", 411 / 6856 * 100, 4.8, "sICH_HBC"),
    ("Diastolic blood pressure, mmHg", "median", (82, 73, 92), (82, 73, 92), "bl_phy_rrdiast"),
    ("Final eTICI (0-5)", "median", (3, 2, 5), (3, 2, 5), "postetici_c"),
    ("General anaesthesia", "share", 1451 / 6856 * 100, 17.1, "general"),
    ("Procedure duration, min", "mean", (61.09, 32.55), (62.56, 41.11), "TI.groin_to_endproc"),
]
# three rows give the same 52 % (3,565): male sex, hypertension, left side (a possible copying error)
SAME_52 = {"Male sex", "Hypertension", "Occlusion side left"}


def _value(src: pd.DataFrame, how: str, reading: str, kind: str):
    """MRS CLEAN's value for a reference row, as the mapping reads the variable."""
    s = src
    if how == "male":
        return float((s["r_sex"] == 0).mean() * 100)
    if how == "left":
        return float((s["occlside_LeftRight"] == 1).mean() * 100)
    if how == "general":
        return float((s["deepest_anesthmanag"] == 1).mean() * 100)
    if how == "ivt":
        return float((s["r_ivt"] == 1).mean() * 100)
    if how == "smoking":
        codes = {"current": [2], "ever": [1, 2]}[reading]
        return float(s["bl_hist_smoking"].isin(codes).mean() * 100)
    if how.startswith("occ:"):
        site = s["bl_cta_prox_occl_loc_CatSAP"].map(mrsclean.OCCLUSION_DETAIL[reading]).replace({"ICA-T": "ICA"})
        return float((site == how[4:]).mean() * 100)
    if how.startswith("coll:"):
        return float((s["collateral_grade"] == int(how[5:])).mean() * 100)
    if kind == "share":
        return float((s[how] == 1).mean() * 100)
    return s[how]


def coding_check(source: Path) -> dict:
    """Step 2. No outcome is read (mrs_90d and the adverse-event columns are dropped first)."""
    src = mrsclean.load(source).drop(columns=["mrs_90d", "SAE_any", "SAE_strokeprog", "SAE_pneu"])
    rows = []
    for label, kind, orig, tab, how in REF:
        readings = (["registry", "codebook"] if how.startswith("occ:") else ["current", "ever"] if how == "smoking" else [None])
        for rd in readings:
            v = _value(src, how, rd, kind)
            r = {"label": label, "kind": kind, "reading": rd, "reference": orig, "tabddpm_sample": tab}
            if kind == "share":
                d, dt_ = v - orig, tab - orig
                r.update(value=v, difference_pp=d, flag=bool(abs(d) > SHARE_TOL),
                         generator=bool(abs(dt_) > SHARE_TOL and np.sign(dt_) == np.sign(d)))
            elif kind == "median":
                x = v.to_numpy(float)
                med, q1, q3 = (float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75)))
                r.update(value=(med, q1, q3), flag=bool(not (orig[1] <= med <= orig[2])),
                         generator=bool(not (orig[1] <= tab[0] <= orig[2])))
            else:
                x = v.to_numpy(float)
                mu, sd = float(x.mean()), float(x.std(ddof=1))
                z, zt = (mu - orig[0]) / orig[1], (tab[0] - orig[0]) / orig[1]
                r.update(value=(mu, sd), difference_sd=z, flag=bool(abs(z) > SD_TOL),
                         generator=bool(abs(zt) > SD_TOL and np.sign(zt) == np.sign(z)))
            r["same_52_row"] = label in SAME_52
            rows.append(r)
    used_reading = {"registry", "current", None}
    flags = [r for r in rows if r["flag"]]
    return {"at": now(), "n": len(src), "reference_n": 6856, "rows": rows,
            "flags": [{"label": r["label"], "reading": r["reading"], "generator": bool(r["generator"]),
                       "primary_mapping": r["reading"] in used_reading} for r in flags],
            "source_sha256": sha256(source)}


# --------------------------------------------------------------------------- step 3: fidelity
def fidelity_v2() -> dict:
    res = mrsclean.fidelity(OUT, n_boot=N_BOOT)
    checks = []
    for m in ("mr_predicts", "mr_predicts_24h"):
        for tg in TARGETS:
            e = res["models"][m][tg]
            pub = e.get("published_c")
            if pub is None:
                continue
            ok_c = abs(e["registry"]["auroc"] - pub) <= C_TOLERANCE
            ok_s = SLOPE_RANGE[0] <= e["registry"]["slope"] <= SLOPE_RANGE[1]
            e["passes_c"], e["passes_slope"] = bool(ok_c), bool(ok_s)
            ok_oe = True
            if (m, tg) in PUBLISHED_OE:
                e["published_oe_hermes"] = PUBLISHED_OE[(m, tg)]
                ok_oe = abs(e["hermes"]["oe"] - PUBLISHED_OE[(m, tg)]) <= OE_TOLERANCE
                e["passes_oe"] = bool(ok_oe)
            e["passes_v2"] = bool(ok_c and ok_s and ok_oe)
            checks.append(e["passes_v2"])
    res["faithful_v2"] = bool(all(checks))
    res["at"] = now()
    return res


# --------------------------------------------------------------------------- step 5a: FNO reference
def fno_reference(n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """The temporal validation refitted (development 2005-2022, published settings), checked against
    the published test estimates, then the inputs MRS CLEAN lacks averaged over development values
    (patients treated 2016-2022), as the tool does. FNO data stay local; output is aggregates."""
    from sklearn.metrics import roc_auc_score

    from .cohort import flow, split_temporal, stage_cohort
    from .config import features, settings
    from .export import imputation_values
    from .features import design_matrix
    from .metrics import bootstrap_ci
    from .splines import fit_spline_logreg
    base, _ = flow(pd.read_parquet(paths.PROCESSED / "clean.parquet"))
    train = json.loads((paths.PRIVATE / "train_primary.json").read_text())
    stored = pd.read_parquet(paths.PRIVATE / "predictions_primary.parquet")
    test_from = int(train["test_from_year"])
    imp = imputation_values(base[base["year"] < test_from])
    req_ivt = {f.key for f in features() if f.spec.get("requires_ivt")}
    feats = {f.key: f for f in features()}
    res = {"at": now(), "test_from_year": test_from, "averaging_values_from": f"2016-{test_from - 1}", "stages": {}}
    for st in PRIMARY:
        d = stage_cohort(base, st)
        dev, test = split_temporal(d, test_from)
        X_all = design_matrix(d, st)
        model_keys = set(X_all.columns.str.split("__").str[0])
        absent = [k for k in mrsclean.NOT_RECORDED if k in model_keys and k in imp]
        for tg in TARGETS:
            s = train["stages"][st]["targets"][tg]["settings"]["Logistic regression (splines)"]
            y_dev, y_test = dev[tg].to_numpy(float), test[tg].to_numpy(float)
            # the published settings are saved to 6 decimals, so the search is repeated (same seed and
            # candidates) to reproduce the model exactly; its choice must match the saved settings
            m = fit_spline_logreg(X_all.loc[dev.index], y_dev, int(settings()["seed"]))
            assert len(m.knots[next(iter(m.knots))]) == int(s["knots"]) and abs(m.model.settings["C"] - float(s["C"])) < 5e-6, "settings differ"

            p_full = m.predict_proba(X_all.loc[test.index])[:, 1]
            ref = stored[(stored["stage"] == st) & (stored["target"] == tg) & (stored["model"] == "Logistic regression (splines)")]
            ref = ref.set_index("patient_id").loc[test["patient_id"].to_numpy(), "p"].to_numpy()
            reproduced = float(np.max(np.abs(p_full - ref)))
            # averaged copies of each test patient
            perms = {k: _perm(len(imp[k]["values"]), k) for k in absent}
            rows, owner = [], []
            for j, (_, r) in enumerate(test.iterrows()):
                for smp in range(20):
                    x = r.copy()
                    for k in absent:
                        if k in req_ivt and r["ivt"] == 0:
                            continue
                        vals = imp[k]["values"]
                        x[k] = vals[perms[k][smp % len(vals)]]
                    rows.append(x)
                    owner.append(j)
            T = pd.DataFrame(rows)
            for k in absent:
                if feats[k].type != "categorical":
                    T[k] = pd.to_numeric(T[k], errors="coerce").astype(float)
            Xa = design_matrix(T, st)
            pa = m.predict_proba(Xa)[:, 1]
            owner = np.asarray(owner)
            p_avg = np.bincount(owner, weights=pa) / np.bincount(owner)
            ci = bootstrap_ci(y_test, {"all_inputs": p_full, "averaged": p_avg}, n_boot=n_boot, seed=seed,
                              ref="all_inputs", calibration=False)
            res["stages"].setdefault(st, {})[tg] = {
                "n_test": len(test), "events_test": int(y_test.sum()), "reproduced_max_difference": reproduced,
                "auroc_all_inputs": float(roc_auc_score(y_test, p_full)), "auroc_all_inputs_ci": ci["all_inputs"]["auroc"],
                "auroc_averaged": float(roc_auc_score(y_test, p_avg)), "auroc_averaged_ci": ci["averaged"]["auroc"],
                "difference": float(roc_auc_score(y_test, p_avg) - roc_auc_score(y_test, p_full)),
                "difference_ci": ci["averaged"]["delta_auroc_vs_ref"]["ci"],
                "mbc_all_inputs": model_based_concordance(p_full), "mbc_averaged": model_based_concordance(p_avg),
                "averaged_inputs": absent}
    return res


# --------------------------------------------------------------------------- outcomes
def outcomes(random_seed: int | None = None) -> dict:
    """The 90-day outcomes, or random ones (for testing the code before the real outcomes are read)."""
    df = mrsclean.typed(OUT)
    mrs = df["mrs90"].to_numpy(float)
    if random_seed is not None:
        mrs = np.random.default_rng(random_seed).integers(0, 7, len(mrs)).astype(float)
    return {"good_outcome": (mrs <= 2).astype(float), "death90": (mrs == 6).astype(float)}


def _comparator(cm: str, frames: dict, intercepts: str = "registry", no_collaterals: bool = False):
    X = frames[cm][0].copy()
    if no_collaterals:
        X["collaterals"] = 0.0      # removes the collateral terms (MR PREDICTS also its interaction with treatment)
    f = mrpredicts.lp_mr_predicts if cm == "mr_predicts" else mrpredicts.lp_mr_predicts_24h
    lp = f(**{k: X[k].to_numpy() for k in X})     # @24H: 0.115803 x score, so a score of 0 removes it
    return X.index.to_numpy(), mrpredicts.outcomes(lp, cm, intercepts)


def verdict(diff: float, lo: float, hi: float) -> str:
    """Section 6, checked in this order."""
    if diff >= MARGIN and lo > 0:
        return "better"
    if lo > -MARGIN:
        return "non-inferior"
    if diff <= -MARGIN and hi < 0:
        return "worse"
    return "not shown to be non-inferior"


def _boot_mean_ratio(y, p, n_boot, seed):
    return mrsclean._boot_oe(y, p, n_boot, seed)


def _recalibrate(y, p, half):
    """Step 7: intercept update (and, where needed, logistic recalibration) in one half, checked in the other."""
    from .metrics import _fit_logistic, expit, logit, oe_ratio, summary
    a_idx, b_idx = half, ~half
    lp = logit(p)
    shift = float(_fit_logistic(np.zeros(a_idx.sum()), y[a_idx], offset=lp[a_idx], slope=False)[0])
    pb = expit(lp[b_idx] + shift)
    sb, s0 = summary(y[b_idx], pb), summary(y[b_idx], p[b_idx])
    out = {"shift": shift, "n_estimate": int(a_idx.sum()), "n_check": int(b_idx.sum()),
           "check_before": {"citl": s0["citl"], "slope": s0["slope"], "oe": oe_ratio(y[b_idx], p[b_idx]), "ici": s0["ici"]},
           "check_after": {"citl": sb["citl"], "slope": sb["slope"], "oe": oe_ratio(y[b_idx], pb), "ici": sb["ici"]}}
    w = _fit_logistic(lp[a_idx], y[a_idx])
    pl = expit(w[0] + w[1] * lp[b_idx])
    out["logistic"] = {"intercept": float(w[0]), "slope": float(w[1]), "check_ici": summary(y[b_idx], pl)["ici"]}
    return out


def main_results(preds_path: Path, random_seed: int | None = None, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Steps 5b, 6, 7 and 10 with the rules of section 6."""
    from sklearn.metrics import roc_auc_score

    from .metrics import (
        bootstrap_ci,
        calibration_bins,
        decision_curve,
        expit,
        logit,
        oe_ratio,
        summary,
    )
    preds = mrsclean.load_predictions(preds_path)
    Y = outcomes(random_seed)
    n = len(Y["death90"])
    frames = mrsclean.comparator_frames(OUT)
    half_all = np.random.default_rng(seed).permutation(n) % 2 == 0
    res = {"at": now(), "random_outcomes": random_seed, "predictions_sha256": sha256(preds_path),
           "primary": {}, "secondary": {}, "learners": {}}
    for st in STAGES:
        res["learners"][st] = {}
        for tg in TARGETS:
            idx, p = _chosen(preds, st, tg, n)
            y = Y[tg][idx]
            res["learners"][st][tg] = {}
            for nm in preds["meta"][st][tg]["learners"]:
                q = np.array([preds["preds"][str(i)][st][tg][nm] for i in idx])
                s = summary(y, q)
                res["learners"][st][tg][nm] = {**{k: s[k] for k in ("n", "auroc", "citl", "slope", "brier_scaled")},
                                               "oe": oe_ratio(y, q), "chosen": nm == preds["meta"][st][tg]["chosen"]}
            s = summary(y, p)
            ci = bootstrap_ci(y, {"mtop": p}, n_boot=n_boot, seed=seed)["mtop"]
            oe = oe_ratio(y, p)
            oe_ci = _boot_mean_ratio(y, p, n_boot, seed)
            e = {"model": preds["meta"][st][tg]["chosen"],
                 **{k: s[k] for k in ("n", "events", "prevalence", "mean_pred", "auroc", "brier", "brier_scaled", "citl", "slope", "ici")},
                 "ci": {"auroc": ci["auroc"], "citl": ci["citl"], "slope": ci["slope"], "oe": oe_ci}, "oe": oe,
                 "calibration_bins": calibration_bins(y, p), "mbc": model_based_concordance(p)}
            e["mbc_gap"] = e["auroc"] - e["mbc"]
            e["loss_beyond_case_mix"] = bool(e["auroc"] < e["mbc"] - MBC_GAP)
            e["update_needed"] = bool((oe < OE_RANGE[0] or oe > OE_RANGE[1]) and (oe_ci[0] > 1 or oe_ci[1] < 1))
            e["slope_ok"] = bool(SLOPE_RANGE[0] <= s["slope"] <= SLOPE_RANGE[1])
            rc = _recalibrate(y, p, half_all[idx])
            e["recalibration"] = rc
            e["update_enough"] = e["slope_ok"]
            if not e["slope_ok"]:
                e["refit_recommended"] = bool(rc["logistic"]["check_ici"] > ICI_REFIT)
            if st not in COMPARATOR:
                res["secondary"].setdefault(st, {})[tg] = e
                continue
            cm = COMPARATOR[st]
            cidx, qo = _comparator(cm, frames)
            pos = {i: j for j, i in enumerate(idx)}
            keep_c = np.array([k for k, i in enumerate(cidx) if i in pos])
            keep = np.array([pos[cidx[k]] for k in keep_c])
            yy, pp, qq = y[keep], p[keep], qo[tg][keep_c]
            b = bootstrap_ci(yy, {"mtop": pp, cm: qq}, n_boot=n_boot, seed=seed, ref=cm, calibration=False)
            diff = float(roc_auc_score(yy, pp) - roc_auc_score(yy, qq))
            lo, hi = b["mtop"]["delta_auroc_vs_ref"]["ci"]
            e["vs_comparator"] = {"comparator": cm, "n": len(yy), "mtop_auroc": float(roc_auc_score(yy, pp)),
                                  "comparator_auroc": float(roc_auc_score(yy, qq)), "comparator_auroc_ci": b[cm]["auroc"],
                                  "difference": diff, "difference_ci": [lo, hi], "verdict": verdict(diff, lo, hi),
                                  "comparator_oe": oe_ratio(yy, qq)}
            pu = expit(logit(pp) + rc["shift"])
            e["decision_curve"] = {"n": len(yy), "rows": [
                {"t": a["t"], "mtop": a["model"], "mtop_updated": u["model"], "comparator": c["model"], "all": a["all"]}
                for a, u, c in zip(decision_curve(yy, pp), decision_curve(yy, pu), decision_curve(yy, qq))]}
            res["primary"].setdefault(st, {})[tg] = e
    return res


# --------------------------------------------------------------------------- step 8: subgroups
SUBGROUPS = dict(mrsclean.SUBGROUPS)
SUBGROUPS["Pre-stroke mRS 0-2"] = lambda d: d["mrs_pre"] <= 2
SUBGROUPS["Pre-stroke mRS 3-5"] = lambda d: d["mrs_pre"] >= 3


def subgroups(preds_path: Path, random_seed: int | None = None, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    from sklearn.metrics import roc_auc_score

    from .metrics import oe_ratio, summary
    preds = mrsclean.load_predictions(preds_path)
    Y = outcomes(random_seed)
    df = mrsclean.typed(OUT)
    n = len(df)
    frames = mrsclean.comparator_frames(OUT)
    res = {"at": now(), "random_outcomes": random_seed, "groups": {}}
    for st in PRIMARY:
        cm = COMPARATOR[st]
        cidx, qo = _comparator(cm, frames)
        cpos = {i: k for k, i in enumerate(cidx)}
        for tg in TARGETS:
            idx, p = _chosen(preds, st, tg, n)
            y = Y[tg][idx]
            key = f"{st}/{tg}"
            res["groups"][key] = {}
            for name, f in SUBGROUPS.items():
                m = f(df).fillna(False).to_numpy()[idx]
                ys, ps = y[m], p[m]
                g = {"n": int(m.sum()), "events": int(ys.sum())}
                if ys.sum() < mrsclean.MIN_EVENTS or (1 - ys).sum() < mrsclean.MIN_EVENTS:
                    g["insufficient"] = True
                    res["groups"][key][name] = g
                    continue
                s = summary(ys, ps)
                g.update(auroc=s["auroc"], slope=s["slope"], citl=s["citl"], oe=oe_ratio(ys, ps),
                         auroc_ci=fast_boot_auc(ys, {"m": ps}, n_boot, seed)["m"]["auroc"])
                sel = [j for j, i in enumerate(idx) if m[j] and i in cpos]
                if sel:
                    yy, pp = y[sel], p[sel]
                    qq = qo[tg][[cpos[idx[j]] for j in sel]]
                    if yy.sum() >= mrsclean.MIN_EVENTS and (1 - yy).sum() >= mrsclean.MIN_EVENTS:
                        b = fast_boot_auc(yy, {"mtop": pp, cm: qq}, n_boot, seed, ref=cm)
                        g["vs_comparator"] = {"n": len(yy), "difference": float(roc_auc_score(yy, pp) - roc_auc_score(yy, qq)),
                                              "difference_ci": b["mtop"]["delta_auroc_vs_ref"]["ci"]}
                res["groups"][key][name] = g
    return res


# --------------------------------------------------------------------------- step 9: sensitivity
def _frames_codebook():
    """Comparator inputs with the occlusion codes read as the codebook labels them."""
    import shutil
    tmp = OUT / "run" / "codebook_view"
    tmp.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT / "template_occlusion_codebook.csv", tmp / "template_primary.csv")
    shutil.copy(OUT / "comparators_occlusion_codebook.csv", tmp / "comparators.csv")
    return mrsclean.comparator_frames(tmp), tmp


def sensitivity(preds_path: Path, variants: dict, random_seed: int | None = None, n_boot: int = N_BOOT,
                seed: int = SEED) -> dict:
    from sklearn.metrics import roc_auc_score

    from .fed.schema import read_site_csv
    from .metrics import oe_ratio, summary
    preds = mrsclean.load_predictions(preds_path)
    Y = outcomes(random_seed)
    df = mrsclean.typed(OUT)
    n = len(df)
    _, rep = read_site_csv(OUT / "template_primary.csv")
    recorded = [k for k in rep["columns_found"] if k in df.columns and k not in ("mrs90",)]
    complete = df[recorded].notna().all(axis=1).to_numpy()
    frames = mrsclean.comparator_frames(OUT)
    fcb, _ = _frames_codebook()
    vp = {k: mrsclean.load_predictions(Path(v)) for k, v in variants.items()}
    res = {"at": now(), "random_outcomes": random_seed, "mtop": {}, "comparators": {}}

    def m3(y, p):
        s = summary(y, p)
        return {"n": s["n"], "auroc": s["auroc"], "citl": s["citl"], "oe": oe_ratio(y, p)}
    for st in PRIMARY:
        for tg in TARGETS:
            idx, p = _chosen(preds, st, tg, n)
            y = Y[tg][idx]
            key = f"{st}/{tg}"
            base = m3(y, p)
            out = {"primary": base}
            for name, P in vp.items():
                vidx, q = _chosen(P, st, tg, n)
                assert np.array_equal(vidx, idx), f"{name}: different patients"
                r = m3(y, q)
                r["delta_auroc"] = r["auroc"] - base["auroc"]
                r["changed_patients"] = int((np.abs(q - p) > 1e-12).sum())
                out[name] = r
            c = complete[idx]
            out["complete_patients"] = m3(y[c], p[c])
            res["mtop"][key] = out
            # comparators: collateral term removed; HERMES intercepts; codebook occlusion
            cm = COMPARATOR[st]
            comp = {}
            pos = {i: j for j, i in enumerate(idx)}
            for label, fr, kw in (("registry", frames, {}), ("hermes", frames, {"intercepts": "hermes"}),
                                  ("no_collaterals", frames, {"no_collaterals": True}), ("occlusion_codebook", fcb, {})):
                cidx, qo = _comparator(cm, fr, **kw)
                if label == "occlusion_codebook":
                    Pv = vp.get("occlusion_codebook")
                    if Pv is None:
                        continue
                    vidx, qv = _chosen(Pv, st, tg, n)
                    vpos = {i: j for j, i in enumerate(vidx)}
                    kc = [k for k, i in enumerate(cidx) if i in vpos]
                    yy = Y[tg][cidx[kc]]
                    pp = qv[[vpos[cidx[k]] for k in kc]]
                else:
                    kc = [k for k, i in enumerate(cidx) if i in pos]
                    yy = y[[pos[cidx[k]] for k in kc]]
                    pp = p[[pos[cidx[k]] for k in kc]]
                qq = qo[tg][kc]
                b = fast_boot_auc(yy, {"mtop": pp, cm: qq}, n_boot, seed, ref=cm)
                comp[label] = {"n": len(yy), "comparator_auroc": float(roc_auc_score(yy, qq)),
                               "comparator_oe": oe_ratio(yy, qq), "comparator_citl": summary(yy, qq)["citl"],
                               "mtop_auroc": float(roc_auc_score(yy, pp)),
                               "difference": float(roc_auc_score(yy, pp) - roc_auc_score(yy, qq)),
                               "difference_ci": b["mtop"]["delta_auroc_vs_ref"]["ci"]}
            res["comparators"][key] = comp
    return res


# --------------------------------------------------------------------------- command line
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert"); c.add_argument("source", type=Path)
    sub.add_parser("records")
    t = sub.add_parser("toolcheck"); t.add_argument("predictions", type=Path)
    sub.add_parser("ui-sample")
    u = sub.add_parser("ui-compare"); u.add_argument("ui", type=Path); u.add_argument("predictions", type=Path)
    k = sub.add_parser("codingcheck"); k.add_argument("source", type=Path)
    sub.add_parser("fidelity")
    sub.add_parser("casemix")
    sub.add_parser("reference")
    for name in ("main", "subgroups", "sensitivity"):
        p = sub.add_parser(name)
        p.add_argument("predictions", type=Path)
        p.add_argument("--random-outcomes", type=int, default=None)
        p.add_argument("--variant", nargs=2, action="append", metavar=("NAME", "PREDICTIONS"), default=[])
    a = ap.parse_args(argv)
    sfx = "" if getattr(a, "random_outcomes", None) is None else f"_random{a.random_outcomes}"
    if a.cmd == "convert":
        print(save("conversion.json", convert_all(a.source)))
    elif a.cmd == "records":
        print(json.dumps(write_records_all(), indent=1))
    elif a.cmd == "toolcheck":
        print(save("step1_toolcheck.json", tool_check(a.predictions)))
    elif a.cmd == "ui-sample":
        print(ui_sample())
    elif a.cmd == "ui-compare":
        print(save("step1_browser_check.json", ui_compare(a.ui, a.predictions)))
    elif a.cmd == "codingcheck":
        print(save("step2_codingcheck.json", coding_check(a.source)))
    elif a.cmd == "fidelity":
        print(save("step3_fidelity.json", fidelity_v2()))
    elif a.cmd == "casemix":
        cm = mrsclean.case_mix(OUT)
        cm["at"] = now()
        print(save("step4_casemix.json", cm))
    elif a.cmd == "reference":
        print(save("step5a_fno_reference.json", fno_reference()))
    elif a.cmd == "main":
        print(save(f"steps5b_6_7_10_main{sfx}.json", main_results(a.predictions, a.random_outcomes)))
    elif a.cmd == "subgroups":
        print(save(f"step8_subgroups{sfx}.json", subgroups(a.predictions, a.random_outcomes)))
    elif a.cmd == "sensitivity":
        print(save(f"step9_sensitivity{sfx}.json",
                   sensitivity(a.predictions, {nm: f for nm, f in a.variant}, a.random_outcomes)))


if __name__ == "__main__":
    main()
