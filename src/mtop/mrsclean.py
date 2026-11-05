"""MRS CLEAN synthetic dataset -> MTOP upload template (dry run before the MR CLEAN Registry request).

MRS CLEAN (te Nijenhuis et al., Zenodo, DOI 10.5281/zenodo.18846304) holds 15,000 synthetic
patients generated with TabDDPM from the MR CLEAN Registry and the MR CLEAN-LATE, -MED and -NOIV
trials. Its licence allows private use only: the source file and every row-level output of this
module stay outside git (default output folder data/raw/mrs_clean/, which .gitignore covers; the
command refuses a folder git would track). Only aggregate counts are printed.

The mapping is the variable-mapping table of the analysis plan ("MTOP on MRS CLEAN synthetic data:
analysis plan", 5 November 2026). Values are written as a partner centre would write them, and the
app's own template reader (fed/schema.py; web/view_network.js in the browser) turns them into model
inputs: out-of-range values become unknown there, exactly as in "Validate on your data".

Two decisions differ from the Zenodo codebook, both documented in the plan:
  * occlusion codes: the codebook's labels are shifted by one. The authors' own table of the real
    patients (preprint Table A.5: ICA 23.1 %, M1 55.4 %, M2 20.0 %, other 0.7 %) matches
    0 = other, 1 = ICA, 2 = ICA-T, 3 = M1, 4 = M2 and not the codebook's 0 = ICA ... 4 = M2.
  * anaesthesia code 0 is "no general anaesthesia" (local or sedation); it is written as sedation.

Usage:
    python -m mtop.mrsclean convert path/to/mrs_clean_synthetic_dataset.csv [--out data/raw/mrs_clean]
writes the primary template, the three conversion-level sensitivity variants and a side file with
the comparator variables (collateral score, ICA vs ICA-T, eTICI) used by the MR PREDICTS formulas.
    python -m mtop.mrsclean casemix
compares MRS CLEAN with the FNO cohort on the app's Table 1 rows (aggregates, reports/mrs_clean/).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import mrpredicts, paths

SOURCE_COLUMNS = [
    "StudySubjectID", "r_age", "r_sex", "bl_NIHSS_sumscore", "bl_phy_rrsyst", "bl_phy_rrdiast",
    "bl_lab_glucose", "bl_hist_is", "bl_hist_ht", "bl_hist_af", "bl_hist_dm", "bl_hist_mi", "bl_hist_pad",
    "bl_hist_hc", "bl_hist_smoking", "bl_hist_premrs", "bl_hist_apt", "bl_hist_hep", "bl_hist_ahtd",
    "bl_hist_stat", "occlside_LeftRight", "bl_cta_prox_occl_loc_CatSAP", "bl_ASPECTS", "collateral_grade",
    "postetici_c", "r_ivt", "TI.onsetlsw_to_groin", "deepest_anesthmanag", "TI.groin_to_endproc",
    "number_iat_attempts", "balloon_used", "fu24h_NIHSS_sumscore", "sICH_HBC", "mrs_90d", "SAE_any",
    "SAE_strokeprog", "SAE_pneu",
]

# occlusion site: detailed MR CLEAN site, then the MTOP category it joins
OCCLUSION_DETAIL = {"registry": {0: "other", 1: "ICA", 2: "ICA-T", 3: "M1", 4: "M2"},       # primary
                    "codebook": {0: "ICA", 1: "ICA-T", 2: "M1", 3: "M1", 4: "M2"}}          # as labelled
OCCLUSION_MTOP = {"ICA": "ICA", "ICA-T": "ICA", "M1": "M1", "M2": "M2", "other": ""}
ETICI = {0: "0", 1: "1", 2: "2a", 3: "2b", 4: "2c", 5: "3"}
ANESTHESIA = {0: "sedation", 1: "general"}                 # 2-4: other, undefined -> unknown
SIDE = {1: "1", 2: "0"}                                    # 3 both, 4 unknown -> unknown

# MTOP inputs MRS CLEAN does not record: left out of the template, so the app lists them as
# "not found" and treats them as unknown (averaged over FNO registry values since 2016)
NOT_RECORDED = ["ivt_external", "onset_to_ivt", "transfer", "onset_hour", "weekend", "bmi", "tia_prior",
                "alcohol", "technique", "stenting", "angioplasty", "ia_lysis", "ich_other", "nihss_7d",
                "edema", "decompression"]

VARIANTS = {
    # name: (eTICI 2c written as, smoker definition, antithrombotics when neither drug is recorded)
    "primary": ("2c", "current", "none"),
    "tici2c_as_2b": ("2b", "current", "none"),
    "smoker_ever": ("2c", "ever", "none"),
    "antithrombotic_unknown": ("2c", "current", ""),
}


def _txt(v) -> str:
    """A number as the template writes it: integers without decimals, missing as empty."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    f = float(v)
    return str(int(f)) if f.is_integer() else f"{f:g}"


def load(path) -> pd.DataFrame:
    src = pd.read_csv(path)
    missing = [c for c in SOURCE_COLUMNS if c not in src.columns]
    if missing:
        raise SystemExit(f"Not the MRS CLEAN layout; missing columns: {', '.join(missing)}")
    return src


def convert(src: pd.DataFrame, tici_2c: str = "2c", smoker: str = "current",
            antithrombotic_none: str = "none", occlusion: str = "registry") -> tuple[pd.DataFrame, pd.DataFrame]:
    """MRS CLEAN rows -> (template rows as text, comparator side file). Row order is kept."""
    s = src
    detail = s["bl_cta_prox_occl_loc_CatSAP"].map(OCCLUSION_DETAIL[occlusion])
    smoke = s["bl_hist_smoking"].isin([2] if smoker == "current" else [1, 2]).astype(int)
    anti = np.where(s["bl_hist_hep"] == 1, "anticoagulant",
                    np.where(s["bl_hist_apt"] == 1, "antiplatelet", antithrombotic_none))
    etici = s["postetici_c"].map(ETICI)
    t = pd.DataFrame({
        "id": s["StudySubjectID"],
        "age": s["r_age"].map(_txt),
        "sex_male": (1 - s["r_sex"]).map(_txt),                       # 0 = male, 1 = female
        "mrs_pre": s["bl_hist_premrs"].map(_txt),
        "nihss_adm": s["bl_NIHSS_sumscore"].map(_txt),
        "occlusion": detail.map(OCCLUSION_MTOP).fillna(""),
        "side_left": s["occlside_LeftRight"].map(SIDE).fillna(""),
        "aspects": s["bl_ASPECTS"].map(_txt),
        "ivt": s["r_ivt"].where(s["r_ivt"].isin([0, 1])).map(_txt),   # 9 = unknown
        "onset_to_puncture": s["TI.onsetlsw_to_groin"].map(_txt),     # from last seen well
        "glucose": s["bl_lab_glucose"].map(_txt),
        "sbp": s["bl_phy_rrsyst"].map(_txt),
        "dbp": s["bl_phy_rrdiast"].map(_txt),
        "htn": s["bl_hist_ht"].map(_txt),
        "dm": s["bl_hist_dm"].map(_txt),
        "hld": s["bl_hist_hc"].map(_txt),                             # hypercholesterolaemia
        "arrhythmia": s["bl_hist_af"].map(_txt),                      # atrial fibrillation only
        "stroke_prior": s["bl_hist_is"].map(_txt),                    # ischaemic stroke only
        "smoking": smoke.map(_txt),
        "statin": s["bl_hist_stat"].map(_txt),
        "heart3": np.where(s["bl_hist_mi"] == 1, "coronary", "none"),  # myocardial infarction only
        "antithrombotic": anti,                                       # heparin only; no oral anticoagulants
        "tici": etici.replace({"2c": tici_2c}),                       # the reader counts 2c as 3
        "procedure_min": s["TI.groin_to_endproc"].map(_txt),
        "onset_to_recan": (s["TI.onsetlsw_to_groin"] + s["TI.groin_to_endproc"]).map(_txt),
        "anesthesia": s["deepest_anesthmanag"].map(ANESTHESIA).fillna(""),
        "nihss_24h": s["fu24h_NIHSS_sumscore"].map(_txt),
        "sich": s["sICH_HBC"].map(_txt),                              # Heidelberg classification
        "mrs90": s["mrs_90d"].map(_txt),
        "collaterals_tan": s["collateral_grade"].map(_txt),           # optional template column
    })
    side = pd.DataFrame({
        "id": s["StudySubjectID"],
        "occlusion_detail": detail,                                   # ICA vs ICA-T for MR PREDICTS
        "collateral_grade": s["collateral_grade"],
        "etici": etici,
        "ivt_code": s["r_ivt"],
        "late_window": (s["TI.onsetlsw_to_groin"] > 390).astype(int),  # punctured after 6.5 h
        "mrs_90d": s["mrs_90d"],
    })
    return t, side


def _check_ignored(out: Path) -> None:
    """Refuse an output folder that git would track (the licence forbids sharing the data)."""
    probe = out / "probe.csv"
    r = subprocess.run(["git", "-C", str(paths.ROOT), "check-ignore", "-q", str(probe)], capture_output=True, check=False)
    inside = paths.ROOT in out.resolve().parents or out.resolve() == paths.ROOT
    if inside and r.returncode != 0:
        raise SystemExit(f"Refusing to write row-level MRS CLEAN data to {out}: git would track it.")


def typed(out: Path) -> pd.DataFrame:
    """The primary template as the app reads it, plus the columns the Table 1 rows need."""
    from .fed.schema import read_site_csv
    df, _ = read_site_csv(out / "template_primary.csv")
    side = pd.read_csv(out / "comparators.csv")
    # ICA-T is an intracranial ICA occlusion in MTOP's categories; "other" has no MTOP category
    df["occlusion_detail"] = side["occlusion_detail"].map({"ICA": "ICA", "ICA-T": "ICA", "M1": "M1", "M2": "M2"})
    df["reperfusion_2b3"] = (df["tici"] >= 3).astype(float).where(df["tici"].notna())
    return df


# FNO patients comparable with the MR CLEAN Registry: anterior-circulation intracranial occlusion
# (the subset in the baseline table sent to the MR PREDICTS team)
MR_CLEAN_LIKE = ["ICA", "M1", "M2", "ACA", "tandem", "tandem_cca"]


def case_mix(out: Path, min_cell: int = 10) -> dict:
    """Table 1 rows of the app for all FNO patients, the MR CLEAN-like FNO subset and MRS CLEAN, with
    the standardised mean difference (SMD) of MRS CLEAN against each. Aggregates only; an FNO count
    under min_cell is reported as small, and the SMD that would give it away is withheld."""
    from .cohort import flow
    from .explorer import _smd_cont, _smd_prop, _t1_cell, _t1_specs, _t1_values
    base, _ = flow(pd.read_parquet(paths.PROCESSED / "clean.parquet"))
    groups = {"fno_all": base, "fno_mrclean_like": base[base["occlusion_detail"].isin(MR_CLEAN_LIKE)],
              "mrs_clean": typed(out)}
    rows = []
    for spec in _t1_specs():
        if "section" in spec:
            rows.append({"section": spec["section"]["en"]})
            continue
        cells = {k: _t1_cell(spec, g, min_cell) for k, g in groups.items()}
        recorded = cells["mrs_clean"]["missing"] is not None and cells["mrs_clean"]["missing"] < 100
        smd = _smd_cont if spec["kind"] == "median" else _smd_prop
        row = {"var": spec["var"], "label": spec["label"]["en"], "kind": spec["kind"], "recorded": recorded, **cells}
        for k in ("fno_all", "fno_mrclean_like"):
            ok = recorded and not cells[k].get("small")
            row[f"smd_vs_{k}"] = smd(_t1_values(spec, groups[k]), _t1_values(spec, groups["mrs_clean"])) if ok else None
        rows.append(row)
    return {"n": {k: int(len(g)) for k, g in groups.items()}, "min_cell": min_cell, "rows": rows}


# Published results in the MR CLEAN Registry (the plan's comparator table): c-statistics, and for MR
# PREDICTS@24H the survival c-statistic, which equals that for death
PUBLISHED_C = {("mr_predicts", "good_outcome"): 0.80, ("mr_predicts_24h", "good_outcome"): 0.91,
               ("mr_predicts_24h", "death90"): 0.89}
C_TOLERANCE, SLOPE_RANGE = 0.03, (0.8, 1.2)


def comparator_frames(out: Path) -> dict:
    """Inputs of both published models for the patients they apply to (settings fixed in the plan):
    MR PREDICTS on patients punctured within 6.5 h, MR PREDICTS@24H on all; patients with occlusion code
    0, unknown IVT or an input outside the app's ranges are left out."""
    df = typed(out)
    side = pd.read_csv(out / "comparators.csv", dtype={"etici": str})
    loc = side["occlusion_detail"].map({"ICA": "ICA-(T)", "ICA-T": "ICA-(T)", "M1": "M1", "M2": "M2"})
    etici = side["etici"].map({"0": 0, "1": 1, "2a": 2, "2b": 3, "2c": 4, "3": 5}).astype(float)
    y = df[["good_outcome", "death90"]]
    mrp = pd.DataFrame({"evt": 1.0, "age": df["age"], "nihss": df["nihss_adm"], "premrs": df["mrs_pre"],
                        "diabetes": df["dm"], "sbp": df["sbp"], "ivt": df["ivt"],
                        "glucose_mgdl": df["glucose"] * mrpredicts.GLUCOSE_MG_PER_MMOL, "aspects": df["aspects"],
                        "location": loc, "collaterals": df["collaterals_tan"], "onset_to_groin": df["onset_to_puncture"]})
    m24 = pd.DataFrame({"age": df["age"], "nihss": df["nihss_adm"], "diabetes": df["dm"], "premrs": df["mrs_pre"],
                        "location": loc, "collaterals": df["collaterals_tan"], "etici": etici,
                        "nihss_24h": df["nihss_24h"], "sich": df["sich"]})
    ok_mrp = mrp.notna().all(axis=1) & (mrp["onset_to_groin"] <= 390)
    ok_24 = m24.notna().all(axis=1)
    return {"mr_predicts": (mrp[ok_mrp], y[ok_mrp]), "mr_predicts_24h": (m24[ok_24], y[ok_24]),
            "excluded": {"mr_predicts_beyond_6_5h": int((mrp.drop(columns="onset_to_groin").notna().all(axis=1)
                                                          & (mrp["onset_to_groin"] > 390)).sum()),
                         "mr_predicts_incomplete": int((~mrp.notna().all(axis=1)).sum()),
                         "mr_predicts_24h_incomplete": int((~ok_24).sum())}}


def fidelity(out: Path, n_boot: int = 2000) -> dict:
    """The plan's fidelity check: both published models on MRS CLEAN, against their Registry results."""
    from .metrics import bootstrap_ci, oe_ratio, summary
    fr = comparator_frames(out)
    lp = {"mr_predicts": mrpredicts.lp_mr_predicts(**{k: fr["mr_predicts"][0][k].to_numpy() for k in fr["mr_predicts"][0]}),
          "mr_predicts_24h": mrpredicts.lp_mr_predicts_24h(**{k: fr["mr_predicts_24h"][0][k].to_numpy()
                                                               for k in fr["mr_predicts_24h"][0]})}
    res = {"excluded": fr["excluded"], "models": {}}
    for model in ("mr_predicts", "mr_predicts_24h"):
        y_all = fr[model][1]
        res["models"][model] = {"n": int(len(y_all))}
        for target in ("good_outcome", "death90"):
            y = y_all[target].to_numpy()
            entry = {}
            for which in ("registry", "hermes"):
                p = mrpredicts.outcomes(lp[model], model, which)[target]
                s = summary(y, p)
                entry[which] = {k: s[k] for k in ("events", "prevalence", "mean_pred", "auroc", "citl", "slope", "ici", "brier_scaled")}
                entry[which]["oe"] = oe_ratio(y, p)
                if which == "registry":
                    ci = bootstrap_ci(y, {"m": p}, n_boot=n_boot, seed=2026)["m"]
                    entry[which]["ci"] = {k: ci[k] for k in ("auroc", "citl", "slope")}
            pub = PUBLISHED_C.get((model, target))
            if pub is not None:
                a, b = entry["registry"]["auroc"], entry["registry"]["slope"]
                entry["published_c"] = pub
                entry["passes"] = bool(abs(a - pub) <= C_TOLERANCE and SLOPE_RANGE[0] <= b <= SLOPE_RANGE[1])
            res["models"][model][target] = entry
    checks = [e["passes"] for m in res["models"].values() for t, e in m.items() if isinstance(e, dict) and "passes" in e]
    res["faithful"] = bool(all(checks))
    return res


def _cell_text(c: dict, kind: str) -> str:
    if kind == "median":
        return "–" if c["median"] is None else f"{c['median']:g} ({c['iqr'][0]:g}–{c['iqr'][1]:g})"
    if c.get("small"):
        return "<10"
    return "–" if c["pct"] is None else f"{c['n']:,} ({c['pct']:.1f} %)"


def case_mix_markdown(cm: dict) -> str:
    n = cm["n"]
    head = (f"| Characteristic | FNO, all (n = {n['fno_all']:,}) | FNO, MR CLEAN-like (n = {n['fno_mrclean_like']:,}) "
            f"| MRS CLEAN (n = {n['mrs_clean']:,}) | SMD vs MR CLEAN-like |\n| --- | --- | --- | --- | --- |")
    out = [head]
    for r in cm["rows"]:
        if "section" in r:
            out.append(f"| **{r['section']}** | | | | |")
            continue
        mrs = _cell_text(r["mrs_clean"], r["kind"]) if r["recorded"] else "not recorded"
        smd = r["smd_vs_fno_mrclean_like"]
        out.append(f"| {r['label']} | {_cell_text(r['fno_all'], r['kind'])} | {_cell_text(r['fno_mrclean_like'], r['kind'])} "
                   f"| {mrs} | {'' if smd is None else f'{smd:+.2f}'} |")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="MRS CLEAN csv -> MTOP template files")
    c.add_argument("source", type=Path)
    c.add_argument("--out", type=Path, default=paths.RAW / "mrs_clean")
    m = sub.add_parser("casemix", help="Table 1 rows: FNO vs MRS CLEAN (aggregates, to reports/)")
    m.add_argument("--out", type=Path, default=paths.RAW / "mrs_clean")
    fd = sub.add_parser("fidelity", help="published MR PREDICTS models on MRS CLEAN vs their Registry results")
    fd.add_argument("--out", type=Path, default=paths.RAW / "mrs_clean")
    a = ap.parse_args(argv)
    if a.cmd == "fidelity":
        res = fidelity(a.out)
        rep = paths.REPORTS / "mrs_clean"
        rep.mkdir(parents=True, exist_ok=True)
        (rep / "fidelity.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return
    if a.cmd == "casemix":
        cm = case_mix(a.out)
        rep = paths.REPORTS / "mrs_clean"
        rep.mkdir(parents=True, exist_ok=True)
        (rep / "casemix.json").write_text(json.dumps(cm, indent=1, ensure_ascii=False))
        (rep / "casemix.md").write_text(case_mix_markdown(cm) + "\n", encoding="utf-8")
        print(json.dumps(cm["n"]))
        return
    _check_ignored(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    src = load(a.source)
    written = {}
    for name, (t2c, smk, anti) in VARIANTS.items():
        t, side = convert(src, tici_2c=t2c, smoker=smk, antithrombotic_none=anti)
        f = a.out / f"template_{name}.csv"
        t.to_csv(f, index=False)
        written[name] = str(f)
        if name == "primary":
            side.to_csv(a.out / "comparators.csv", index=False)
    summary = {"rows": int(len(src)), "files": written, "not_recorded": NOT_RECORDED}
    (a.out / "conversion.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"rows": summary["rows"], "variants": list(written)}, indent=2))


if __name__ == "__main__":
    main()
