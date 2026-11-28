"""ISLES'24 training set: CT perfusion volume extractor and technical demonstration node.

Plan: docs/registrations/v2_imaging_plan.md. Settings: config/isles24.yaml.

  python -m mtop isles --root ISLES24_DIR --headers-only   step 1: clinical column names and missing counts
  python -m mtop isles --root ISLES24_DIR                  step 2: volumes, agreement, demonstration node

The extractor turns a case's perfusion maps (Tmax, CBF) into the optional future columns of the
partner template (config/future.yaml): hypoperfusion (Tmax > 6 s), ischaemic core (relative CBF
< 30 % inside it), mismatch volume, ratio and relative mismatch. Aim A compares the core with the
infarct on follow-up MRI. Aim B writes the cases in the MTOP upload template so that a network node
can read them. No model is fitted or evaluated here.

Row-level outputs go to a git-ignored folder (default data/raw/isles24/); the only published file
is the aggregate summary artifacts/public/isles24_extractor.json.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import paths

NII = (".nii.gz", ".nii")


def settings() -> dict:
    with open(paths.CONFIG / "isles24.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm(s) -> str:
    from .fed.schema import norm

    return norm(s)


# ---------------------------------------------------------------- volumes

def volumes(tmax: np.ndarray, cbf: np.ndarray, voxel_ml: float, ex: dict, clean: bool = False) -> dict:
    """Perfusion volumes of one case, by the definitions fixed in config/isles24.yaml.

    Valid tissue: finite Tmax inside the valid range, finite CBF above 0. Hypoperfusion: valid and
    Tmax > threshold. Reference CBF: median CBF of valid voxels with Tmax below the reference limit
    (needs at least min_reference_voxels, otherwise the core is missing). Core: valid, CBF below
    rcbf_threshold x reference, and (by default) inside the hypoperfused region.
    clean=True drops connected clusters smaller than min_cluster_ml (sensitivity analysis only).
    """
    t = np.asarray(tmax, dtype=float) * float(ex.get("tmax_unit_factor", 1.0))
    c = np.asarray(cbf, dtype=float)
    if t.shape != c.shape:
        raise ValueError(f"Tmax {t.shape} and CBF {c.shape} maps are on different grids")
    lo, hi = ex["tmax_valid_range_s"]
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(t) & np.isfinite(c) & (c > 0) & (t >= lo) & (t <= hi)
        hypo = valid & (t > ex["tmax_threshold_s"])
        ref_vox = valid & (t < ex["reference_tmax_below_s"])
    n_ref = int(ref_vox.sum())
    ref = float(np.median(c[ref_vox])) if n_ref >= ex["min_reference_voxels"] else np.nan
    core = None
    if np.isfinite(ref):
        core = valid & (c < ex["rcbf_threshold"] * ref)
        if ex.get("core_requires_tmax", True):
            core &= hypo
    if clean:
        hypo = _drop_small(hypo, ex["min_cluster_ml"] / voxel_ml)
        if core is not None:
            core = _drop_small(core, ex["min_cluster_ml"] / voxel_ml)
    tmax6 = float(hypo.sum()) * voxel_ml
    core_ml = float(core.sum()) * voxel_ml if core is not None else np.nan
    mismatch = tmax6 - core_ml if np.isfinite(core_ml) else np.nan
    return {
        "ctp_tmax6_ml": tmax6,
        "ctp_core_ml": core_ml,
        "ctp_mismatch_ml": mismatch,
        "ctp_mismatch_ratio": tmax6 / core_ml if np.isfinite(core_ml) and core_ml > 0 else np.nan,
        "ctp_mismatch_rel": 100.0 * mismatch / tmax6 if np.isfinite(mismatch) and tmax6 > 0 else np.nan,
        "reference_cbf": ref,
        "reference_voxels": n_ref,
        "valid_ml": float(valid.sum()) * voxel_ml,
    }


def _drop_small(mask: np.ndarray, min_voxels: float) -> np.ndarray:
    from scipy import ndimage

    if not mask.any():
        return mask
    lab, n = ndimage.label(mask, structure=np.ones((3,) * mask.ndim))
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_voxels
    keep[0] = False
    return keep[lab]


def load_map(path: Path) -> tuple[np.ndarray, float]:
    """A 3D map and its voxel volume in mL (from the header)."""
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path.name}: expected a 3D map, got shape {data.shape}")
    zooms = img.header.get_zooms()[:3]
    return data, float(np.prod(zooms)) / 1000.0


def lesion_ml(path: Path) -> float:
    m, vox = load_map(path)
    return float((m > 0.5).sum()) * vox


# ---------------------------------------------------------------- files

def _session(p: Path) -> int | None:
    m = re.search(r"ses-0*(\d+)", str(p))
    return int(m.group(1)) if m else None


def _case_id(p: Path) -> str | None:
    m = re.search(r"(sub-[A-Za-z0-9]+)", str(p))
    return m.group(1) if m else None


def _pick(files: list[Path], patterns: list[str]) -> Path | None:
    """First file whose name contains a pattern; files under raw_data/ are preferred."""
    ranked = sorted(files, key=lambda p: (0 if "raw_data" in p.parts else 1, str(p)))
    for pat in patterns:
        for p in ranked:
            if pat.lower() in p.name.lower():
                return p
    return None


def index(root: Path) -> dict[str, dict]:
    """Image files of every case: {case: {"ses1": [...], "ses2": [...]}}."""
    out: dict[str, dict] = {}
    for p in sorted(root.rglob("*")):
        if not (p.is_file() and p.name.lower().endswith(NII)):
            continue
        case, ses = _case_id(p), _session(p)
        if case is None or ses not in (1, 2):
            continue
        out.setdefault(case, {"ses1": [], "ses2": []})[f"ses{ses}"].append(p)
    return out


def extract(root: Path, cfg: dict | None = None) -> pd.DataFrame:
    """Perfusion volumes and follow-up infarct volume of every case (row-level; keep local)."""
    cfg = cfg or settings()
    ex, fp = cfg["extractor"], cfg["files"]
    rows = []
    for case, f in index(Path(root)).items():
        r = {"case": case, "error": ""}
        tmax_p = _pick([p for p in f["ses1"] if "msk" not in p.name.lower()], fp["tmax"])
        cbf_p = _pick([p for p in f["ses1"] if "msk" not in p.name.lower()], fp["cbf"])
        try:
            if tmax_p is None or cbf_p is None:
                raise FileNotFoundError("Tmax or CBF map not found")
            t, vox_t = load_map(tmax_p)
            c, vox_c = load_map(cbf_p)
            if not np.isclose(vox_t, vox_c, rtol=1e-4):
                raise ValueError("Tmax and CBF voxel sizes differ")
            r.update(volumes(t, c, vox_t, ex))
            r.update({f"{k}_clean": v for k, v in volumes(t, c, vox_t, ex, clean=True).items()
                      if k in ("ctp_tmax6_ml", "ctp_core_ml")})
            r["voxel_ml"] = vox_t
        except Exception as e:                     # noqa: BLE001 - recorded per case, not imputed
            r["error"] = f"{type(e).__name__}: {e}"
        les = _pick(f["ses2"], fp["followup_lesion"])
        r["final_infarct_ml"] = np.nan
        if les is not None:
            try:
                r["final_infarct_ml"] = lesion_ml(les)
            except Exception as e:                 # noqa: BLE001
                r["error"] = (r["error"] + "; " if r["error"] else "") + f"lesion: {e}"
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- clinical tables

def _read_table(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, dtype=str)
    head = p.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    sep = "\t" if p.suffix.lower() == ".tsv" else next((s for s in (",", ";", "\t") if head and s in head[0]), ",")
    return pd.read_csv(p, sep=sep, dtype=str, keep_default_na=False)


def clinical(root: Path, cfg: dict | None = None) -> pd.DataFrame:
    """The demographic and outcome tables joined by case (raw strings). Works with one file per
    case (case id in the file name) or with one combined table that has a case column."""
    cfg = cfg or settings()
    parts = {}
    for kind in ("demographics", "outcome"):
        pats = [s.lower() for s in cfg["files"][kind]]
        files = sorted(p for p in Path(root).rglob("*")
                       if p.is_file() and p.suffix.lower() in (".csv", ".tsv", ".xlsx", ".xls")
                       and any(s in p.name.lower() for s in pats))
        frames = []
        for p in files:
            d = _read_table(p)
            cid = _case_id(p)
            if cid is not None and len(d) == 1:
                d.insert(0, "case", cid)
            else:
                col = next((c for c in d.columns if _norm(c) in ("case", "caseid", "subject", "subjectid",
                                                                 "participantid", "id")), None)
                if col is None:
                    continue
                d = d.rename(columns={col: "case"})
                d["case"] = d["case"].map(lambda v: _case_id(Path(str(v))) or str(v))
            frames.append(d)
        parts[kind] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"case": []})
    out = parts["demographics"].merge(parts["outcome"], on="case", how="outer", suffixes=("", "_outcome"))
    return out.drop_duplicates("case").reset_index(drop=True)


def headers(root: Path, cfg: dict | None = None) -> dict:
    """Step 1 of the plan: column names and missing counts only. No values, no distributions."""
    d = clinical(root, cfg)
    miss = {c: int((d[c].astype(str).str.strip().isin(["", "nan", "NaN", "NA", "None"])).sum())
            for c in d.columns if c != "case"}
    ranges = []
    for case, f in list(index(Path(root)).items())[:5]:          # units check on the first 5 cases
        fp = (cfg or settings())["files"]
        tp = _pick([p for p in f["ses1"] if "msk" not in p.name.lower()], fp["tmax"])
        if tp is not None:
            t, _ = load_map(tp)
            t = t[np.isfinite(t) & (t > 0)]
            if t.size:
                ranges.append({"case": case, "tmax_p5_p50_p95": [round(float(q), 2) for q in np.percentile(t, [5, 50, 95])]})
    return {"cases": int(len(d)), "columns": list(miss), "missing": miss, "tmax_value_ranges": ranges}


def map_clinical(d: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Rename clinical columns to the MTOP template by the fixed mapping. Values stay raw strings;
    the network's template reader (fed/schema.py) parses and range-checks them."""
    cfg = cfg or settings()
    by_norm = {_norm(c): c for c in d.columns if c != "case"}
    out = pd.DataFrame({"case": d["case"].values})
    mapped = {}
    for key, cands in cfg["mapping"].items():
        col = next((by_norm[_norm(c)] for c in cands if _norm(c) in by_norm), None)
        if col is not None:
            out[key] = d[col].values
            mapped[key] = col
    used = set(mapped.values())
    return out, {"mapped": mapped, "unmapped_template_keys": [k for k in cfg["mapping"] if k not in mapped],
                 "unused_columns": [c for c in d.columns if c != "case" and c not in used]}


def build_site(vol: pd.DataFrame, clin: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Cases in the MTOP upload template with the perfusion volumes as optional future columns."""
    from .fed.schema import template_columns

    cfg = cfg or settings()
    keys = ["ctp_core_ml", "ctp_tmax6_ml", "ctp_mismatch_ml", "ctp_mismatch_ratio", "ctp_mismatch_rel"]
    v = vol[["case", "error"] + [k for k in keys if k in vol.columns]].copy()
    ok = v["error"].eq("") & v.get("ctp_tmax6_ml", pd.Series(np.nan, index=v.index)).notna()
    v["ctp_done"] = np.where(ok, "1", "")
    v["ctp_software"] = np.where(ok, cfg["dataset"]["software_label"], "")
    site = clin.merge(v.drop(columns="error"), on="case", how="left")
    for col in template_columns():
        if col not in site.columns:
            site[col] = ""
    for k in keys:
        if k in site.columns:
            site[k] = site[k].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    cols = template_columns() + ["ctp_done"] + [k for k in keys if k in site.columns] + ["ctp_software"]
    return site[cols].fillna("")


# ---------------------------------------------------------------- Aim A: agreement

def agreement(vol: pd.DataFrame, cfg: dict | None = None, core_col: str = "ctp_core_ml") -> dict:
    """Automated core against the follow-up infarct: Spearman correlation with a bootstrap CI and
    Bland-Altman difference (follow-up minus core). Aggregates only."""
    from scipy.stats import spearmanr

    cfg = cfg or settings()
    ag = cfg["agreement"]
    d = vol[[core_col, "final_infarct_ml"]].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(d)
    if n < 10:
        return {"n": n, "note": "fewer than 10 cases with both volumes"}
    x, y = d[core_col].to_numpy(), d["final_infarct_ml"].to_numpy()
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": n, "note": "one of the volumes is the same in every case; correlation not defined"}
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(ag["seed"])
    boots = []
    for _ in range(ag["bootstrap"]):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            boots.append(spearmanr(x[i], y[i]).statistic)
    diff = y - x
    sd = float(np.std(diff, ddof=1))
    return {
        "n": n,
        "spearman": round(rho, 3),
        "spearman_ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
        "core_median_ml": round(float(np.median(x)), 1),
        "followup_median_ml": round(float(np.median(y)), 1),
        "bland_altman_mean_ml": round(float(diff.mean()), 1),
        "bland_altman_loa_ml": [round(float(diff.mean() - 1.96 * sd), 1), round(float(diff.mean() + 1.96 * sd), 1)],
        "share_followup_larger": round(float((diff > 0).mean()), 3),
        "usable": bool(rho >= ag["usable_if_spearman_at_least"]),
    }


# ---------------------------------------------------------------- command line

def main(args) -> None:
    root = Path(args.root)
    cfg = settings()
    if args.headers_only:
        print(json.dumps(headers(root, cfg), indent=1, ensure_ascii=False))
        return
    out = Path(args.out) if args.out else paths.RAW / "isles24"
    out.mkdir(parents=True, exist_ok=True)
    vol = extract(root, cfg)
    vol.to_csv(out / "isles24_volumes.csv", index=False)
    summary = {
        "dataset": cfg["dataset"]["name"],
        "run_date": date.today().isoformat(),
        "cases_with_images": int(len(vol)),
        "cases_with_volumes": int((vol["error"] == "").sum()),
        "cases_with_core": int(pd.to_numeric(vol.get("ctp_core_ml"), errors="coerce").notna().sum())
        if "ctp_core_ml" in vol else 0,
        "cases_with_followup": int(vol["final_infarct_ml"].notna().sum()),
        "errors": int((vol["error"] != "").sum()),
        "agreement_primary": agreement(vol, cfg) if "ctp_core_ml" in vol else {},
        "agreement_clusters_removed": agreement(vol, cfg, "ctp_core_ml_clean") if "ctp_core_ml_clean" in vol else {},
    }
    if cfg.get("mapping_status") == "fixed":
        from .fed.schema import read_site_csv
        from .fed.site import SiteNode

        clin, rep = map_clinical(clinical(root, cfg), cfg)
        site = build_site(vol, clin, cfg)
        site.to_csv(out / "isles24_site.csv", index=False)
        df, mapping = read_site_csv(out / "isles24_site.csv")
        summary["node"] = {"mapping": rep, "template_report": mapping,
                           "eligible": SiteNode("ISLES24", df).describe()["stages"]}
    else:
        summary["node"] = "not built: config/isles24.yaml mapping_status is not 'fixed' (plan, section 5)"
    paths.PUBLIC.mkdir(parents=True, exist_ok=True)
    (paths.PUBLIC / "isles24_extractor.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                                       encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"row-level files in {out} (git-ignored); summary in {paths.PUBLIC / 'isles24_extractor.json'}")
