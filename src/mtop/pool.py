"""The reviewed pool of external validation results (artifacts/public/validation_pool.json).

Any centre can validate MTOP on its own data in the browser (Multicentre tab) and pool its own
files there; those stay in that browser. The results everyone sees in the pooled analysis come
only from this file, which the MTOP team updates after a data-sharing agreement and a check of the
submitted results file:

  python -m mtop pool check results.json                       what would be accepted, and why not
  python -m mtop pool add results.json --site "Hospital X" --agreement "DUA 2026-03" --reviewer "V. G."
  python -m mtop pool list

A results file is accepted only if it is a real-data run (not the synthetic demo), in the current
format, for a published model version, with aggregate numbers only. Any stage, outcome and model
with fewer than privacy.min_cell events or non-events is kept as "too few" without its metrics.
The git history of the file is the audit trail.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import re

from .config import settings
from .paths import PUBLIC

POOL_FILE = PUBLIC / "validation_pool.json"
FORMAT = "mtop-validation/2"
METRIC_KEYS = {"n", "events", "auroc", "auroc_ci", "se_logit", "brier", "citl", "slope", "calibration", "insufficient"}
TOP_KEYS = {"format", "model_version", "model_fingerprint", "site", "generated", "rows", "columns_found", "stages", "demo",
            "period", "development_period", "rows_after_development", "rows_with_year"}


def load_pool() -> dict:
    if POOL_FILE.exists():
        return json.loads(POOL_FILE.read_text(encoding="utf-8"))
    return {"format": "mtop-validation-pool/1", "updated": None, "results": []}


def current_fingerprint() -> str | None:
    f = PUBLIC / "model_bundle.json"
    return json.loads(f.read_text(encoding="utf-8")).get("fingerprint") if f.exists() else None


def check_result(j: dict, fingerprint: str | None = None, min_cell: int | None = None) -> tuple[dict, list[str], list[str]]:
    """(cleaned copy, errors, notes). Errors block acceptance; notes describe what was removed."""
    min_cell = int(settings()["privacy"]["min_cell"]) if min_cell is None else min_cell
    errors, notes = [], []
    if not isinstance(j, dict) or j.get("format") != FORMAT:
        return j, [f"not an MTOP results file in format {FORMAT}"], notes
    if j.get("demo"):
        errors.append("computed on the synthetic demo data")
    extra = set(j) - TOP_KEYS
    if extra:
        errors.append(f"unexpected fields (only aggregate results are accepted): {sorted(extra)}")
    fp = j.get("model_fingerprint")
    if not isinstance(fp, str) or not re.fullmatch(r"[0-9a-f]{64}", fp):
        errors.append("no model fingerprint: the model version cannot be identified")
    elif fingerprint and fp != fingerprint:
        notes.append("made with an earlier model version: it is pooled separately from the current version")
    stages = j.get("stages")
    if not isinstance(stages, dict) or not stages:
        errors.append("no stage results")
        return j, errors, notes
    out = copy.deepcopy(j)
    for k in ("rows_with_year", "rows_after_development"):
        v = out.get(k)
        if isinstance(v, int) and 0 < v < min_cell:
            out[k] = None
            notes.append(f"{k}: fewer than {min_cell}, not published")
    for st, tgs in out["stages"].items():
        for tg, r in tgs.items():
            models = r.get("models", {})
            for nm, m in list(models.items()):
                bad = set(m) - METRIC_KEYS
                if bad:
                    errors.append(f"{st}/{tg}/{nm}: unexpected fields {sorted(bad)}")
                if m.get("insufficient"):
                    continue
                n, ev = int(m.get("n", 0)), int(m.get("events", 0))
                if min(ev, n - ev) < min_cell:
                    models[nm] = {"n": n, "insufficient": True}
                    notes.append(f"{st}/{tg}/{nm}: fewer than {min_cell} events or non-events, kept without metrics")
                    continue
                a = m.get("auroc")
                ci = m.get("auroc_ci") or [None, None]
                if not (isinstance(a, (int, float)) and 0 <= a <= 1 and ci[0] is not None and ci[0] <= a <= ci[1]
                        and (m.get("se_logit") or 0) > 0):
                    errors.append(f"{st}/{tg}/{nm}: implausible AUROC or interval")
                cal = m.get("calibration") or []
                if any(int(b.get("n", 0)) < min_cell for b in cal):
                    m["calibration"] = [b for b in cal if int(b.get("n", 0)) >= min_cell]
                    notes.append(f"{st}/{tg}/{nm}: calibration groups under {min_cell} patients removed")
            ch = r.get("chosen")
            for k in list(r):
                if k not in ("chosen", "models"):
                    del r[k]
            if ch in models:
                r.update(models[ch])
    return out, errors, notes


def add_result(path, site: str, agreement: str, reviewer: str, note: str = "") -> dict:
    if not (site and agreement and reviewer):
        raise SystemExit("--site, --agreement and --reviewer are required")
    j = json.loads(open(path, encoding="utf-8").read())
    clean, errors, notes = check_result(j, current_fingerprint())
    if errors:
        raise SystemExit("not accepted:\n  " + "\n  ".join(errors))
    clean["site"] = site
    pool = load_pool()
    entry = {"site": site, "agreement": agreement, "reviewer": reviewer, "accepted": dt.date.today().isoformat(),
             "note": note, "result": clean}
    keep = [e for e in pool["results"] if not (e["site"] == site and e["result"].get("model_fingerprint") == clean.get("model_fingerprint"))]
    replaced = len(keep) != len(pool["results"])
    pool["results"] = keep + [entry]
    pool["updated"] = dt.date.today().isoformat()
    POOL_FILE.write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")
    for n in notes:
        print(f"  note: {n}")
    print(f"pool: {'replaced' if replaced else 'added'} {site}; {len(pool['results'])} reviewed results. "
          "Rebuild the site (python -m mtop site) and commit artifacts/public/validation_pool.json.")
    return entry


def main(args) -> None:
    if args.action == "list":
        pool = load_pool()
        fp = current_fingerprint()
        for e in pool["results"]:
            cur = "current model" if e["result"].get("model_fingerprint") == fp else "earlier model"
            print(f"{e['site']}: accepted {e['accepted']} by {e['reviewer']} ({e['agreement']}), {cur}, "
                  f"{e['result'].get('rows')} rows, period {e['result'].get('period')}")
        print(f"{len(pool['results'])} reviewed results")
        return
    if not args.file:
        raise SystemExit("give the results file")
    if args.action == "check":
        _, errors, notes = check_result(json.loads(open(args.file, encoding="utf-8").read()), current_fingerprint())
        for n in notes:
            print(f"note: {n}")
        print("acceptable" if not errors else "not acceptable:\n  " + "\n  ".join(errors))
        return
    add_result(args.file, args.site, args.agreement, args.reviewer, args.note or "")
