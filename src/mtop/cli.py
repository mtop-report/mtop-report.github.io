"""Command-line entry point: `python -m mtop <command>`.

  clean     raw Excel -> data/processed/clean.parquet + reports/data_quality.xlsx
  train     tune, validate and fit all stage models (artifacts/private)
  export    browser model + validation summary (artifacts/public)
  explorer  suppressed aggregate statistics for the cohort explorer (artifacts/public)
  report    Table 1, figures and a validation summary for the manuscript (reports/)
  site      build the self-contained web app (site/index.html)
  all       everything above, in order
  synth     write a synthetic registry with the same schema (data/synthetic)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from . import paths


def _load_clean() -> pd.DataFrame:
    f = paths.PROCESSED / "clean.parquet"
    if not f.exists():
        raise SystemExit("Run `python -m mtop clean` first.")
    return pd.read_parquet(f)


def cmd_clean(args):
    from .clean import clean_registry, load_raw
    from .cohort import flow

    src = Path(args.input) if args.input else paths.DEFAULT_RAW_FILE
    if not src.exists():
        raise SystemExit(f"Raw file not found: {src}. Put the registry export in data/raw/ "
                         "or pass --input (use --input data/synthetic/registry_synthetic.xlsx to test).")
    clean, issues = clean_registry(load_raw(src))
    clean.to_parquet(paths.PROCESSED / "clean.parquet", index=False)
    with pd.ExcelWriter(paths.REPORTS / "data_quality.xlsx") as xw:
        summary = issues.groupby(["column", "action"]).size().rename("n").reset_index()
        summary.to_excel(xw, sheet_name="summary", index=False)
        issues.to_excel(xw, sheet_name="issues", index=False)
    _, steps = flow(clean)
    print(f"clean: {len(clean)} rows, {len(issues)} logged issues")
    for s in steps:
        print(f"  {s['step']:<36} {s['n']:>5}")


def cmd_train(args):
    from .cohort import flow
    from .train import run

    base, _ = flow(_load_clean())
    run(base, paths.PRIVATE, quick=args.quick)


def cmd_export(args):
    from .export import export_all

    export_all()


def cmd_explorer(args):
    from .explorer import build_explorer

    build_explorer(_load_clean())


def cmd_report(args):
    from .report import build_report

    build_report(_load_clean())


def cmd_site(args):
    from .site import build_site

    out = build_site()
    print(f"site: {out} ({out.stat().st_size / 1e6:.2f} MB)")


def cmd_synth(args):
    from .synth import write_synthetic

    out = write_synthetic(n=args.n, seed=args.seed)
    print(f"synthetic registry: {out}")


def cmd_all(args):
    t = time.time()
    cmd_clean(args)
    cmd_train(args)
    cmd_export(args)
    cmd_explorer(args)
    cmd_report(args)
    cmd_site(args)
    print(f"all done in {time.time() - t:.0f}s")


def main(argv=None):
    paths.ensure_dirs()
    ap = argparse.ArgumentParser(prog="mtop", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("clean", cmd_clean), ("train", cmd_train), ("export", cmd_export),
                     ("explorer", cmd_explorer), ("report", cmd_report), ("site", cmd_site),
                     ("all", cmd_all), ("synth", cmd_synth)]:
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if name in ("clean", "all"):
            p.add_argument("--input", help="registry Excel file (default: data/raw/mt_patients_master.xlsx)")
        if name in ("train", "all"):
            p.add_argument("--quick", action="store_true", help="few tuning trials, for testing")
        if name == "synth":
            p.add_argument("--n", type=int, default=600)
            p.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
