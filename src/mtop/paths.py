"""Central place for repository paths.

Outputs go under the repository by default. Set MTOP_OUT to send all generated files
(processed data, artifacts, reports, site) somewhere else, e.g. a temp folder in CI so a
synthetic test run never overwrites the published model.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("MTOP_OUT", ROOT)).resolve()

CONFIG = ROOT / "config"
WEB = ROOT / "web"
RAW = ROOT / "data" / "raw"
SYNTHETIC = ROOT / "data" / "synthetic"    # fake data, safe to commit

DATA = OUT / "data"
PROCESSED = DATA / "processed"             # row-level, never committed
ARTIFACTS = OUT / "artifacts"
PUBLIC = ARTIFACTS / "public"              # aggregate-only outputs that may be published
PRIVATE = ARTIFACTS / "private"            # row-level model outputs, never committed
REPORTS = OUT / "reports"                  # local reports (may contain patient IDs)
SITE = OUT / "site"                        # built static site (GitHub Pages)

DEFAULT_RAW_FILE = RAW / "mt_patients_master.xlsx"


def ensure_dirs() -> None:
    for p in (PROCESSED, PUBLIC, PRIVATE, REPORTS, SITE):
        p.mkdir(parents=True, exist_ok=True)
