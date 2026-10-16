"""End-to-end run on synthetic data in a temporary output folder (used by CI).

Skipped by default because it takes a few minutes; run with MTOP_E2E=1.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.environ.get("MTOP_E2E") != "1", reason="set MTOP_E2E=1 to run the end-to-end pipeline")
def test_pipeline_on_synthetic(tmp_path):
    env = dict(os.environ, MTOP_OUT=str(tmp_path), PYTHONPATH=str(ROOT / "src"))
    synth = ROOT / "data" / "synthetic" / "registry_synthetic.xlsx"
    if not synth.exists():
        subprocess.run([sys.executable, "-m", "mtop", "synth"], check=True, env=env, cwd=ROOT)
    subprocess.run([sys.executable, "-m", "mtop", "all", "--quick", "--input", str(synth)], check=True, env=env, cwd=ROOT)
    bundle = json.loads((tmp_path / "artifacts" / "public" / "model_bundle.json").read_text())
    assert set(bundle["models"]) == {"admission", "procedure", "h24", "d7"}
    html = (tmp_path / "site" / "index.html").read_text()
    assert html.startswith("<!doctype html>") and "MTOP" in html
