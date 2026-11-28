"""Four-centre network simulation (plan docs/registrations/v2_imaging_plan.md, amendment 1).
Synthetic centres only; no registry, MRS CLEAN or ISLES'24 data."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cryptography")

from mtop.fed import demo_multi as DM  # noqa: E402
from mtop.fed.schema import write_site_csv  # noqa: E402
from test_federated import synthetic_frame  # noqa: E402

ISLES_LIKE = ["age", "sex_male", "mrs_pre", "nihss_adm", "ivt", "htn", "dm", "arrhythmia", "tici", "mrs90"]


def _raw(df: pd.DataFrame, tmp_path, name: str) -> pd.DataFrame:
    f = tmp_path / f"{name}.csv"
    write_site_csv(df, f)
    return pd.read_csv(f, dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("multi")
    fno = _raw(synthetic_frame(400, 501), tmp, "fno")
    mrs = _raw(synthetic_frame(900, 502), tmp, "mrs")
    isl = synthetic_frame(150, 503)
    keep = isl[ISLES_LIKE].copy()
    for c in set(isl.columns) - set(ISLES_LIKE):
        keep[c] = np.nan if isl[c].dtype.kind == "f" else None
    isl_raw = _raw(keep[isl.columns], tmp, "isles")
    isl_raw["ctp_core_ml"] = [f"{v:.1f}" for v in np.random.default_rng(1).gamma(2, 10, len(isl_raw))]
    out = {"Ostrava": fno}
    out.update(DM.split_mrs(mrs, n=300))
    out["ISLES24"] = isl_raw
    return out


def test_mr_clean_split_is_disjoint_and_fixed():
    df = pd.DataFrame({"x": range(1000)})
    a = DM.split_mrs(df, n=300)
    b = DM.split_mrs(df, n=300)
    assert a["MRCLEAN_A"].equals(b["MRCLEAN_A"])
    assert not set(a["MRCLEAN_A"]["x"]) & set(a["MRCLEAN_B"]["x"])
    with pytest.raises(ValueError):
        DM.split_mrs(df, n=600)


def test_suppression_blanks_small_groups():
    out = {"centres": [{"id": "X", "good_outcome": {"n": 40, "events": 5, "rate": 0.1}, "death90": None}],
           "outcomes": {"good_outcome": {"per_centre": [{"id": "X", "n": 40, "events": 5, "alone": {"auroc": 0.7}}],
                                         "loso": [], "curves": {"X": [1]}, "growth": {"X": [1]}}},
           "stages": []}
    DM.suppress(out, 10)
    g = out["outcomes"]["good_outcome"]["per_centre"][0]
    assert g["suppressed"] and g["alone"] is None and g["n"] is None
    assert out["outcomes"]["good_outcome"]["curves"]["X"] is None
    DM.check_public(out, 10)


def test_four_centre_run_end_to_end(frames, tmp_path):
    pub = tmp_path / "federated_multi.json"
    out = DM.run(frames, http=False, quick=True, out_dir=tmp_path / "rows", public_path=pub, trees=6)
    assert [c["id"] for c in out["centres"]] == ["Ostrava", "MRCLEAN_A", "MRCLEAN_B", "ISLES24"]
    isl = next(c for c in out["centres"] if c["id"] == "ISLES24")
    fno = next(c for c in out["centres"] if c["id"] == "Ostrava")
    assert isl["recorded_half_or_more"] < fno["recorded_half_or_more"]
    assert isl["perfusion_recorded"] == 150
    adm = out["outcomes"]["good_outcome"]
    assert set(adm["centres_used"]) == {"Ostrava", "MRCLEAN_A", "MRCLEAN_B", "ISLES24"}
    assert out["equivalence"]["gbdt"]["identical"]
    h24 = next(s for s in out["stages"] if s["stage"] == "h24")
    assert "ISLES24" not in h24["centres_used"]            # no 24-hour NIHSS recorded
    saved = json.loads(pub.read_text())
    assert saved["kind"] == "simulation, software test"
