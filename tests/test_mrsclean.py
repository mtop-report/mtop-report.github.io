"""MRS CLEAN -> MTOP template conversion, on hand-made rows (no MRS CLEAN data in the repository)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtop.fed.schema import read_site_csv  # noqa: E402
from mtop.mrsclean import NOT_RECORDED, SOURCE_COLUMNS, convert  # noqa: E402


def _rows() -> pd.DataFrame:
    base = {c: 0 for c in SOURCE_COLUMNS}
    base.update({"r_age": 70, "bl_NIHSS_sumscore": 15, "bl_phy_rrsyst": 150, "bl_phy_rrdiast": 80, "bl_lab_glucose": 7,
                 "bl_ASPECTS": 9, "TI.onsetlsw_to_groin": 180, "TI.groin_to_endproc": 50, "fu24h_NIHSS_sumscore": 8,
                 "occlside_LeftRight": 1, "r_ivt": 1, "mrs_90d": 2, "collateral_grade": 2})
    rows = []
    for i in range(6):
        r = dict(base, StudySubjectID=f"T{i}")
        r["bl_cta_prox_occl_loc_CatSAP"] = [0, 1, 2, 3, 4, 3][i]
        r["postetici_c"] = i                                   # 0, 1, 2a, 2b, 2c, 3
        r["deepest_anesthmanag"] = [0, 1, 2, 3, 4, 0][i]
        r["occlside_LeftRight"] = [1, 2, 3, 4, 1, 2][i]
        r["r_ivt"] = [0, 1, 9, 1, 0, 1][i]
        r["bl_hist_smoking"] = [0, 1, 2, 0, 1, 2][i]
        r["bl_hist_hep"], r["bl_hist_apt"] = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 0), (0, 1)][i]
        r["r_sex"] = i % 2
        r["mrs_90d"] = i
        rows.append(r)
    rows[5]["TI.onsetlsw_to_groin"] = 2000                     # beyond the app's range -> unknown
    return pd.DataFrame(rows)


def _read(tmp_path, t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    f = tmp_path / "t.csv"
    t.to_csv(f, index=False)
    return read_site_csv(f)


def test_primary_mapping_as_the_app_reads_it(tmp_path):
    t, side = convert(_rows())
    df, rep = _read(tmp_path, t)
    assert rep["columns_missing"] == NOT_RECORDED
    # occlusion: 0 other (unknown), 1 ICA and 2 ICA-T -> intracranial ICA, 3 M1, 4 M2
    assert df["occlusion"].tolist() == [None, "ICA", "ICA", "M1", "M2", "M1"]
    assert side["occlusion_detail"].tolist()[:5] == ["other", "ICA", "ICA-T", "M1", "M2"]
    # eTICI 0..5 = 0, 1, 2a, 2b, 2c, 3; the app's reader counts 2c as 3 (4.0)
    assert df["tici"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 4.0]
    assert df["anesthesia"].tolist() == ["sedation", "general", None, None, None, "sedation"]
    assert df["side_left"].fillna(-1).tolist() == [1.0, 0.0, -1, -1, 1.0, 0.0]   # 3 both, 4 unknown -> unknown
    assert np.isnan(df["ivt"][2]) and df["ivt"][1] == 1.0
    assert df["smoking"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]           # current smokers only
    assert df["antithrombotic"].tolist() == ["none", "antiplatelet", "anticoagulant", "anticoagulant", "none", "antiplatelet"]
    assert df["sex_male"].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]          # r_sex 0 = male
    assert np.isnan(df["onset_to_puncture"][5]) and rep["unrecognised_values"]["onset_to_puncture"] == 1
    assert df["onset_to_recan"][0] == 230.0
    assert df["death90"].tolist() == [0, 0, 0, 0, 0, 0] and df["good_outcome"].tolist() == [1, 1, 1, 0, 0, 0]
    assert rep["future_recorded"] == {"collaterals_tan": 6}


def test_sensitivity_variants_change_only_their_input(tmp_path):
    src = _rows()
    a, _ = _read(tmp_path, convert(src)[0])
    for kw, key in (({"tici_2c": "2b"}, "tici"), ({"smoker": "ever"}, "smoking"), ({"antithrombotic_none": ""}, "antithrombotic")):
        b, _ = _read(tmp_path, convert(src, **kw)[0])
        changed = [c for c in a.columns if not ((a[c] == b[c]) | (a[c].isna() & b[c].isna())).all()]
        assert changed == [key], (kw, changed)
    b, _ = _read(tmp_path, convert(src, tici_2c="2b")[0])
    assert b["tici"][4] == 3.0
