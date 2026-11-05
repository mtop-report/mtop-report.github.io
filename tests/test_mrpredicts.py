"""The MR PREDICTS formulas in mrpredicts.py against the published R code and tables.

The strings below are the supplements' R code and intercept tables as extracted from the PDFs
(Venema et al., Stroke 2021, Data Supplement; Chalos et al., JAMA Neurol 2023, Supplement 1), line
breaks joined. They are evaluated directly, so a transcription slip in mrpredicts.py fails the test.
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtop import mrpredicts as M  # noqa: E402

R_MR_PREDICTS = 'plogis(0.401368*EVT - 0.0118443*age - 1.92454e-05*pmax(age-46,0)^3 + 4.98209e-05*pmax(age- 67.5,0)^3 - 3.05755e-05*pmax(age-81,0)^3 - 0.0737296*NIHSS - 0.427549*premrs - 0.458441*diabetes + 0.00124704** BP - 3.38789e-06*pmax(BP-115,0)^3 + 6.45817e-06*pmax(BP- 144,0)^3 - 3.07028e-06* pmax(BP-176,0)^3 + 0.0947369*IVT - 0.00469116*glucose + 4.51697e- 07*pmax(glucose- 94.5,0)^3 - 7.22715e-07*pmax(glucose-121.8,0)^3 + 2.71018e-07* pmax(glucose-167.3,0)^3 + 0.122167*ASPECTS + 0.501521*(location=="M1") + 0.883908* (location=="M2") + 0.258226*collaterals + (0.230021*EVT*collaterals) - 0.00213380*onsettogroin - (0.00116301*EVT*onsettogroin) + intercept)'
R_MR_PREDICTS_24H = 'plogis((+0.00907192*age - 4.83914e-05*pmax(age-50.5,0)^3 + 0.000118985*pmax(age-72,0)^3 - 7.05932e-05*pmax(age-86.5,0)^3 - 0.00333532*NIHSS baseline - 0.563654*diabetes mellitus - 0.508099*pre-mRS + 0.154629*(occlusion location=="M1") + 0.282501*(occlusion location=="M2") + 0.115803*collateral scoreb + 0.0793556*eTICI - 0.252124*NIHSS24h + 0.000148483*pmax(NIHSS24h-1,0)^3 - 0.000272219*pmax(NIHSS24h-11,0)^3 + 0.000123736*pmax(NIHSS24h-23,0)^3 - 1.68195*sICH) + interceptc)'
INTERCEPT_ROWS_MR_PREDICTS = ['mRS 0                  -1.35209                -1.95789', 'mRS 0-1              -0.0966187               -0.423007', 'mRS 0-2               0.854397                 0.704584', 'mRS 0-3                1.68040                  1.47090', 'mRS 0-4                2.86330                  2.16359', 'mRS 0-5                3.52439                  2.52240']
INTERCEPT_ROWS_MR_PREDICTS_24H = ['mRS 0 vs 1-6                    -1.49640                   -1.60096', 'mRS 0-1 vs 2-6                 0.0655256                  0.186484', 'mRS 0-2 vs 3-6                   1.46388                   1.69889', 'mRS 0-3 vs 4-6                   2.79135                   2.81098', 'mRS 0-4 vs 5-6                   4.19445                   3.80616', 'mRS 0-5 vs 6                     4.86642                   4.28056']


def _py(r: str) -> str:
    """The R expression as Python: PDF line breaks inside numbers, one typo (** for *), names with spaces."""
    s = r.replace("e- ", "e-").replace("** BP", "* BP").replace("^", "**")
    for a, b in (("NIHSS baseline", "NIHSS_baseline"), ("diabetes mellitus", "diabetes_mellitus"), ("pre-mRS", "pre_mRS"),
                 ("occlusion location", "occlusion_location"), ("collateral scoreb", "collateral_score"),
                 ("interceptc", "intercept")):
        s = s.replace(a, b)
    return s


def _env(**kw):
    return {"plogis": lambda z: 1 / (1 + np.exp(-z)), "pmax": np.maximum, **kw}


def _patients(n=400, seed=1):
    rng = np.random.default_rng(seed)
    return {"age": rng.uniform(18, 100, n), "nihss": rng.integers(0, 43, n).astype(float),
            "premrs": rng.integers(0, 6, n).astype(float), "diabetes": rng.integers(0, 2, n).astype(float),
            "sbp": rng.uniform(80, 250, n), "ivt": rng.integers(0, 2, n).astype(float), "glucose": rng.uniform(50, 500, n),
            "aspects": rng.integers(0, 11, n).astype(float), "location": rng.choice(list(M.LOCATIONS), n).astype(object),
            "coll": rng.integers(0, 4, n).astype(float), "otg": rng.uniform(30, 1440, n), "etici": rng.integers(0, 6, n).astype(float),
            "n24": rng.integers(0, 43, n).astype(float), "sich": rng.integers(0, 2, n).astype(float)}


def _rows(rows):
    return [[float(v) for v in re.findall(r"-?\d+\.\d+", r)[-2:]] for r in rows]


def test_intercepts_match_the_published_tables():
    for model, rows in (("mr_predicts", INTERCEPT_ROWS_MR_PREDICTS), ("mr_predicts_24h", INTERCEPT_ROWS_MR_PREDICTS_24H)):
        tab = _rows(rows)
        assert M.INTERCEPTS[model]["hermes"] == [r[0] for r in tab]
        assert M.INTERCEPTS[model]["registry"] == [r[1] for r in tab]


def test_mr_predicts_matches_the_published_r_code():
    p = _patients()
    expr = _py(R_MR_PREDICTS)
    for evt in (0.0, 1.0):
        lp = M.lp_mr_predicts(evt, p["age"], p["nihss"], p["premrs"], p["diabetes"], p["sbp"], p["ivt"], p["glucose"],
                              p["aspects"], p["location"], p["coll"], p["otg"])
        for which in ("hermes", "registry"):
            for k, a in enumerate(M.INTERCEPTS["mr_predicts"][which]):
                ref = eval(expr, _env(EVT=evt, age=p["age"], NIHSS=p["nihss"], premrs=p["premrs"], diabetes=p["diabetes"],
                                      BP=p["sbp"], IVT=p["ivt"], glucose=p["glucose"], ASPECTS=p["aspects"],
                                      location=p["location"], collaterals=p["coll"], onsettogroin=p["otg"], intercept=a))
                assert np.allclose(M.cumulative(lp, "mr_predicts", which)[:, k], ref, rtol=0, atol=1e-12)


def test_mr_predicts_24h_matches_the_published_r_code():
    p = _patients(seed=2)
    premrs3 = np.minimum(p["premrs"], 3)                      # "Pre-mRS: 0=0, 1=1, 2=2, >2=3"
    lp = M.lp_mr_predicts_24h(p["age"], p["nihss"], p["diabetes"], p["premrs"], p["location"], p["coll"], p["etici"],
                              p["n24"], p["sich"])
    expr = _py(R_MR_PREDICTS_24H)
    for which in ("hermes", "registry"):
        for k, a in enumerate(M.INTERCEPTS["mr_predicts_24h"][which]):
            ref = eval(expr, _env(age=p["age"], NIHSS_baseline=p["nihss"], diabetes_mellitus=p["diabetes"], pre_mRS=premrs3,
                                  occlusion_location=p["location"], collateral_score=p["coll"], eTICI=p["etici"],
                                  NIHSS24h=p["n24"], sICH=p["sich"], intercept=a))
            assert np.allclose(M.cumulative(lp, "mr_predicts_24h", which)[:, k], ref, rtol=0, atol=1e-12)


def test_missing_collaterals_use_the_published_constant():
    lp_nan = M.lp_mr_predicts_24h([70], [15], [0], [0], np.array(["M1"], object), [np.nan], [5], [8], [0])
    lp_0 = M.lp_mr_predicts_24h([70], [15], [0], [0], np.array(["M1"], object), [0], [5], [8], [0])
    assert np.isclose(lp_nan - lp_0, 0.198023).all()


def test_coefficients_reproduce_the_published_odds_ratios():
    """exp(coefficient) against the printed odds ratios (two decimals, so within 0.011):
    MR PREDICTS = Supplemental Table II, updated model; MR PREDICTS@24H = eTable, MR CLEAN Registry
    column (the HERMES column differs widely, e.g. M2 2.03 and sICH 0.28)."""
    ica, m1, m2 = (np.array([v], object) for v in M.LOCATIONS)
    base = dict(evt=1, age=60, nihss=10, premrs=0, diabetes=0, sbp=120, ivt=0, glucose_mgdl=100, aspects=8,
                location=ica, collaterals=0, onset_to_groin=0)

    def mrp(**chg):
        return float(np.asarray(M.lp_mr_predicts(**{**base, **chg})).ravel()[0])

    published = [({"nihss": 11}, 0.93), ({"premrs": 1}, 0.65), ({"diabetes": 1}, 0.63), ({"ivt": 1}, 1.10),
                 ({"aspects": 9}, 1.13), ({"location": m1}, 1.65), ({"location": m2}, 2.42)]
    for chg, odds in published:
        assert abs(np.exp(mrp(**chg) - mrp()) - odds) <= 0.011, (chg, odds)
    # collaterals and time: main effect (no treatment) and interaction with treatment
    assert abs(np.exp(mrp(evt=0, collaterals=1) - mrp(evt=0)) - 1.29) <= 0.011
    assert abs(np.exp((mrp(collaterals=1) - mrp()) - (mrp(evt=0, collaterals=1) - mrp(evt=0))) - 1.26) <= 0.011
    assert abs(np.exp(mrp(evt=0, onset_to_groin=30) - mrp(evt=0)) - 0.94) <= 0.011
    assert abs(np.exp(mrp(evt=1, onset_to_groin=0) - mrp(evt=0, onset_to_groin=0)) - 1.49) <= 0.011

    base24 = dict(age=40, nihss=10, diabetes=0, premrs=0, location=ica, collaterals=0, etici=0, nihss_24h=0, sich=0)

    def m24(**chg):
        return float(np.asarray(M.lp_mr_predicts_24h(**{**base24, **chg})).ravel()[0])

    registry = [({"age": 41}, 1.01), ({"nihss": 11}, 1.00), ({"diabetes": 1}, 0.57), ({"premrs": 1}, 0.60),
                ({"location": m1}, 1.16), ({"location": m2}, 1.32), ({"collaterals": 1}, 1.12), ({"sich": 1}, 0.19)]
    for chg, odds in registry:
        assert abs(np.exp(m24(**chg) - m24()) - odds) <= 0.011, (chg, odds)
