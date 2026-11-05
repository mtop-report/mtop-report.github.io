"""The published MR PREDICTS formulas, transcribed from the R code in their supplements.

* MR PREDICTS, updated version (Venema et al., Stroke 2021;52:2764-72, Data Supplement "MR PREDICTS
  regression equation"): ordinal model of the 90-day mRS before treatment. Its coefficients are those
  of the 2021 update in HERMES (they reproduce the updated odds ratios of Supplemental Table II); the
  intercepts are given for HERMES and re-estimated for the MR CLEAN Registry.
* MR PREDICTS@24H (Chalos et al., JAMA Neurol 2023;80:940-8, Supplement 1, eAppendix): ordinal model
  24 hours after treatment. Its coefficients reproduce the effects measured in the MR CLEAN Registry
  (eTable, validation column); intercepts for HERMES and the Registry (the online tool uses the latter).

Both give P(mRS <= k) = plogis(lp + intercept_k) for k = 0..5; mRS 0-2 is k = 2 and survival (mRS 0-5)
is k = 5. Restricted cubic splines are written as in the R code: a linear term plus truncated cubes.
In the FNO registry the collateral score is not recorded, so the app's Validation tab refits the two
predictor sets without it (scores.py); these functions are the published models themselves.
"""
from __future__ import annotations

import numpy as np

INTERCEPTS = {
    "mr_predicts": {
        "hermes": [-1.35209, -0.0966187, 0.854397, 1.68040, 2.86330, 3.52439],
        "registry": [-1.95789, -0.423007, 0.704584, 1.47090, 2.16359, 2.52240],
    },
    "mr_predicts_24h": {
        "hermes": [-1.49640, 0.0655256, 1.46388, 2.79135, 4.19445, 4.86642],
        "registry": [-1.60096, 0.186484, 1.69889, 2.81098, 3.80616, 4.28056],
    },
}
LOCATIONS = ("ICA-(T)", "M1", "M2")
GLUCOSE_MG_PER_MMOL = 18.016
COLLATERALS_MISSING_24H = 0.198023      # MR PREDICTS@24H: used instead of 0.115803 * score when missing


def _c3(x, k):
    return np.maximum(np.asarray(x, float) - k, 0.0) ** 3


def _a(x):
    return np.asarray(x, float)


def lp_mr_predicts(evt, age, nihss, premrs, diabetes, sbp, ivt, glucose_mgdl, aspects, location, collaterals,
                   onset_to_groin):
    """Linear predictor of MR PREDICTS (updated). location in LOCATIONS; glucose in mg/dL; times in minutes."""
    evt, age, sbp, glu, coll, otg = map(_a, (evt, age, sbp, glucose_mgdl, collaterals, onset_to_groin))
    loc = np.asarray(location, dtype=object)
    return (0.401368 * evt
            - 0.0118443 * age - 1.92454e-05 * _c3(age, 46) + 4.98209e-05 * _c3(age, 67.5) - 3.05755e-05 * _c3(age, 81)
            - 0.0737296 * _a(nihss) - 0.427549 * _a(premrs) - 0.458441 * _a(diabetes)
            + 0.00124704 * sbp - 3.38789e-06 * _c3(sbp, 115) + 6.45817e-06 * _c3(sbp, 144) - 3.07028e-06 * _c3(sbp, 176)
            + 0.0947369 * _a(ivt)
            - 0.00469116 * glu + 4.51697e-07 * _c3(glu, 94.5) - 7.22715e-07 * _c3(glu, 121.8)
            + 2.71018e-07 * _c3(glu, 167.3)
            + 0.122167 * _a(aspects) + 0.501521 * (loc == "M1") + 0.883908 * (loc == "M2")
            + 0.258226 * coll + 0.230021 * evt * coll
            - 0.00213380 * otg - 0.00116301 * evt * otg)


def lp_mr_predicts_24h(age, nihss, diabetes, premrs, location, collaterals, etici, nihss_24h, sich):
    """Linear predictor of MR PREDICTS@24H. premrs above 2 counts as 3; eTICI 0..5 = 0, 1, 2a, 2b, 2c, 3;
    a missing collateral score uses the published constant."""
    age, n24, coll = map(_a, (age, nihss_24h, collaterals))
    loc = np.asarray(location, dtype=object)
    coll_term = np.where(np.isnan(coll), COLLATERALS_MISSING_24H, 0.115803 * np.nan_to_num(coll))
    return (0.00907192 * age - 4.83914e-05 * _c3(age, 50.5) + 0.000118985 * _c3(age, 72) - 7.05932e-05 * _c3(age, 86.5)
            - 0.00333532 * _a(nihss) - 0.563654 * _a(diabetes) - 0.508099 * np.minimum(_a(premrs), 3)
            + 0.154629 * (loc == "M1") + 0.282501 * (loc == "M2")
            + coll_term + 0.0793556 * _a(etici)
            - 0.252124 * n24 + 0.000148483 * _c3(n24, 1) - 0.000272219 * _c3(n24, 11) + 0.000123736 * _c3(n24, 23)
            - 1.68195 * _a(sich))


def cumulative(lp, model: str, intercepts: str = "registry") -> np.ndarray:
    """P(mRS <= k) for k = 0..5, one column per cut-point."""
    a = np.asarray(INTERCEPTS[model][intercepts])
    return 1.0 / (1.0 + np.exp(-(np.asarray(lp, float)[:, None] + a[None, :])))


def outcomes(lp, model: str, intercepts: str = "registry") -> dict:
    """Probabilities of mRS 0-2 and of death (mRS 6) at 90 days."""
    c = cumulative(lp, model, intercepts)
    return {"good_outcome": c[:, 2], "death90": 1.0 - c[:, 5]}
