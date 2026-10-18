"""The model-choice rule (amendment 1): Holm's step-down on the temporal criterion."""
from __future__ import annotations

from mtop.train import holm, select_models


def test_holm_step_down():
    r = holm({"a": (0.02, 0.001), "b": (0.01, 0.02), "c": (-0.01, 0.003), "d": (0.03, 0.04)}, 0.05)
    assert [round(r[k]["p_holm"], 4) for k in "acbd"] == [0.004, 0.009, 0.04, 0.04]
    assert r["a"]["pass"] and r["b"]["pass"] and r["d"]["pass"]
    assert not r["c"]["pass"]                                   # a negative difference never passes
    assert not holm({"a": (0.02, 0.03), "b": (0.02, 0.03)}, 0.05)["a"]["pass"]   # 2 x 0.03 > 0.05


def _models(deltas):
    ref = "Logistic regression (splines)"
    m = {ref: {"auroc": 0.80, "citl": -0.4, "ci": {}}}
    for name, (d, p) in deltas.items():
        m[name] = {"auroc": 0.80 + d, "citl": -0.4,
                   "ci": {"delta_auroc_vs_ref": {"mean": d, "p_two_sided": p, "ci": [d - 0.01, d + 0.01]}}}
    return {"admission": {"targets": {"good_outcome": {"models": m}}}}


def test_an_unadjusted_win_is_not_enough_with_seven_candidates():
    names = ["Logistic regression", "LightGBM", "XGBoost", "EBM", "LR + interactions", "LR + multiple imputation",
             "Stacking"]
    deltas = {n: (-0.005, 0.5) for n in names}
    deltas["EBM"] = (0.02, 0.02)            # 95% interval above zero, but 7 x 0.02 > 0.05
    sel = select_models(_models(deltas), None)["admission"]["good_outcome"]
    assert sel["chosen"] == "Logistic regression (splines)"
    assert sel["candidates"]["EBM"]["p_holm"] > 0.05 and not sel["candidates"]["EBM"]["test_rule"]
    deltas["EBM"] = (0.03, 0.001)           # 7 x 0.001 < 0.05
    sel = select_models(_models(deltas), None)["admission"]["good_outcome"]
    assert sel["chosen"] == "EBM"
