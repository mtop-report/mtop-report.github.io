"""Averaging values for unknown inputs (export.imputation_values), on a hand-made registry."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mtop.config import features  # noqa: E402
from mtop.export import IMPUTE_K, imputation_values  # noqa: E402


def _registry(n=400, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = {"year": rng.integers(2010, 2026, n)}
    for f in features():
        if f.type == "derived":
            continue
        if f.type == "numeric":
            d[f.key] = rng.uniform(f.spec.get("min", 0), f.spec.get("max", 100), n)
        elif f.type == "binary":
            d[f.key] = rng.integers(0, 2, n).astype(float)
        else:
            vals = [o["value"] for o in f.spec["options"]]
            d[f.key] = rng.choice(vals, n)
    df = pd.DataFrame(d)
    df["ivt"] = (np.arange(n) % 2).astype(float)
    # without IVT the registry stores "thrombolysis given" as 0 and no onset-to-IVT time
    df.loc[df["ivt"] == 0, "ivt_external"] = 0.0
    df.loc[df["ivt"] == 0, "onset_to_ivt"] = np.nan
    df.loc[df["ivt"] == 1, "ivt_external"] = (np.arange((df["ivt"] == 1).sum()) % 4 == 0).astype(float)  # 25 % elsewhere
    df.loc[: n // 10, "occlusion"] = "ACA"                                  # a site without a model term
    return df


def test_every_input_has_averaging_values():
    imp = imputation_values(_registry())
    keys = [f.key for f in features() if f.type != "derived"]
    assert sorted(imp) == sorted(keys)
    assert all(len(v["values"]) == IMPUTE_K for v in imp.values())


def test_ivt_location_comes_from_thrombolysed_patients():
    imp = imputation_values(_registry())
    share = np.mean(imp["ivt_external"]["values"])
    assert abs(share - 0.25) <= 1 / IMPUTE_K            # all patients would give about 12 %


def test_no_category_without_a_model_term():
    imp = imputation_values(_registry())
    no_term = {str(o["value"]) for f in features() if f.type == "categorical"
               for o in f.spec["options"] if o.get("model") is False}
    assert no_term and not (set(imp["occlusion"]["values"]) & no_term)


def test_numeric_values_are_quantiles_not_extremes():
    reg = _registry()
    imp = imputation_values(reg)
    recent = reg[reg["year"] >= 2016]["age"]
    v = imp["age"]["values"]
    assert min(v) > recent.min() and max(v) < recent.max()
