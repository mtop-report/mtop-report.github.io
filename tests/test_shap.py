"""SHAP for every model (src/mtop/shap_all.py): the sampled method is exact where it must be."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mtop import learners as Lr
from mtop.shap_all import predictor_groups, sampled_shap, summarise


def _data(n=160, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=["a", "b", "c", "d", "e"])
    X.loc[X.index[::9], "c"] = np.nan                      # missing values with an indicator column
    lp = 0.8 * X["a"].fillna(0) - 1.2 * X["b"] + 0.5 * X["c"].fillna(0) + 0.3 * X["d"]
    y = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(float).to_numpy()
    return X, y


def test_sampled_shap_is_exact_for_a_logistic_regression():
    """For a model linear in its (standardised, imputed) inputs every permutation gives the exact
    interventional Shapley values: column contribution minus its mean over the background."""
    X, y = _data()
    f = Lr.refit("Logistic regression", {"C": 1.0}, X, y, 0, [0] * 5)
    groups = [[0], [1], [2, 3], [4]]                         # c and d switched together, as one predictor
    test, bg = X.iloc[:40], X.iloc[100:130]
    phi, base = sampled_shap(f.proba, test, bg, groups, n_perm=2, seed=3)
    c_test, _ = Lr.contributions(f, test)
    c_bg, _ = Lr.contributions(f, bg)
    want = np.column_stack([(c_test.iloc[:, g].sum(axis=1) - c_bg.iloc[:, g].sum(axis=1).mean()).to_numpy() for g in groups])
    assert np.allclose(phi, want, atol=1e-6)
    lp = f.logit(test)
    assert np.allclose(phi.sum(axis=1) + base, lp, atol=1e-6)        # efficiency


def test_sampled_shap_adds_up_for_a_non_additive_model():
    X, y = _data(seed=1)
    f = Lr.refit("MLP", {"hidden_layers": [8], "alpha": 0.01, "learning_rate": 0.01, "activation": "tanh",
                         "batch_size": 32, "ensemble": 2}, X.fillna(0), y, 0, [0] * 5)
    groups = [[0], [1], [2], [3], [4]]
    test, bg = X.fillna(0).iloc[:20], X.fillna(0).iloc[100:120]
    phi, base = sampled_shap(f.proba, test, bg, groups, n_perm=4, seed=0)
    assert np.allclose(phi.sum(axis=1) + base, f.logit(test), atol=1e-6)


def test_predictor_groups_and_shares():
    keys, groups = predictor_groups(["age", "occlusion__M1", "occlusion__M2", "nihss_adm"], "admission")
    assert keys == ["age", "occlusion", "nihss_adm"] and groups == [[0], [1, 2], [3]]
    s = summarise(np.array([[1.0, -3.0, 0.0], [-1.0, 1.0, 0.0]]), ["a", "b", "c"])
    assert list(s["importance"]) == ["b", "a", "c"] and s["share"]["b"] == 66.67 and s["importance"]["a"] == 1.0


def test_expected_gradients_add_up_on_average():
    """FT-Transformer attributions (expected gradients): over many draws they add up to the log-odds
    minus the mean log-odds of the background."""
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("rtdl_revisiting_models")
    from mtop.shap_all import FTModel

    X, y = _data(n=120, seed=3)
    info = {"n_blocks": 1, "d_block": 32, "attention_n_heads": 4, "attention_dropout": 0.0, "ffn_d_hidden_multiplier": 1.0,
            "ffn_dropout": 0.0, "residual_dropout": 0.0, "lr": 1e-3, "weight_decay": 1e-5, "epochs": 3}
    ft = FTModel(X, y, info, seed=0)
    test, bg = X.iloc[:10], X.iloc[60:100]
    phi, base = ft.expected_gradients(test, bg, n=2000, seed=1)
    assert phi.shape == (10, 5)
    p = ft.proba(test)
    lp = np.log(p / (1 - p))
    assert np.max(np.abs(phi.sum(axis=1) + base - lp)) < 0.05
