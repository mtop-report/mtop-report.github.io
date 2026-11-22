"""Unit tests of the ordinal solver (plan docs/registrations/ordinal_plan.md, section 5, step 2).
Simulated data only; no registry data."""
import numpy as np
import pytest

from mtop import ordinal as O
from mtop import reference as R


def _sim(n=600, p=5, K=7, seed=1):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, p))
    beta = rng.normal(scale=0.7, size=p)
    theta = np.sort(rng.normal(scale=1.5, size=K - 1)) + np.arange(K - 1) * 0.3
    eta = Z @ beta
    P = O.probs(theta, eta)
    y = np.array([rng.choice(K, p=row / row.sum()) for row in P])
    return Z, y, K


def test_gradient_and_hessian_match_finite_differences():
    Z, y, K = _sim(n=300, p=4)
    w = np.random.default_rng(2).uniform(0.5, 2.0, len(y))
    C = 0.7
    par = np.concatenate([np.random.default_rng(3).normal(scale=0.3, size=Z.shape[1]), O.start_thresholds(y, w, K)])
    g, H = O.grad_hess(par, Z, y, w, C, K)
    h = 1e-6
    gn = np.zeros_like(par)
    Hn = np.zeros((len(par), len(par)))
    for i in range(len(par)):
        e = np.zeros_like(par)
        e[i] = h
        gn[i] = (O.objective(par + e, Z, y, w, C, K) - O.objective(par - e, Z, y, w, C, K)) / (2 * h)
        gp, _ = O.grad_hess(par + e, Z, y, w, C, K)
        gm, _ = O.grad_hess(par - e, Z, y, w, C, K)
        Hn[:, i] = (gp - gm) / (2 * h)
    assert np.max(np.abs(g - gn)) < 1e-5 * max(1.0, np.max(np.abs(g)))
    assert np.max(np.abs(H - Hn)) < 1e-5 * max(1.0, np.max(np.abs(H)))


def test_unpenalised_fit_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.miscmodels.ordinal_model")
    Z, y, K = _sim(n=800, p=5)
    par = O.newton(Z, y, np.ones(len(y)), 1e10, K)
    mod = sm.OrderedModel(y, Z, distr="logit")
    res = mod.fit(method="newton", disp=False, maxiter=200, tol=1e-12)
    beta_sm = res.params[: Z.shape[1]]
    theta_sm = mod.transform_threshold_params(res.params)[1:-1]
    assert np.max(np.abs(par[: Z.shape[1]] - beta_sm)) < 1e-6      # plan, section 5, step 2(a)
    assert np.max(np.abs(par[Z.shape[1]:] - theta_sm)) < 1e-6


def test_two_categories_reduce_to_the_binary_reference_solver():
    Z, y7, _ = _sim(n=700, p=6)
    yb = (y7 >= 3).astype(int)
    w = np.ones(len(yb))
    for C in (0.01, 0.3, 5.0):
        par = O.newton(Z, yb, w, C, 2)
        beta = R.newton(np.hstack([Z, np.ones((len(yb), 1))]), yb.astype(float), w, C)
        assert np.max(np.abs(par[:-1] - beta[:-1])) < 1e-8
        assert abs(par[-1] + beta[-1]) < 1e-8          # threshold = minus the intercept


def test_row_order_does_not_change_the_fit():
    Z, y, K = _sim(n=500, p=4)
    w = np.ones(len(y))
    perm = np.random.default_rng(9).permutation(len(y))
    a = O.newton(Z, y, w, 0.5, K)
    b = O.newton(Z[perm], y[perm], w, 0.5, K)
    assert np.max(np.abs(a - b)) < 1e-9


def test_weights_equal_copies():
    Z, y, K = _sim(n=300, p=3)
    w = np.random.default_rng(4).integers(1, 4, len(y)).astype(float)
    a = O.newton(Z, y, w, 0.8, K)
    rows = np.repeat(np.arange(len(y)), w.astype(int))
    b = O.newton(Z[rows], y[rows], np.ones(len(rows)), 0.8, K)
    assert np.max(np.abs(a - b)) < 1e-8


def test_probabilities_are_a_distribution_and_thresholds_ordered():
    Z, y, K = _sim(n=400, p=4)
    par = O.newton(Z, y, np.ones(len(y)), 1.0, K)
    th = par[Z.shape[1]:]
    assert np.all(np.diff(th) > 0)
    P = O.probs(th, Z @ par[: Z.shape[1]])
    assert np.allclose(P.sum(1), 1.0) and P.min() >= 0


def test_threshold_refit_at_optimum_is_unchanged():
    Z, y, _ = _sim(400, 3, seed=11)
    par = O.newton(Z, y, np.ones(len(y)), 0.5, 7)
    th = O.fit_thresholds(Z @ par[:3], y, 7)
    assert np.max(np.abs(th - par[3:])) < 1e-7


def test_calibration_slope_matches_statsmodels():
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    Z, y, _ = _sim(500, 2, seed=12)
    lp = 0.7 * (Z @ np.array([1.0, -0.5]))
    s, th = O.calibration_slope(lp, y, 7)
    mod = OrderedModel(y, lp[:, None], distr="logit")
    res = mod.fit(method="newton", disp=False, maxiter=200, tol=1e-12)
    assert abs(s - res.params[0]) < 1e-6
    assert np.max(np.abs(th - mod.transform_threshold_params(res.params)[1:-1])) < 1e-6


def test_threshold_shift_is_the_likelihood_maximum():
    from scipy.optimize import minimize_scalar
    Z, y, _ = _sim(500, 2, seed=13)
    par = O.newton(Z, y, np.ones(len(y)), 1e10, 7)
    eta, th = Z @ par[:2] + 0.4, par[2:]          # a mis-calibrated linear predictor
    d = O.threshold_shift(eta, th, y, 7)
    r = minimize_scalar(lambda v: O.nll(th + v, np.zeros(0), np.zeros((len(y), 0)), y, np.ones(len(y)), eta),
                        bracket=(-1, 1), tol=1e-12)
    assert abs(d - r.x) < 1e-6 and abs(d - 0.4) < 0.3


def test_fused_and_numpy_arithmetic_agree():
    if not O.HAVE_NUMBA:
        return
    Z, y, K = _sim(400, 4, seed=21)
    w = np.random.default_rng(3).integers(1, 4, len(y)).astype(float)
    par = np.r_[np.array([0.3, -0.2, 0.1, 0.5]), O.start_thresholds(y, w, K) + 0.2]
    try:
        O.USE_NUMBA = True
        g1, H1 = O.grad_hess(par, Z, y, w, 0.7, K, 0.1, 1.0)
        f1 = O.objective(par, Z, y, w, 0.7, K, 0.1)
        O.USE_NUMBA = False
        g0, H0 = O.grad_hess(par, Z, y, w, 0.7, K, 0.1, 1.0)
        f0 = O.objective(par, Z, y, w, 0.7, K, 0.1)
    finally:
        O.USE_NUMBA = O.HAVE_NUMBA
    assert np.max(np.abs(g1 - g0)) < 1e-9 * max(1, np.max(np.abs(g0)))
    assert np.max(np.abs(H1 - H0)) < 1e-9 * max(1, np.max(np.abs(H0)))
    assert abs(f1 - f0) < 1e-9 * abs(f0)


def _registry_like(n=900, seed=3):
    import pandas as pd
    rng = np.random.default_rng(seed)
    occ = rng.choice(["M1", "basilar", "basilar_vertebral"], size=n, p=[0.8, 0.185, 0.015])
    X = pd.DataFrame({
        "age": rng.normal(70, 12, n).round(),
        "nihss_adm": rng.integers(0, 40, n).astype(float),
        "occlusion__M1": (occ == "M1").astype(float),
        "occlusion__basilar": (occ == "basilar").astype(float),
        "occlusion__basilar_vertebral": (occ == "basilar_vertebral").astype(float),
        "glucose": rng.normal(7.5, 2.5, n).round(1).clip(3, 25),
        "flag": rng.integers(0, 2, n).astype(float),
    })
    X.loc[rng.random(n) < 0.05, ["occlusion__M1", "occlusion__basilar", "occlusion__basilar_vertebral"]] = np.nan
    X.loc[rng.random(n) < 0.2, "glucose"] = np.nan
    eta = 0.04 * (X["age"] - 70) + 0.1 * X["nihss_adm"] - 0.5 * X["occlusion__basilar"].fillna(0)
    P = O.probs(np.array([-2.5, -1.5, -0.8, 0.0, 0.9, 2.0]), eta.to_numpy())
    y = np.array([rng.choice(7, p=r / r.sum()) for r in P])
    return X, y


def test_export_and_browser_reproduce_the_model(tmp_path):
    import json
    import shutil
    import subprocess
    from pathlib import Path
    X, y = _registry_like()
    m = O.fit(X, y, 7, 5, repeats=1)
    spec = m.export(shift=0.137)
    rows = np.vstack([X.sample(200, random_state=2).to_numpy(float), np.full(X.shape[1], np.nan)])
    import pandas as pd
    Xr = pd.DataFrame(rows, columns=X.columns)
    lp, P, phi = O.eval_export(spec, Xr)
    assert np.max(np.abs(lp - m.lp(Xr))) < 1e-10
    assert np.max(np.abs(P - O.probs(m.theta + 0.137, m.lp(Xr)))) < 1e-12
    assert np.max(np.abs(phi - m.contributions(Xr).to_numpy())) < 1e-10
    assert np.max(np.abs(phi.sum(1) - lp)) < 1e-10
    node = shutil.which("node")
    if node is None:
        return
    here = Path(__file__).parent
    inp, out = tmp_path / "in.json", tmp_path / "out.json"
    inp.write_text(json.dumps({"model": {"kind": "ordinal", "spec": spec},
                               "rows": [[None if np.isnan(v) else float(v) for v in r] for r in rows]}))
    subprocess.run([node, str(here / "node_ordinal.js"), str(inp), str(out)], check=True)
    res = json.loads(out.read_text())
    assert np.max(np.abs(np.array([r["lp"] for r in res]) - lp)) < 1e-9
    assert np.max(np.abs(np.array([r["p"] for r in res]) - P)) < 1e-9
    assert np.max(np.abs(np.array([r["phi"] for r in res]) - phi)) < 1e-9
    assert max(abs(r["base"]) for r in res) == 0.0


def test_browser_ordinal_measures_match_python(tmp_path):
    """web/metrics.js ordinalSummary (the validation tool) against ordinal_run.dist_measures."""
    import json
    import shutil
    import subprocess
    from pathlib import Path

    from mtop import ordinal_run as OR
    node = shutil.which("node")
    if node is None:
        return
    rng = np.random.default_rng(5)
    P = rng.dirichlet(np.ones(7) * 0.8, 300)
    y = np.array([rng.choice(7, p=p) for p in P])
    root = Path(__file__).parent.parent
    script = tmp_path / "run.js"
    script.write_text(
        "const M = require(%r); const fs = require('fs');" % str(root / "web" / "metrics.js")
        + "const j = JSON.parse(fs.readFileSync(process.argv[2]));"
        + "fs.writeFileSync(process.argv[3], JSON.stringify(M.ordinalSummary(j.y, j.P, j.uw, 10)));")
    inp, out = tmp_path / "in.json", tmp_path / "out.json"
    inp.write_text(json.dumps({"y": y.tolist(), "P": P.tolist(), "uw": OR.UW.tolist()}))
    subprocess.run([node, str(script), str(inp), str(out)], check=True)
    js = json.loads(out.read_text())
    py = OR.dist_measures(y, P)
    for k in ("rps", "ologloss", "genc_uw", "genc_grade", "uw_pred", "uw_obs"):
        assert abs(js[k] - py[k]) < 1e-12, k
    for d in ("mrs01", "mrs02", "mrs03", "death"):
        assert abs(js[d]["auroc"] - py[d]["auroc"]) < 1e-12, d
        assert abs(js[d]["citl"] - py[d]["citl"]) < 1e-4, d
        assert abs(js[d]["slope"] - py[d]["slope"]) < 1e-4, d
    assert np.max(np.abs(np.array(js["grade_pred"]) - np.array(py["grade_pred"]))) < 1e-12
