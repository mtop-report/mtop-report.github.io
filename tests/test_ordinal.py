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
