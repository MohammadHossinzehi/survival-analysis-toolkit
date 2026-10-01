import warnings

import numpy as np
import pytest

from survkit import ConvergenceWarning, CoxPH, load_gehan, simulate_weibull_ph


@pytest.fixture(scope="module")
def gehan():
    return load_gehan()


@pytest.mark.parametrize("ties, coef, se, lr, wald, score", [
    # R: coxph(Surv(time, status) ~ placebo, data, ties = ...)
    ("efron", 1.5721, 0.4124, 16.35, 14.53, 17.25),
    ("breslow", 1.5092, 0.4096, 15.21, 13.58, 15.93),
])
def test_gehan_reference(gehan, ties, coef, se, lr, wald, score):
    t, e, g = gehan
    m = CoxPH(ties=ties).fit(g, t, e)
    assert m.coef[0] == pytest.approx(coef, abs=1e-4)
    assert m.se[0] == pytest.approx(se, abs=1e-4)
    assert m.tests["likelihood_ratio"][0] == pytest.approx(lr, abs=6e-3)
    assert m.tests["wald"][0] == pytest.approx(wald, abs=6e-3)
    assert m.tests["score"][0] == pytest.approx(score, abs=6e-3)
    assert m.converged


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_gradient_and_hessian_match_finite_differences(ties):
    X, t, e, _ = simulate_weibull_ph(80, [0.5, -1.0, 0.3], rng=4)
    t = np.round(t)  # force lots of ties so the Efron terms matter
    m = CoxPH(ties=ties)
    prepared = m._prepare(X - X.mean(0), t, e.astype(bool), np.zeros(80))
    beta = np.array([0.2, -0.4, 0.1])
    ll, g, H = m._loglik(beta, prepared)
    h = 1e-6
    for i in range(3):
        step = np.zeros(3)
        step[i] = h
        lp, gp, _ = m._loglik(beta + step, prepared)
        lm, gm, _ = m._loglik(beta - step, prepared)
        assert (lp - lm) / (2 * h) == pytest.approx(g[i], rel=1e-6, abs=1e-6)
        np.testing.assert_allclose((gp - gm) / (2 * h), H[:, i], rtol=1e-5, atol=1e-6)


def test_recovers_true_coefficients():
    beta = np.array([0.8, -0.5, 0.0, 0.3])
    X, t, e, _ = simulate_weibull_ph(4000, beta, rng=7)
    m = CoxPH().fit(X, t, e)
    assert np.all(np.abs(m.coef - beta) < 4 * m.se)
    np.testing.assert_allclose(m.coef, beta, atol=0.08)


def test_efron_equals_breslow_without_ties():
    X, t, e, _ = simulate_weibull_ph(300, [0.4, 0.2], rng=1)
    a = CoxPH(ties="efron").fit(X, t, e)
    b = CoxPH(ties="breslow").fit(X, t, e)
    np.testing.assert_allclose(a.coef, b.coef, rtol=1e-10)


def test_invariance_to_covariate_shift():
    X, t, e, _ = simulate_weibull_ph(200, [0.4, -0.2], rng=9)
    a = CoxPH().fit(X, t, e)
    b = CoxPH().fit(X + np.array([100.0, -3.0]), t, e)
    np.testing.assert_allclose(a.coef, b.coef, rtol=1e-8)
    np.testing.assert_allclose(a.predict_survival_function(X[:3], [1, 5]),
                               b.predict_survival_function(X[:3] + [100.0, -3.0], [1, 5]),
                               rtol=1e-8)


def test_stratification_absorbs_different_baselines():
    rng = np.random.default_rng(8)
    parts = []
    for k, (shape, scale) in enumerate([(0.7, 3.0), (2.5, 20.0)]):
        X, t, e, _ = simulate_weibull_ph(1500, [0.6], shape=shape, scale=scale,
                                         censor_rate=0.02, rng=rng)
        parts.append((X, t, e, np.full(t.size, k)))
    X, t, e, s = (np.concatenate(p) for p in zip(*parts))
    m = CoxPH().fit(X, t, e, strata=s)
    assert m.coef[0] == pytest.approx(0.6, abs=0.06)
    with pytest.raises(ValueError):
        m.baseline_cumulative_hazard([1.0])
    early = m.baseline_cumulative_hazard([1.0], stratum=0)[0]
    late = m.baseline_cumulative_hazard([1.0], stratum=1)[0]
    assert early > late  # stratum 0 has a much higher early hazard


def test_penalizer_shrinks_and_rescues_separation():
    t = np.arange(1, 21, dtype=float)
    x = (t < 10).astype(float)  # perfectly separates early from late deaths
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        free = CoxPH().fit(x, t)
    ridge = CoxPH(penalizer=1.0).fit(x, t)
    assert ridge.converged
    assert abs(ridge.coef[0]) < abs(free.coef[0])
    assert 0 < ridge.coef[0] < 5


def test_baseline_and_prediction_consistency(gehan):
    t, e, g = gehan
    m = CoxPH(ties="breslow").fit(g, t, e)
    times = np.array([5.0, 10.0, 20.0])
    S = m.predict_survival_function(np.array([[0.0], [1.0]]), times)
    assert S.shape == (3, 2)
    assert np.all(S[:, 1] < S[:, 0])  # placebo does worse
    np.testing.assert_allclose(np.log(S[:, 1]) / np.log(S[:, 0]), np.exp(m.coef[0]))
    assert np.all(np.diff(S, axis=0) <= 0)
    med = m.predict_median(np.array([[0.0], [1.0]]))
    assert med[1] < med[0]


def test_martingale_residuals_sum_to_zero(gehan):
    t, e, g = gehan
    m = CoxPH(ties="breslow").fit(g, t, e)
    assert m.martingale_residuals().sum() == pytest.approx(0.0, abs=1e-8)


def test_schoenfeld_residuals_sum_to_zero_at_mle():
    X, t, e, _ = simulate_weibull_ph(200, [0.5, -0.5], rng=12)
    m = CoxPH(ties="breslow").fit(X, t, e)
    _, r = m.schoenfeld_residuals()
    np.testing.assert_allclose(r.sum(axis=0), 0.0, atol=1e-7)


def test_ph_check_flags_time_varying_effect():
    rng = np.random.default_rng(21)
    n = 1500
    x = rng.integers(0, 2, n).astype(float)
    # group 1: hazard 3x higher before t = 1, 3x lower afterwards
    t0 = rng.exponential(1.0, n) / np.where(x == 1, 3.0, 1.0)
    late = t0 > 1.0
    t = np.where(late & (x == 1), 1.0 + (t0 - 1.0) * 9.0, t0)
    good_X, gt, ge, _ = simulate_weibull_ph(n, [0.7], rng=22)
    bad = CoxPH().fit(x, t).check_proportional_hazards("km")["x0"][1]
    fine = CoxPH().fit(good_X, gt, ge).check_proportional_hazards("km")["x0"][1]
    assert bad < 1e-6
    assert fine > 0.01


def test_summary_mentions_everything(gehan):
    t, e, g = gehan
    text = CoxPH().fit(g, t, e, feature_names=["placebo"]).summary()
    for word in ("placebo", "exp(coef)", "likelihood ratio", "concordance"):
        assert word in text


def test_no_events_is_an_error():
    with pytest.raises(ValueError):
        CoxPH().fit(np.ones(3), [1, 2, 3], [0, 0, 0])
