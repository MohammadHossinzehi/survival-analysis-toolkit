import numpy as np
import pytest

from survkit import (CoxPH, KaplanMeier, WeibullFitter, ExponentialFitter, brier_score,
                     concordance_index, integrated_brier_score, simulate_weibull_ph)


def brute_c_index(t, s, e):
    num = den = 0.0
    for i in range(len(t)):
        if not e[i]:
            continue
        for j in range(len(t)):
            if t[i] < t[j] or (t[i] == t[j] and not e[j] and i != j):
                den += 1
                num += 1.0 if s[i] < s[j] else 0.5 if s[i] == s[j] else 0.0
    return num / den


def test_c_index_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    t = rng.integers(1, 15, 150).astype(float)
    e = rng.random(150) < 0.6
    s = rng.integers(0, 6, 150).astype(float)
    assert concordance_index(t, s, e) == pytest.approx(brute_c_index(t, s, e), rel=1e-12)


def test_c_index_extremes():
    t = np.arange(1, 11, dtype=float)
    assert concordance_index(t, t) == 1.0
    assert concordance_index(t, -t) == 0.0
    assert concordance_index(t, np.zeros(10)) == 0.5


def test_brier_without_censoring_is_plain_mse():
    rng = np.random.default_rng(1)
    t = rng.exponential(size=300)
    p = rng.random(300)
    tau = 0.7
    expected = np.mean((p - (t > tau)) ** 2)
    assert brier_score(t, np.ones(300), p, tau) == pytest.approx(expected, rel=1e-12)


def test_ipcw_brier_is_close_to_uncensored_truth():
    # With independent censoring, IPCW should recover the full data Brier score.
    X, t, e, T = simulate_weibull_ph(20000, [0.8], censor_rate=0.08, rng=2)
    tau = 6.0
    true_s = np.exp(-(tau / 10.0) ** 1.5 * np.exp(0.8 * X[:, 0]))
    full = np.mean((true_s - (T > tau)) ** 2)
    ipcw = brier_score(t, e, true_s, tau)
    assert ipcw == pytest.approx(full, abs=0.006)


def test_cox_beats_km_on_integrated_brier():
    X, t, e, _ = simulate_weibull_ph(1500, [1.0, -0.8], rng=3)
    train, test = slice(0, 1000), slice(1000, None)
    times = np.quantile(t[test][e[test] == 1], [0.1, 0.25, 0.5, 0.75])
    grid = np.linspace(times[0], times[-1], 20)
    cox = CoxPH().fit(X[train], t[train], e[train])
    S_cox = cox.predict_survival_function(X[test], grid)
    km = KaplanMeier().fit(t[train], e[train])
    S_km = np.repeat(km.survival_at(grid)[:, None], X[test].shape[0], axis=1)
    ibs_cox = integrated_brier_score(t[test], e[test], S_cox, grid)
    ibs_km = integrated_brier_score(t[test], e[test], S_km, grid)
    assert ibs_cox < ibs_km - 0.02


def test_weibull_recovers_parameters():
    rng = np.random.default_rng(4)
    shape, scale = 1.8, 12.0
    T = scale * rng.weibull(shape, 5000)
    C = rng.exponential(25.0, 5000)
    wf = WeibullFitter().fit(np.minimum(T, C), T <= C)
    assert wf.shape == pytest.approx(shape, rel=0.05)
    assert wf.scale == pytest.approx(scale, rel=0.03)
    assert wf.shape_ci[0] < shape < wf.shape_ci[1]
    assert wf.exponential_lr_test[1] < 1e-10


def test_weibull_mle_is_stationary():
    rng = np.random.default_rng(5)
    t = rng.weibull(0.8, 400) * 3 + 0.01
    e = rng.random(400) < 0.75
    wf = WeibullFitter().fit(t, e)
    base = wf._loglik(np.log(wf.shape), np.log(wf.scale), t, e)
    for da, db in [(1e-4, 0), (-1e-4, 0), (0, 1e-4), (0, -1e-4)]:
        assert wf._loglik(np.log(wf.shape) + da, np.log(wf.scale) + db, t, e) <= base


def test_exponential_on_exponential_data():
    rng = np.random.default_rng(6)
    t = rng.exponential(4.0, 3000)
    ef = ExponentialFitter().fit(t)
    assert ef.rate == pytest.approx(0.25, rel=0.05)
    wf = WeibullFitter().fit(t)
    assert wf.shape == pytest.approx(1.0, abs=0.05)
    assert wf.exponential_lr_test[1] > 0.01
