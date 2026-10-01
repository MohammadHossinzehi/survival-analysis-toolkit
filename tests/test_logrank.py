import numpy as np
import pytest

from survkit import CoxPH, load_gehan, logrank_test, logrank_trend_test, simulate_weibull_ph


def test_gehan_logrank_textbook_value():
    t, e, g = load_gehan()
    res = logrank_test(t, e, g)
    # R: survdiff(Surv(time, status) ~ group) gives Chisq = 16.8, p = 4.17e-05
    assert res.statistic == pytest.approx(16.7929, abs=1e-4)
    assert res.p_value == pytest.approx(4.169e-5, rel=1e-3)
    np.testing.assert_allclose(res.observed, [9, 21])
    np.testing.assert_allclose(res.expected, [19.25, 10.75], atol=5e-3)


def test_observed_minus_expected_sums_to_zero():
    t, e, g = load_gehan()
    res = logrank_test(t, e, g)
    assert res.observed.sum() == pytest.approx(res.expected.sum())


def test_logrank_equals_cox_score_test_without_ties():
    # Without tied event times the log rank statistic is exactly the score
    # test of a Cox model with a group indicator.
    X, t, e, _ = simulate_weibull_ph(250, [0.6], rng=11)
    group = (X[:, 0] > 0).astype(int)
    lr = logrank_test(t, e, group).statistic
    cox = CoxPH(ties="breslow").fit(group, t, e)
    assert lr == pytest.approx(cox.tests["score"][0], rel=1e-10)


def test_null_p_values_are_roughly_uniform():
    rng = np.random.default_rng(5)
    pvals = []
    for _ in range(1000):
        t = rng.exponential(size=60)
        c = rng.exponential(2, size=60)
        g = rng.integers(0, 3, 60)
        pvals.append(logrank_test(np.minimum(t, c), t <= c, g).p_value)
    pvals = np.array(pvals)
    # the chi square approximation is slightly liberal at n = 60, hence the band
    assert 0.03 < np.mean(pvals < 0.05) < 0.09
    assert 0.4 < np.mean(pvals) < 0.6


@pytest.mark.parametrize("w", ["logrank", "wilcoxon", "tarone_ware", "peto"])
def test_weightings_detect_strong_effect(w):
    t, e, g = load_gehan()
    assert logrank_test(t, e, g, weighting=w).p_value < 0.01


def test_fleming_harrington_zero_zero_is_logrank():
    t, e, g = load_gehan()
    a = logrank_test(t, e, g).statistic
    b = logrank_test(t, e, g, weighting="fleming_harrington", p=0, q=0).statistic
    assert a == pytest.approx(b)


def test_trend_test_beats_omnibus_for_monotone_effect():
    rng = np.random.default_rng(2)
    dose = rng.integers(0, 4, 400)
    X, t, e, _ = simulate_weibull_ph(400, [0.15], X=dose[:, None].astype(float), rng=3)
    omni = logrank_test(t, e, dose)
    z, p = logrank_trend_test(t, e, dose)
    assert z > 0  # higher dose, more events than expected
    assert p < omni.p_value


def test_requires_two_groups():
    with pytest.raises(ValueError):
        logrank_test([1, 2, 3], [1, 1, 1], [0, 0, 0])
    with pytest.raises(ValueError):
        logrank_test([1, 2, 3], [1, 1, 1], [0, 1, 0], weighting="nope")
