import numpy as np
import pytest

from survkit import KaplanMeier, NelsonAalen, event_table, load_gehan

# Reference numbers for the 6MP arm of the Gehan trial, as printed by R's
# survfit with the log(-log S) interval and reproduced in most textbooks.
GEHAN_6MP = [
    # time, n.risk, n.event, survival, std.err, lower, upper
    (6, 21, 3, 0.8571, 0.0764, 0.6197, 0.9516),
    (7, 17, 1, 0.8067, 0.0869, 0.5631, 0.9228),
    (10, 15, 1, 0.7529, 0.0963, 0.5032, 0.8894),
    (13, 12, 1, 0.6902, 0.1068, 0.4316, 0.8491),
    (16, 11, 1, 0.6275, 0.1141, 0.3675, 0.8049),
    (22, 7, 1, 0.5378, 0.1282, 0.2678, 0.7468),
    (23, 6, 1, 0.4482, 0.1346, 0.1881, 0.6801),
]


@pytest.fixture
def six_mp():
    t, e, g = load_gehan()
    return t[g == 0], e[g == 0]


def test_gehan_kaplan_meier_table(six_mp):
    km = KaplanMeier().fit(*six_mp)
    for time, n, d, s, se, lo, hi in GEHAN_6MP:
        i = np.searchsorted(km.timeline, time)
        assert km.table.at_risk[i] == n
        assert km.table.events[i] == d
        assert km.survival[i] == pytest.approx(s, abs=5e-5)
        assert np.sqrt(km.variance[i]) == pytest.approx(se, abs=5e-5)
        assert km.ci_lower[i] == pytest.approx(lo, abs=5e-5)
        assert km.ci_upper[i] == pytest.approx(hi, abs=5e-5)


def test_gehan_medians():
    t, e, g = load_gehan()
    assert KaplanMeier().fit(t[g == 0], e[g == 0]).median == 23
    assert KaplanMeier().fit(t[g == 1], e[g == 1]).median == 8
    lo, hi = KaplanMeier().fit(t[g == 0], e[g == 0]).median_confidence_interval()
    assert lo == 13 and hi == np.inf


def test_no_censoring_reduces_to_empirical_survival():
    rng = np.random.default_rng(0)
    t = rng.exponential(size=200)
    km = KaplanMeier().fit(t)
    grid = np.linspace(0, t.max() * 1.1, 97)
    empirical = np.array([(t > u).mean() for u in grid])
    np.testing.assert_allclose(km.survival_at(grid), empirical, atol=1e-12)


def test_left_limit_and_step_shape(six_mp):
    km = KaplanMeier().fit(*six_mp)
    assert km.survival_at(0) == 1.0
    assert km.survival_at(6, left_limit=True) == 1.0
    assert km.survival_at(6) == pytest.approx(18 / 21)
    assert km.survival_at(6.999) == pytest.approx(18 / 21)


def test_left_truncation_matches_brute_force():
    rng = np.random.default_rng(3)
    n = 120
    t = rng.exponential(5, n)
    entry = np.where(rng.random(n) < 0.4, t * rng.random(n), 0.0)
    e = rng.random(n) < 0.7
    km = KaplanMeier().fit(t, e, entry)
    s = 1.0
    for u in np.unique(t[e]):
        risk = np.sum((entry < u) & (t >= u))
        s *= 1 - np.sum((t == u) & e) / risk
        assert km.survival_at(u) == pytest.approx(s, rel=1e-12)


def test_event_table_counts():
    tab = event_table([1, 2, 2, 3, 3, 3], [1, 0, 1, 1, 1, 0])
    np.testing.assert_array_equal(tab.time, [1, 2, 3])
    np.testing.assert_array_equal(tab.at_risk, [6, 5, 3])
    np.testing.assert_array_equal(tab.events, [1, 1, 2])
    np.testing.assert_array_equal(tab.censored, [0, 1, 1])


def test_restricted_mean_matches_integral(six_mp):
    km = KaplanMeier().fit(*six_mp)
    tau = 30.0
    grid = np.linspace(0, tau, 300_001)
    numeric = np.sum(km.survival_at(grid[:-1]) * np.diff(grid))
    rmst, se = km.restricted_mean(tau)
    assert rmst == pytest.approx(numeric, rel=1e-4)
    assert se > 0


def test_last_observation_event_gives_zero_survival():
    km = KaplanMeier().fit([1, 2, 3], [1, 1, 1])
    assert km.survival[-1] == 0.0
    assert km.ci_lower[-1] == 0.0 and km.ci_upper[-1] == 0.0
    assert km.median == 2


def test_nelson_aalen_increments():
    na = NelsonAalen().fit([1, 2, 2, 3, 4], [1, 1, 1, 0, 1])
    expected = np.cumsum([1 / 5, 2 / 4, 0, 1 / 1])
    np.testing.assert_allclose(na.cumulative_hazard, expected)
    assert na.cumulative_hazard_at(0.5) == 0.0


def test_input_validation():
    with pytest.raises(ValueError):
        KaplanMeier().fit([1, -2, 3])
    with pytest.raises(ValueError):
        KaplanMeier().fit([1, 2, 3], [1, 0])
    with pytest.raises(ValueError):
        KaplanMeier().fit([1, 2, 3], [1, 2, 0])
    with pytest.raises(ValueError):
        KaplanMeier().fit([1, 2, 3], entry=[0, 3, 0])
    with pytest.raises(RuntimeError):
        KaplanMeier().survival_at(1.0)
