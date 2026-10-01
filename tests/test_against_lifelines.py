"""Optional cross check against lifelines. Skipped when it is not installed."""
import numpy as np
import pytest

lifelines = pytest.importorskip("lifelines")
pd = pytest.importorskip("pandas")

from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter  # noqa: E402
from lifelines import WeibullFitter as LWeibull  # noqa: E402
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test  # noqa: E402

import survkit as sk  # noqa: E402


@pytest.fixture(scope="module")
def data():
    X, t, e, _ = sk.simulate_weibull_ph(300, [0.7, -0.5, 0.2], rng=2)
    t = np.round(t, 1) + 0.1  # plenty of ties
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["T"], df["E"] = t, e
    df["g"] = np.random.default_rng(0).integers(0, 3, len(t))
    return X, t, e, df


def test_kaplan_meier_and_nelson_aalen(data):
    _, t, e, _ = data
    grid = np.linspace(0, t.max(), 60)
    km = sk.KaplanMeier().fit(t, e)
    ref = KaplanMeierFitter().fit(t, e)
    np.testing.assert_allclose(km.survival_at(grid), ref.survival_function_at_times(grid), atol=1e-12)
    na = sk.NelsonAalen().fit(t, e)
    ref = NelsonAalenFitter(nelson_aalen_smoothing=False).fit(t, e)
    np.testing.assert_allclose(na.cumulative_hazard_at(grid), ref.cumulative_hazard_at_times(grid), atol=1e-12)


@pytest.mark.parametrize("w, kw", [("logrank", {}), ("wilcoxon", {}), ("tarone_ware", {}),
                                   ("peto", {}), ("fleming_harrington", {"p": 1, "q": 1})])
def test_weighted_logrank(data, w, kw):
    _, t, e, df = data
    ours = sk.logrank_test(t, e, df["g"].values, weighting=w, **kw).statistic
    theirs = multivariate_logrank_test(t, df["g"], e, weightings=None if w == "logrank" else w.replace("_", "-"), **kw)
    assert ours == pytest.approx(theirs.test_statistic, rel=1e-9)


def test_cox_efron(data):
    X, t, e, df = data
    ours = sk.CoxPH().fit(X, t, e)
    ref = CoxPHFitter().fit(df.drop(columns="g"), "T", "E")
    np.testing.assert_allclose(ours.coef, ref.params_.values, atol=1e-5)
    np.testing.assert_allclose(ours.se, ref.standard_errors_.values, atol=1e-5)
    assert ours.log_likelihood == pytest.approx(ref.log_likelihood_, abs=1e-6)
    assert ours.concordance == pytest.approx(ref.concordance_index_, abs=1e-12)


def test_cox_stratified(data):
    X, t, e, df = data
    ours = sk.CoxPH().fit(X, t, e, strata=df["g"].values)
    ref = CoxPHFitter().fit(df, "T", "E", strata=["g"])
    np.testing.assert_allclose(ours.coef, ref.params_.values, atol=1e-5)


def test_ph_test_and_weibull():
    X, t, e, _ = sk.simulate_weibull_ph(400, [0.5, -0.3], rng=5)
    df = pd.DataFrame(X, columns=["a", "b"])
    df["T"], df["E"] = t, e
    ours = sk.CoxPH().fit(X, t, e)
    ref = CoxPHFitter().fit(df, "T", "E")
    for transform in ("rank", "identity", "log", "km"):
        theirs = proportional_hazard_test(ref, df, time_transform=transform).test_statistic
        got = [v[0] for v in ours.check_proportional_hazards(transform).values()]
        np.testing.assert_allclose(got, np.asarray(theirs), rtol=1e-4)
    wf = sk.WeibullFitter().fit(t, e)
    lw = LWeibull().fit(t, e)
    assert wf.shape == pytest.approx(lw.rho_, rel=1e-5)
    assert wf.scale == pytest.approx(lw.lambda_, rel=1e-5)
