import math

import pytest

from survkit._stats import chi2_sf, norm_cdf, norm_ppf


@pytest.mark.parametrize("x, df, expected", [
    (3.841458820694124, 1, 0.05),
    (5.991464547107979, 2, 0.05),
    (6.634896601021214, 1, 0.01),
    (11.070497693516351, 5, 0.05),
    (0.454936423119572, 1, 0.5),
    (100.0, 3, 1.8017389404140525e-21),
])
def test_chi2_sf_reference_values(x, df, expected):
    assert chi2_sf(x, df) == pytest.approx(expected, rel=1e-9)


def test_chi2_sf_edges():
    assert chi2_sf(0.0, 4) == 1.0
    assert chi2_sf(-1.0, 4) == 1.0


def test_chi2_two_df_closed_form():
    for x in (0.1, 1.0, 7.5, 40.0):
        assert chi2_sf(x, 2) == pytest.approx(math.exp(-x / 2), rel=1e-12)


@pytest.mark.parametrize("p", [1e-10, 0.001, 0.025, 0.3, 0.5, 0.8, 0.975, 0.999999])
def test_norm_ppf_inverts_cdf(p):
    assert norm_cdf(norm_ppf(p)) == pytest.approx(p, rel=1e-12)


def test_norm_ppf_known():
    assert norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-12)
